import pickle
import time
import torch
import numpy as np
import random
from env.task_env import TaskEnv
from attention import AttentionNet
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical
from utils.evidence_recorder import (
    EpisodeEvidenceBuffer,
    build_candidate_records,
    get_task_state,
    to_python,
)
from utils.bias_manager import (
    bias_snapshot_to_tensor,
    EXPLICIT_BIAS_FEATURES,
    normalize_bias_snapshot,
)


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def zero_padding(x, padding_size, length):
    pad = torch.nn.ZeroPad2d((0, 0, 0, padding_size - length))
    x = pad(x)
    return x


class Worker:
    def __init__(self, mete_agent_id, local_network, local_baseline, global_step,
                 device='cuda', save_image=False, seed=None, env_params=None,
                 bias_config=None):

        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image
        if env_params is None:
            env_params = [EnvParams.SPECIES_AGENTS_RANGE, EnvParams.SPECIES_RANGE, EnvParams.TASKS_RANGE]
        self.env = TaskEnv(*env_params, EnvParams.TRAIT_DIM, EnvParams.DECISION_DIM, seed=seed, plot_figure=save_image)
        self.baseline_env = copy.deepcopy(self.env)
        self.local_baseline = local_baseline
        self.local_net = local_network
        self.experience = {idx:[] for idx in range(9)}
        self.episode_number = None
        self.perf_metrics = {}
        self.p_rnn_state = {}
        self.max_time = EnvParams.MAX_TIME
        self.evidence_payload = []
        self.last_episode_evidence = None
        self.bias_config = normalize_bias_snapshot(bias_config)

    def run_episode(self, training=True, sample=False, max_waiting=False, episode_id=None):
        buffer_dict = {idx:[] for idx in range(9)}
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0
        self.last_episode_evidence = None
        collect_evidence = (
            EvidenceParams.ENABLE_EVIDENCE_LOGGING and bool(training))
        evidence_buffer = (
            EpisodeEvidenceBuffer(
                self.global_step if episode_id is None else episode_id)
            if collect_evidence else None)
        open_task_counts = []
        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                random.shuffle(release_agents[0])
                finished_task = []
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting))
                    block_flag = mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    if training:
                        task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    explicit_features, bias_params = self._get_bias_tensors(
                        agent_id, task_info.shape[1])
                    bias_kwargs = {}
                    if self.bias_config['apply_bias']:
                        bias_kwargs = {
                            'explicit_features': explicit_features,
                            'bias_params': bias_params,
                        }
                    if collect_evidence:
                        (probs, _, _, raw_logits, logit_bias,
                         biased_logits) = self.local_net(
                            task_info,
                            total_agents,
                            mask,
                            index,
                            return_details='all',
                            **bias_kwargs)
                    else:
                        probs, _ = self.local_net(
                            task_info, total_agents, mask, index, **bias_kwargs)
                    if training:
                        action = Categorical(probs).sample()
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                    decision_record = None
                    if collect_evidence:
                        try:
                            decision_record = self._build_decision_evidence(
                                agent_id,
                                action.item(),
                                probs,
                                raw_logits,
                                mask,
                                logit_bias,
                                biased_logits,
                                explicit_features)
                        except Exception:
                            # Evidence is best-effort and must never stop training.
                            decision_record = None
                    r, doable, f_t = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    finished_task.append(f_t)
                    if training and doable:
                        buffer_dict[0] += total_agents
                        buffer_dict[1] += task_info
                        buffer_dict[2] += action.unsqueeze(0)
                        buffer_dict[3] += mask
                        buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # reward
                        buffer_dict[5] += index
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                        buffer_dict[7] += explicit_features
                        buffer_dict[8] += bias_params
                        current_action_index += 1
                        if collect_evidence and decision_record is not None:
                            evidence_buffer.record_decision(decision_record)
                            open_task_counts.append(
                                decision_record['current_open_task_count'])
                self.env.finished = self.env.check_finished()
                decision_step += 1

        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time)

        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]
        perf_metrics['makespan'] = [self.env.current_time]
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [self.env.get_efficiency()]
        if collect_evidence:
            try:
                summary = self._build_episode_evidence_summary(
                    terminal_reward,
                    finished_tasks,
                    perf_metrics,
                    open_task_counts,
                    len(evidence_buffer.decisions))
                self.last_episode_evidence = evidence_buffer.record_episode_end(summary)
            except Exception:
                # Keep the rollout usable even if optional evidence is incomplete.
                self.last_episode_evidence = None
        return terminal_reward, buffer_dict, perf_metrics

    def _get_bias_tensors(self, agent_id, action_count):
        feature_values = self.env.get_explicit_bias_feature_matrix(
            agent_id, action_count=action_count)
        explicit_features = torch.tensor(
            feature_values,
            dtype=torch.float,
            device=self.device).unsqueeze(0)
        bias_params = bias_snapshot_to_tensor(
            self.bias_config,
            device=self.device,
            dtype=torch.float)
        return explicit_features, bias_params

    def _build_decision_evidence(self, agent_id, chosen_action, probs, logits,
                                 mask, logit_bias=None,
                                 biased_logits=None, explicit_features=None):
        """Capture a lightweight decision snapshot before the environment mutates."""
        agent = self.env.agent_dic[agent_id]
        action_count = self.env.tasks_num + 1
        unfinished = self.env.get_unfinished_tasks()
        completed_count = sum(
            bool(task['finished']) for task in self.env.task_dic.values())
        chosen_task = (
            self.env.task_dic[chosen_action - 1] if chosen_action > 0 else None)
        candidates = build_candidate_records(
            self.env,
            agent_id,
            chosen_action,
            probs[0],
            logits[0],
            mask[0],
            EvidenceParams.MAX_CANDIDATES_PER_DECISION,
            top_k=EvidenceParams.TOP_K_CANDIDATES,
            low_threshold=EvidenceParams.LOW_CAPABILITY_THRESHOLD,
            logit_bias=(
                logit_bias[0] if logit_bias is not None else None),
            biased_logits=(
                biased_logits[0] if biased_logits is not None else None),
            explicit_features=(
                explicit_features[0]
                if explicit_features is not None else None),
            feature_names=EXPLICIT_BIAS_FEATURES)
        capability_match = self.env.get_capability_match_action_vector(
            agent_id, action_count=action_count)
        explicit_feature_values = to_python(
            explicit_features[0, :action_count]
            if explicit_features is not None else None)
        return {
            'train_step': None,
            'current_agent_id': agent_id,
            'current_agent_species': agent['species'],
            'current_agent_skill_vector': to_python(agent['abilities']),
            'current_open_task_count': int(sum(unfinished)),
            'current_completed_task_count': completed_count,
            'remaining_task_count': self.env.tasks_num - completed_count,
            'chosen_task_id': (
                chosen_task['ID'] if chosen_task is not None else None),
            'chosen_task_state': (
                get_task_state(chosen_task)
                if chosen_task is not None else 'depot'),
            'model_entropy': to_python(Categorical(probs).entropy()[0]),
            'valid_action_count': int(
                (~mask[0, :self.env.tasks_num + 1].bool()).sum().item()),
            'eventual_episode_success': None,
            'eventual_deadlock': None,
            'eventual_makespan': None,
            'eventual_awt': None,
            'eventual_awar': None,
            'decoder_logit_debug': {
                'bias_global_step': self.bias_config['global_step'],
                'bias_apply': self.bias_config['apply_bias'],
                'used_weights': self.bias_config['used_weights'],
                'used_lambda': self.bias_config['used_lambda'],
                'clip_range': self.bias_config['clip_range'],
                'feature_names': self.bias_config['feature_names'],
                'action_index_order': '0=depot, action_i=task_id_i_minus_1',
                'valid_action_mask': to_python(
                    (~mask[0, :action_count].bool())),
                'capability_match': to_python(capability_match),
                'explicit_features': explicit_feature_values,
                'raw_decoder_logits': to_python(
                    logits[0, :action_count]),
                'explicit_feature_logit_bias': to_python(
                    logit_bias[0, :action_count]
                    if logit_bias is not None else None),
                'biased_decoder_logits': to_python(
                    biased_logits[0, :action_count]
                    if biased_logits is not None else None),
            },
            'candidates': candidates,
        }

    def _build_episode_evidence_summary(self, terminal_reward, finished_tasks,
                                        perf_metrics, open_task_counts,
                                        decision_count):
        """Classify termination and build JSON-safe episode-level evidence."""
        success = bool(self.env.finished and np.all(finished_tasks))
        timeout = bool(
            not success and self.env.current_time >= self.max_time)
        deadlock = bool(not success and not timeout)
        return {
            'success': success,
            'deadlock': deadlock,
            'timeout': timeout,
            'final_makespan': to_python(self.env.current_time),
            'average_waiting_time': to_python(perf_metrics['waiting_time'][0]),
            # This version deliberately does not introduce a wasted-ability
            # feature. The requested compatibility field remains null.
            'average_wasted_ability_ratio': None,
            'reward': to_python(terminal_reward),
            'num_agents': self.env.agents_num,
            'num_tasks': self.env.tasks_num,
            'num_species': self.env.species_num,
            'num_skills': self.env.traits_dim,
            'task_to_agent_ratio': (
                self.env.tasks_num / self.env.agents_num
                if self.env.agents_num else None),
            'max_open_tasks': max(open_task_counts) if open_task_counts else 0,
            'average_open_tasks': (
                float(np.mean(open_task_counts)) if open_task_counts else 0.0),
            'deadlock_step': decision_count if deadlock else None,
            'termination_reason': (
                'success' if success else 'timeout' if timeout else 'deadlock'),
        }

    def baseline_test(self):
        self.baseline_env.plot_figure = False
        perf_metrics = {}
        current_action_index = 0
        start = time.time()
        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.baseline_env.next_decision()
                random.shuffle(release_agents[0])
                self.baseline_env.current_time = current_time
                if time.time() - start > 30:
                    break
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.baseline_env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.baseline_env.agent_observe(agent_id, False))
                    return_flag = mask[0, 1:].all().item()
                    if return_flag and not np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')): ## add condition on returning to depot
                        self.baseline_env.agent_dic[agent_id]['no_choice'] = return_flag
                        continue
                    elif return_flag and np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                    task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    explicit_features, bias_params = self._get_bias_tensors(
                        agent_id, task_info.shape[1])
                    bias_kwargs = {}
                    if self.bias_config['apply_bias']:
                        bias_kwargs = {
                            'explicit_features': explicit_features,
                            'bias_params': bias_params,
                        }
                    probs, _ = self.local_baseline(
                        task_info, total_agents, mask, index, **bias_kwargs)
                    action = torch.argmax(probs, 1)
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    current_action_index += 1
                self.baseline_env.finished = self.baseline_env.check_finished()

        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def work(self, episode_number):
        """
        Interacts with the environment. The agent gets either gradients or experience buffer
        """
        baseline_rewards = []
        buffers = []
        metrics = []
        self.evidence_payload = []
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK
        for pomo_index in range(TrainParams.POMO_SIZE):
            self.env.init_state()
            episode_id = episode_number * TrainParams.POMO_SIZE + pomo_index
            terminal_reward, buffer, perf_metrics = self.run_episode(
                episode_number,
                True,
                max_waiting,
                episode_id=episode_id)
            if self.last_episode_evidence is not None:
                self.evidence_payload.append(self.last_episode_evidence)
            if terminal_reward is np.nan:
                max_waiting = True
                continue
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)
        baseline_reward = np.nanmean(baseline_rewards)

        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key == 6:
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except:
                pass
        self.episode_number = episode_number

    def convert_torch(self, args):
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data

    @staticmethod
    def obs_padding(task_info, agents, mask):
        task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
        agents = F.pad(agents, (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]), 'constant', 0)
        mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
        return task_info, agents, mask


if __name__ == '__main__':
    device = torch.device('cuda')
    # torch.manual_seed(9)
    # checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
    localNetwork = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    # localNetwork.load_state_dict(checkpoint['best_model'])
    for i in range(100):
        worker = Worker(1, localNetwork, localNetwork, 0, device=device, seed=i, save_image=False)
        worker.work(i)
        print(i)
