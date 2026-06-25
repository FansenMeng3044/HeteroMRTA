import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from attention import AttentionNet
from runner import RLRunner
from parameters import *
from env.task_env import TaskEnv
from scipy.stats import ttest_rel
from torch.distributions import Categorical
from utils.evidence_recorder import EvidenceRecorder
from utils.bias_manager import BiasManager
from utils.deepseek_client import DeepSeekClient
from utils.deepseek_bias_controller import DeepSeekBiasController
from utils.rollout_bias_sync import (
    complete_evidence_payloads,
    rollout_uses_active_snapshot,
    slice_transition_buffer,
)


class Logger(object):
    def __init__(self):
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)

    def set(self,  global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay

    def write_to_board(self, tensorboard_data, curr_episode):
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
        reward, p_l, entropy, grad_norm, success_rate, time, time_cost, waiting, distance, effi = tensorboard_data
        metrics = {'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],
                   'Loss/Policy Loss': p_l,
                   'Loss/Entropy': entropy,
                   'Loss/Grad Norm': grad_norm,
                   'Loss/Reward': reward,
                   'Perf/Makespan': time,
                   'Perf/Success rate': success_rate,
                   'Perf/Time cost': time_cost,
                   'Perf/Waiting time': waiting,
                   'Perf/Traveling distance':distance,
                   'Perf/Waiting Efficiency': effi
                   }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

    def load_saved_model(self):
        print('Loading Model...')
        checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
        if SaverParams.LOAD_FROM == 'best':
            model = 'best_model'
        else:
            model = 'model'
        self.global_net.load_state_dict(checkpoint[model])
        self.baseline_net.load_state_dict(checkpoint[model])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        best_perf = checkpoint['best_perf']
        print("curr_episode set to ", curr_episode)
        print("best_perf so far is ", best_perf)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
        return curr_episode, curr_level, best_perf

    def save_model(self, curr_episode, curr_level, best_perf):
        print('Saving model', end='\n')
        checkpoint = {"model": self.global_net.state_dict(),
                      "best_model": self.baseline_net.state_dict(),
                      "best_optimizer": self.optimizer.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "episode": curr_episode,
                      "lr_decay": self.lr_decay.state_dict(),
                      "level": curr_level,
                      "best_perf": best_perf
                      }
        path_checkpoint = "./" + SaverParams.MODEL_PATH + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)
        print('Saved model', end='\n')

    @staticmethod
    def generate_env_params(curr_level=None):
        per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
        species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
        tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
        params = [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num)]
        return params

    @staticmethod
    def generate_test_set_seed():
        test_seed = np.random.randint(low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES).tolist()
        return test_seed


def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    if ini_dictionary2 is not None:
        merged_dict = {**ini_dictionary1, **ini_dictionary2}
        final_dict = {}
        for k, v in merged_dict.items():
            final_dict[k] = ini_dictionary1[k] + v
        return final_dict
    else:
        return ini_dictionary1


def main():
    logger = Logger()
    evidence_recorder = EvidenceRecorder(
        enabled=EvidenceParams.ENABLE_EVIDENCE_LOGGING,
        interval_steps=EvidenceParams.EVIDENCE_LOG_INTERVAL_STEPS,
        output_dir=EvidenceParams.EVIDENCE_OUTPUT_DIR,
        max_cases_per_report=EvidenceParams.MAX_CASES_PER_REPORT,
        low_threshold=EvidenceParams.LOW_CAPABILITY_THRESHOLD,
        better_alternative_gap=EvidenceParams.BETTER_ALTERNATIVE_GAP,
        deadlock_lookback=EvidenceParams.DEADLOCK_LOOKBACK_DECISIONS)
    bias_manager = BiasManager(
        enabled=DeepSeekBiasParams.ENABLE_DEEPSEEK_BIAS,
        output_dir=DeepSeekBiasParams.BIAS_CONFIG_OUTPUT_DIR,
        response_output_dir=(
            DeepSeekBiasParams.DEEPSEEK_RESPONSE_OUTPUT_DIR),
        ema_alpha=DeepSeekBiasParams.LLM_BIAS_EMA_ALPHA,
        update_interval_steps=(
            DeepSeekBiasParams.DEEPSEEK_BIAS_UPDATE_INTERVAL_STEPS),
        weight_range=DeepSeekBiasParams.WEIGHT_RANGE,
        lambda_range=DeepSeekBiasParams.LAMBDA_RANGE,
        clip_bound_range=DeepSeekBiasParams.CLIP_BOUND_RANGE)
    deepseek_client = DeepSeekClient(
        base_url=DeepSeekBiasParams.DEEPSEEK_BASE_URL,
        model=DeepSeekBiasParams.DEEPSEEK_MODEL,
        temperature=DeepSeekBiasParams.DEEPSEEK_TEMPERATURE,
        max_tokens=DeepSeekBiasParams.DEEPSEEK_MAX_TOKENS,
        timeout=DeepSeekBiasParams.DEEPSEEK_TIMEOUT,
        use_json_response=DeepSeekBiasParams.DEEPSEEK_USE_JSON_RESPONSE,
        thinking_type=DeepSeekBiasParams.DEEPSEEK_THINKING_TYPE)
    bias_controller = DeepSeekBiasController(
        enabled=DeepSeekBiasParams.ENABLE_DEEPSEEK_BIAS,
        client=deepseek_client,
        bias_manager=bias_manager,
        response_output_dir=(
            DeepSeekBiasParams.DEEPSEEK_RESPONSE_OUTPUT_DIR),
        update_interval_steps=(
            DeepSeekBiasParams.DEEPSEEK_BIAS_UPDATE_INTERVAL_STEPS))
    active_bias_config = bias_manager.get_snapshot()
    ray.init()
    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')

    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)

    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode = 0
    curr_level = 0
    best_perf = -200
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

    # get initial weights
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)

    # launch the first job on each runner
    jobs = []

    env_params = logger.generate_env_params(curr_level)
    for i, meta_agent in enumerate(meta_agents):
        jobs.append(meta_agent.training.remote(
            weights_memory,
            baseline_weights_memory,
            curr_episode,
            env_params,
            active_bias_config))
        curr_episode += 1
    test_set = logger.generate_test_set_seed()
    baseline_value = None
    experience_buffer = {idx:[] for idx in range(9)}
    perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    training_data = []

    try:
        while True:
            # wait for any job to be completed
            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
            buffer, metrics, info, evidence_payload = done_job
            generated_reports = []
            stale_rollout = not rollout_uses_active_snapshot(
                info, active_bias_config)
            if stale_rollout:
                print(
                    'Discarded stale rollout bias version',
                    info.get('bias_global_step'),
                    'while active version is',
                    active_bias_config['global_step'])
            else:
                transition_count = len(buffer[0])
                accepted_count = transition_count
                accepted_payload = evidence_payload
                if evidence_recorder.enabled and transition_count:
                    steps_to_boundary = (
                        evidence_recorder.window_start
                        + evidence_recorder.interval_steps
                        - evidence_recorder.global_step)
                    accepted_count = min(
                        transition_count, max(0, steps_to_boundary))
                    accepted_payload = complete_evidence_payloads(
                        evidence_payload,
                        accepted_count,
                        info.get('episode_number', 0))
                accepted_buffer = slice_transition_buffer(
                    buffer, accepted_count)
                experience_buffer = fuse_two_dicts(
                    experience_buffer, accepted_buffer)
                if accepted_count == transition_count:
                    perf_metrics = fuse_two_dicts(perf_metrics, metrics)

                for episode_payload in accepted_payload:
                    try:
                        written_paths = (
                            evidence_recorder.record_episode_payload(
                                episode_payload))
                        generated_reports.extend(
                            path for path in written_paths
                            if str(path).lower().endswith('.md'))
                    except Exception as error:
                        # Logging failures must not terminate training.
                        print('Evidence logging warning:', error)
                for report_path in generated_reports:
                    active_bias_config, updated = (
                        bias_controller.process_report(report_path))
                    if updated:
                        print(
                            'Updated cached capability_match bias at step',
                            active_bias_config['global_step'])

            update_done = False
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
                # env_params = logger.generate_env_params(curr_level)
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]

                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)  # (batch,sample_size,2)
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)  # (batch,sample_size,k_size)
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)  # (batch,1,1)
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)  # (batch,1,1)
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)  # (batch,1,1)
                    explicit_features_batch = torch.stack(
                        rollouts[7], dim=0).to(device)
                    bias_params_batch = torch.stack(
                        rollouts[8], dim=0).to(device)

                    # REINFORCE
                    probs, _ = global_network(
                        task_inputs,
                        agent_inputs,
                        global_mask_batch,
                        index,
                        explicit_features=explicit_features_batch,
                        bias_params=bias_params_batch)
                    dist = Categorical(probs)
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
                    policy_loss = - logp * advantage_batch.flatten().detach()
                    policy_loss = policy_loss.mean()

                    loss = policy_loss
                    global_optimizer.zero_grad()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=100, norm_type=2)
                    global_optimizer.step()

                    train_metrics.append([reward_batch.mean().item(), policy_loss.item(), entropy.item(), grad_norm.item()])
                lr_decay.step()

                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
                train_metrics = np.nanmean(train_metrics, axis=0)
                for v in perf_metrics.values():
                    del v[:]
                data = [*train_metrics, *perf_data]
                training_data.append(data)

            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            # get the updated global weights
            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)

            env_params = logger.generate_env_params(curr_level)
            jobs.append(meta_agents[info['id']].training.remote(
                weights_memory,
                baseline_weights_memory,
                curr_episode,
                env_params,
                active_bias_config))
            curr_episode += 1

            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)

            if curr_episode % 512 == 0:
                logger.save_model(curr_episode, curr_level, best_perf)

            if TrainParams.EVALUATE:
                if curr_episode % 1024 == 0:
                    # stop the training
                    ray.wait(jobs, num_returns=TrainParams.NUM_META_AGENT)
                    for a in meta_agents:
                        ray.kill(a)
                    print('Evaluate baseline model at ', curr_episode)

                    # test the baseline model on the new test set
                    if baseline_value is None:
                        test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                        for _, test_agent in enumerate(test_agent_list):
                            ray.get(test_agent.set_baseline_weights.remote(baseline_weights_memory))
                        rewards = dict()
                        seed_list = copy.deepcopy(test_set)
                        evaluate_jobs = [
                            test_agent_list[i].testing.remote(
                                seed=seed_list.pop(),
                                bias_config=active_bias_config)
                            for i in range(TrainParams.NUM_META_AGENT)]
                        while True:
                            test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                            test_result = ray.get(test_done_id)[0]
                            reward, seed, meta_id = test_result
                            rewards[seed] = reward
                            if seed_list:
                                evaluate_jobs.append(
                                    test_agent_list[meta_id].testing.remote(
                                        seed=seed_list.pop(),
                                        bias_config=active_bias_config))
                            if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                                break
                        rewards = dict(sorted(rewards.items()))
                        baseline_value = np.stack(list(rewards.values()))
                        for a in test_agent_list:
                            ray.kill(a)

                    # test the current model's performance
                    test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(weights_memory))
                    rewards = dict()
                    seed_list = copy.deepcopy(test_set)
                    evaluate_jobs = [
                        test_agent_list[i].testing.remote(
                            seed=seed_list.pop(),
                            bias_config=active_bias_config)
                        for i in range(TrainParams.NUM_META_AGENT)]
                    while True:
                        test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                        test_result = ray.get(test_done_id)[0]
                        reward, seed, meta_id = test_result
                        rewards[seed] = reward
                        if seed_list:
                            evaluate_jobs.append(
                                test_agent_list[meta_id].testing.remote(
                                    seed=seed_list.pop(),
                                    bias_config=active_bias_config))
                        if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                            break
                    rewards = dict(sorted(rewards.items()))
                    test_value = np.stack(list(rewards.values()))
                    for a in test_agent_list:
                        ray.kill(a)

                    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

                    # update baseline if the model improved more than 5%
                    print('test value', test_value.mean())
                    print('baseline value', baseline_value.mean())
                    if test_value.mean() > baseline_value.mean():
                        _, p = ttest_rel(test_value, baseline_value)
                        print('p value', p)
                        if p < 0.05:
                            print('update baseline model at ', curr_episode)
                            if device != local_device:
                                weights = global_network.to(local_device).state_dict()
                                global_network.to(device)
                            else:
                                weights = global_network.state_dict()
                            baseline_weights = copy.deepcopy(weights)
                            baseline_network.load_state_dict(baseline_weights)
                            weights_memory = ray.put(weights)
                            baseline_weights_memory = ray.put(baseline_weights)
                            test_set = logger.generate_test_set_seed()
                            print('update test set')
                            baseline_value = None
                            best_perf = test_value.mean()
                            logger.save_model(curr_episode, None, best_perf)
                    jobs = []
                    for i, meta_agent in enumerate(meta_agents):
                        jobs.append(meta_agent.training.remote(
                            weights_memory,
                            baseline_weights_memory,
                            curr_episode,
                            env_params,
                            active_bias_config))
                        curr_episode += 1

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    finally:
        try:
            written_paths = evidence_recorder.flush_window_if_needed(force=True)
            for report_path in written_paths:
                if str(report_path).lower().endswith('.md'):
                    active_bias_config, _ = bias_controller.process_report(
                        report_path)
            if written_paths:
                print('Flushed evidence report files:')
                for path in written_paths:
                    print(path)
        except Exception as error:
            print('Evidence flush warning:', error)


if __name__ == "__main__":
    main()
