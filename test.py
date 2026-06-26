import os
import torch
from attention import AttentionNet
from worker import Worker
import numpy as np
from env.task_env import TaskEnv
import time
import pickle
import json
import pandas as pd
import glob
from natsort import natsorted
import multiprocessing
from parameters import EnvParams, TrainParams, DeepSeekBiasParams

EnvParams.TASKS_RANGE = (20, 20)
EnvParams.SPECIES_RANGE = (5, 5)
EnvParams.SPECIES_AGENTS_RANGE = (3, 3)
EnvParams.MAX_TIME = 200
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0
NUM_META_AGENT = 1
GAMMA = 1
FOLDER_NAME = 'save'
testSet = 'RALTestSet'
model_path = f'model/{FOLDER_NAME}'
sampling = False
max_task = False
sampling_num = 10 if sampling else 1
save_img = False


def load_active_bias(checkpoint=None,
                     bias_dir=DeepSeekBiasParams.BIAS_CONFIG_OUTPUT_DIR):
    """Return the frozen bias snapshot to apply at test time.

    Prefer a snapshot embedded in the checkpoint (exact match to best_model);
    otherwise fall back to the newest active_bias_config_*.json written during
    training. Return None when no bias was ever produced (runs with no bias,
    i.e. the original behaviour).
    """
    if isinstance(checkpoint, dict):
        embedded = checkpoint.get('bias_config') or checkpoint.get(
            'active_bias_config')
        if embedded:
            return embedded
    files = natsorted(
        glob.glob(os.path.join(bias_dir, 'active_bias_config_*.json')))
    if not files:
        return None
    with open(files[-1], encoding='utf-8') as fp:
        return json.load(fp)


def main(f):
    device = torch.device('cuda:0') if USE_GPU_GLOBAL else torch.device('cpu')
    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    global_network.load_state_dict(checkpoint['best_model'])
    bias_config = load_active_bias(checkpoint)
    worker = Worker(0, global_network, global_network, 0, device,
                    bias_config=bias_config)
    snapshot = worker.bias_config
    print('[test] bias apply={} global_step={} lambda={:.4f} weights={}'.format(
        snapshot['apply_bias'], snapshot['global_step'],
        snapshot['used_lambda'], snapshot['used_weights']))
    index = int(f.split('/')[-1].replace('.pkl', '').replace('env_', ''))
    env = pickle.load(open(f, 'rb'))
    results_best = None
    start = time.time()
    for i in range(sampling_num):
        env.init_state()
        worker.env = env
        _, _, results = worker.run_episode(False, sampling, max_task)
        # print(results)
        if results_best is None:
            results_best = results
        else:
            if results_best['makespan'] >= results['makespan']:
                results_best = results
    if save_img:
        env.plot_animation(f'{testSet}/env_{index}', index)
    end = time.time() - start
    df_ = pd.DataFrame(results_best, index=[index])
    print(f)
    return df_, end


files = natsorted(glob.glob(f'{testSet}/env*.pkl'), key=lambda y: y.lower())
b = []
# pool = multiprocessing.Pool(processes=1)
# final_results = pool.map(main, files)
final_results =[main(f) for f in files]
perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
df = pd.DataFrame(perf_metrics)
for r in final_results:
    df = pd.concat([df, r[0]])
    b.append(r[1])
print(np.mean(b))
df.to_csv(f'{testSet}/RL_sampling_{sampling}_{sampling_num}.csv')
