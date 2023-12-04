import torch
import argparse
import time
import importlib
import optuna
import csv
import os
import numpy as np

from sess.utils.utils import get_logger, ACC_KPI
from sess.utils.model_selection import handle_adj, build_graph
from sess.utils.metrics import accuracy_calculator

from sess.model.gcegnn import CombineGraph
from sess.model.hide import HIDE
from sess.model.attenMixer import AreaAttnModel

from config import Model_setting, HyperParameter_setting, Dataset_setting


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='bundle', help='bundle/games/ml-1m')
parser.add_argument('--model', default='FPMC', help='MCPRN/STAMP/NARM/GCE-GNN/FPMC/HIDE/AttenMixer')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--sample_num', type=int, default=50, help='50, 150')
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--trials', type=int, default=30)

opt = parser.parse_args()
init_seed(opt.seed)

TRIAL_CNT = 0
def tune():
    global train_dataset
    global valid_dataset
    global candidate_data

    base_path = '../Dataset/'
    train_data = np.load(f'{base_path}{opt.dataset}/ID/train_sample_{opt.sample_num}.npy', allow_pickle=True).tolist()
    vali_data = np.load(f'{base_path}{opt.dataset}/ID/valid.npy', allow_pickle=True).tolist()
    candidate_data = np.load(f'{base_path}{opt.dataset}/ID/valid_candidate.npy', allow_pickle=True).tolist()

    model_config = Model_setting[opt.model]
    model_config['gpu'] = opt.gpu
    data_config = Dataset_setting[opt.dataset]
    logger = get_logger(f'tune_{model_config["description"]}_{opt.dataset}')
    
    dataloader = getattr(importlib.import_module('sess.utils.dataset'), model_config['dataloader'], None)
    train_dataset = dataloader(train_data, model_config)
    valid_dataset = dataloader(vali_data, model_config, candidate_set=candidate_data, isTrain=False)
    if opt.model in ['NARM','FPMC','STAMP','MCPRN', 'AttenMixer']:
        train_dataset = train_dataset.get_loader(model_config, shuffle=True)
        valid_dataset = valid_dataset.get_loader(model_config, shuffle=False)

    tune_params = []
    def objective(trial):
        global TRIAL_CNT
        for key, value in HyperParameter_setting[opt.model].items():
            if key == 'int':
                for para_name, scales in value.items():
                    model_config[para_name] = trial.suggest_int(para_name, scales['min'], scales['max'], step=scales['step'])
                    tune_params.append(para_name)
            elif key == 'categorical':
                for para_name, scales in value.items():
                    model_config[para_name] = trial.suggest_categorical(para_name, scales)
                    tune_params.append(para_name)
    

        if opt.model in ['NARM','FPMC','STAMP','MCPRN']:
            model = getattr(importlib.import_module('sess.model.{}'.format(model_config['model_dir'])), opt.model, None)
            model = model(data_config['num_node'], model_config, logger)

        elif opt.model in ['GCE-GNN']:
            adj, num = build_graph(train_data, data_config, model_config)
            num_node = data_config['num_node'] + 1
            adj, num = handle_adj(adj, num_node, model_config['n_sample_all'], num)
            model = CombineGraph(model_config, num_node, adj, num, logger)
        elif opt.model in ['HIDE']:
            num_node = data_config['num_node'] + 1
            model = HIDE(model_config, num_node, logger=logger)
        elif opt.model in ['AttenMixer']:
            num_node = data_config['num_node'] + 1
            model = AreaAttnModel(model_config, num_node, logger)

    
        # training process
        model.fit(train_dataset)#, valid_dataset)
        preds, truth = model.predict(valid_dataset, k=opt.topK)
        metrics = accuracy_calculator(preds, truth, ACC_KPI)
        kpi = metrics[0]
        logger.info(f"Finish {TRIAL_CNT+1} trial for {opt.model}...")
        TRIAL_CNT += 1

        return kpi

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=opt.seed))
    study.optimize(objective, n_trials=opt.trials)

    tune_params = list(set(tune_params))
    tune_log_path = f'./tune_log/sample_{opt.sample_num}/{opt.dataset}/'
    if not os.path.exists(tune_log_path):
        os.makedirs(tune_log_path)
    res_csv = tune_log_path + f'result_{opt.dataset}_{opt.model}.csv'
    with open(res_csv, 'w', newline='') as f:
        fieldnames = ['Trial ID'] + tune_params + ['NDCG@5']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for estudio in study.trials:
            w_dict = {}
            w_dict['Trial ID'] = estudio.number+1
            for paras in tune_params:
                w_dict[paras] = estudio.params[paras]
            w_dict['NDCG@5'] = estudio.value
            writer.writerow(w_dict)

        best_dict = {}
        best_dict['Trial ID'] = study.best_trial.number+1
        best_dict['NDCG@5'] = study.best_value
        for paras in tune_params:
            best_dict[paras] = study.best_trial.params[paras]
        writer.writerow(best_dict)
        f.flush()
        f.close()

    logger.info(f"Best trial for {opt.model}: {study.best_trial.number+1}")

if __name__ == '__main__':
    tune()