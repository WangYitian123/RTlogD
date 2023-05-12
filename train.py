# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from copy import deepcopy
from dgllife.utils import Meter, EarlyStopping
from hyperopt import fmin, tpe
from shutil import copyfile
import shutil
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import get_label_mean_and_std, collate, load_model
from hyper import init_hyper_space
from utils import get_configure, mkdir_p, init_trial_path, \
    collate_molgraphs, load_model, predict, init_featurizer, load_dataset
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from torch.optim import lr_scheduler

def run_a_train_epoch(args, epoch,model, data_loader, criterion, optimizer):
    model.train()
    train_meter = Meter(args['train_mean'], args['train_std'])
    epoch_loss = torch.zeros(args['n_tasks'])
    for smiles, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles)==1:
            continue
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        prediction = predict(args, model, bg)
        # Normalize the labels so that the scale of labels will be similar
        loss = criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        # Mask non-existing labels
        loss = (loss * (masks != 0).float()).sum(0)
        epoch_loss = epoch_loss + loss.detach().cpu().data
        loss = loss.sum() / bg.batch_size
        optimizer.zero_grad()
        loss.backward()
       
        clip_grad_norm_(model.parameters(),max_norm=20,norm_type=2)
        optimizer.step()
        train_meter.update(prediction, labels, masks)
    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_loss = epoch_loss.cpu().detach().tolist()
    train_score = np.mean(train_meter.compute_metric(args['metric']))
   
    
    return epoch_loss, train_meter.pearson_r2(), train_meter.mae(),train_meter.rmse()
    
def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args['train_mean'], args['train_std'])
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            prediction = predict(args, model, bg)
            eval_meter.update(prediction, labels, masks)

    return eval_meter.pearson_r2(), eval_meter.mae(),eval_meter.rmse()


def log_model_evaluation(args, model, train_loader, val_loader, test_loader,best_epoch):
    with open(args['trial_path'] + '/results.txt', 'w') as f:
        def _log_values(metric, train_values, val_values, test_values):
            f.write(f'best_epoch is {best_epoch+1}\n')
            f.write('{}\n'.format(metric))
            headline = '|            | averaged |'
            for t in args['task']:
                headline += ' {:15} |'.format(t)
            headline += '\n'
            f.write(headline)
            f.write('| ' + '-' * (len(headline) - 5) + ' |\n')
            for name, values in {'Training': train_values,
                                 'Validation': val_values,
                                 'Test': test_values}.items():
                row = '| {:10s} | {:8.3f} |'.format(name, np.mean(values))
                
                for t in range(len(args['task'])):
                    if len(args['task'])==1:
                        row += ' {:15.3f} |'.format(values)
                    else:
                        row += ' {:15.3f} |'.format(values[t])
                row += '\n'
                f.write(row)
            f.write('| ' + '-' * (len(headline) - 5) + ' |\n')
            f.write('\n')

        train_r2, train_mae,train_rmse = run_an_eval_epoch(args, model, train_loader)
        val_r2, val_mae,val_rmse = run_an_eval_epoch(args, model, val_loader)
        test_r2, test_mae,test_rmse = run_an_eval_epoch(args, model, test_loader)
        _log_values('r2', train_r2, val_r2, test_r2)
        _log_values('mae', train_mae, val_mae, test_mae)
        _log_values('rmse', train_rmse, val_rmse, test_rmse)



def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'mode':args['mode'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()+2
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    # Set up directory for saving results
    args = init_trial_path(args)
    t0 = time.time()

    train_mean, train_std = get_label_mean_and_std(train_set)
    train_mean, train_std = train_mean.to(args['device']), train_std.to(args['device'])

    args['train_mean'], args['train_std'] = train_mean, train_std
    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config).to(args['device'])
    print('pretrain model load!')
    x = torch.load('RT_pre-trained model/pka_76.pth',map_location='cpu')['model_state_dict']
    del x['regressor.predict.4.weight']
    del x['regressor.predict.4.bias']
    model.load_state_dict(x, strict=False)
    loss_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                     weight_decay=exp_config['weight_decay'])

    best_epoch=0
    best_val_rmse=100
    for epoch in range(args['num_epochs']):

        # Train
        loss, train_r2, train_mae,train_rmse=run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        # Validation and early stop
        val_r2,val_mae,val_score = run_an_eval_epoch(args, model, val_loader)
        if np.mean(val_score)<best_val_rmse:
            best_val_rmse=np.mean(val_score)
            best_epoch=epoch
            torch.save(model.state_dict(),args['result_path']+"/epoch/epoch{}_model_param.pickle".format(epoch))
        print('epoch {:d}/{:d}, training {} {:.4f},valid {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'],np.mean(train_rmse),args['metric'],np.mean(val_score)))

    model.load_state_dict(torch.load(args['result_path']+"/epoch/epoch{}_model_param.pickle".format(best_epoch)))
    torch.save(model.state_dict(),os.path.join(args['trial_path'],"model.pth"))
    r2,val_mae,best_score = run_an_eval_epoch(args, model, val_loader)

    with open(args['trial_path'] + '/configure.json', 'w') as f:
        json.dump(exp_config, f, indent=2)
    with open(args['trial_path'] +'/args.pickle', 'wb') as f:
        pickle.dump(args,f)

    print('It took {:.4f}s to complete the task'.format(time.time() - t0))
    log_model_evaluation(args, model, train_loader, val_loader, test_loader,best_epoch)
    return args['trial_path'], best_score
   

def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []

    candidate_hypers = init_hyper_space(args['model'])
 
    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args['metric'] in ['r2']:
            # Maximize R2 is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path

if __name__ == '__main__':
    from argparse import ArgumentParser
    from argparse import ArgumentParser
    import torch
    from utils import mkdir_p, setup
    for i in range(0,10):
        random_state=i
        parser = ArgumentParser('(Multitask) Regression')
        parser.add_argument('-sc', '--smiles-column', type=str,default='smiles',
                            help='Header for the SMILES column in the CSV file')
        parser.add_argument('-t', '--task', default='standard_value,logp', type=str,
                            help='Header for the tasks to model. If None, we will model '
                                'all the columns except for the smiles_column in the CSV file. '
                                '(default: None)')
        parser.add_argument('-me', '--metric', choices=['r2', 'mae', 'rmse'], default='mae',
                            help='Metric for evaluation')
        parser.add_argument('-mo', '--model', default='AttentiveFP',
                            help='Model to use (default: GCN)')
        parser.add_argument('-a', '--atom-featurizer-type', choices=['canonical', 'attentivefp'],
                            default='canonical',
                            help='Featurization for atoms (default: canonical)')
        parser.add_argument('-b', '--bond-featurizer-type', choices=['canonical', 'attentivefp'],
                            default='canonical',
                            help='Featurization for bonds (default: canonical)')
        parser.add_argument('-n', '--num-epochs', type=int, default=300)
        parser.add_argument('-nw', '--num-workers', type=int, default=1,
                            help='Number of processes for data loading (default: 1)')
        parser.add_argument('-pe', '--print-every', type=int, default=20,
                            help='Print the training progress every X mini-batches')
        parser.add_argument('-p', '--result-path', type=str, default='trained_model',
                            help='Path to save training results (default: regression_results)')
        parser.add_argument('-ne', '--num-evals', type=int, default=None,
                            help='Number of trials for hyperparameter search (default: None)')
        parser.add_argument('--mode', type=str, choices=['parallel', 'bypass'],default='parallel',
                            help='Architecture to use for multitask learning')
        args = parser.parse_args().__dict__

        if torch.cuda.is_available():
            args['device'] = torch.device('cuda:0')
        else:
            args['device'] = torch.device('cpu')
        tensorboard = SummaryWriter(log_dir="runs/"+str(args['result_path']))
        if args['task'] is not None:
            args['task'] = args['task'].split(',')

        args = init_featurizer(args)
        mkdir_p(args['result_path'])
        mkdir_p(args['result_path']+'/epoch')
        
        train_set = pd.read_csv('train_data/scaffold_train_three_all_new_norm.csv')
        val_set = pd.read_csv('train_data/scaffold_valid_three_all_new_norm.csv')
        test_set = pd.read_csv('train_data/scaffold_test_three_all_new_norm.csv')

        train_set = load_dataset(args, train_set,"train")
        val_set = load_dataset(args, val_set,"valid")
        test_set = load_dataset(args,test_set,"test")
      
        args['n_tasks'] = train_set.n_tasks
        args = setup(args)
        assert train_set.n_tasks == val_set.n_tasks == test_set.n_tasks
        if args['num_evals'] is not None:
            assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                        'be greater than 0, got {:d}'.format(args['num_evals'])
            print('Start hyperparameter search with Bayesian '
                'optimization for {:d} trials'.format(args['num_evals']))
            trial_path = bayesian_optimization(args, train_set, val_set, test_set)
        else:
            print('Use the manually specified hyperparameters')
            exp_config = get_configure(args['model'])
            main(args, exp_config, train_set, val_set, test_set)
            trial_path = args['result_path'] + '/1'
        
        path=args['result_path']+'/epoch'
        shutil.rmtree(path,ignore_errors = False,onerror = None)

