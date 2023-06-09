# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
from process import run
import sys
sys.path.append("..") 
from torch.utils.data import DataLoader
from utils import  load_model
from utils import get_configure, \
    collate_molgraphs, load_model, predict, load_dataset



def run_an_eval_epoch(smiles_list,args, model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            prediction = predict(args, model, bg)
            prediction = prediction.detach().cpu() * args['train_std'].cpu()+args['train_mean'].cpu()
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)

        output_data = {'canonical_smiles': smiles_list}
        if args['task'] is None:
            args['task'] = ['task_{:d}'.format(t) for t in range(1, args['n_tasks'] + 1)]
        else:
            pass
        for task_id, task_name in enumerate(args['task']):
            output_data[task_name] = predictions[:, task_id]
        df = pd.DataFrame(output_data)
        out=pd.read_csv("example.csv",header=None, names=['smiles'])
        out['predict']=round(df['standard_value'],3)
        out.to_csv('results/predict.csv', index=False)



def main(smiles_list,args, exp_config, test_set):
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

    t0 = time.time()
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config).to(args['device'])
    checkpoint = torch.load("final_model/RTlogD/model_pretrain_76.pth",map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    run_an_eval_epoch(smiles_list,args, model, test_loader)

    print('It took {:.4f}s to complete the task'.format(time.time() - t0))

   
if __name__ == '__main__':
    import torch
    from utils import setup
    with open('final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
    with open('final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)
    test_set = pd.read_csv('example.csv',header=None, names=['smiles'])

    print("test_set",test_set)
    #standardize the molecules
    test_set=run(test_set)
    print("out: ",test_set)
    test_set['logp']=np.nan
    test_set['exp']=np.nan
    test_set['standard_value']=np.nan
    smiles_list=test_set['smiles'].to_list()
    test_set = load_dataset(args,test_set,"test")
    exp_config = get_configure(args['model'],"test")
    main(smiles_list,args, exp_config,test_set)