# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import dgl
import errno
import json
import os
import torch
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, ScaffoldSplitter, RandomSplitter
from functools import partial
import numpy
import numpy as np
from My_Pka_Model import Pka_acidic_view,Pka_basic_view

import random
seed = 0
random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from model import *

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def setup(args, random_seed=0):
    """Decide the device to use for computing, set random seed and perform sanity check."""

    os.environ['PYTHONHASHSEED']=str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    dgl.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args['n_tasks'] == 1:
        assert args['mode'] == 'parallel', \
            'Bypass architecture is not applicable for single-task experiments.'

    return args

def default(self, obj):
    if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
        numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
        numpy.uint16,numpy.uint32, numpy.uint64)):
        return int(obj)
    elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
        numpy.float64)):
        return float(obj)
    elif isinstance(obj, (numpy.ndarray,)): # add this line
        return obj.tolist() # add this line
    return json.JSONEncoder.default(self, obj)


def get_label_mean_and_std(dataset):
    """Compute the mean and std of labels.

    Non-existing labels are excluded for computing mean and std.

    Parameters
    ----------
    dataset
        We assume that len(dataset) gives the number of datapoints
        in the dataset and dataset[i] gives the SMILES, RDKit molecule
        instance, DGLGraph, label and mask for the i-th datapoint.

    Returns
    -------
    labels_mean: float32 tensor of shape (T)
        Mean of the labels for all tasks
    labels_std: float32 tensor of shape (T)
        Std of the labels for all tasks
    """
    _, _, label, _ = dataset[0]
    n_tasks = label.shape[-1]
    task_values = {t: [] for t in range(n_tasks)}
    for i in range(len(dataset)):
        _, _, label, mask = dataset[i]
        for t in range(n_tasks):
            if mask[t].data.item() == 1.:
                task_values[t].append(label[t].data.item())

    labels_mean = torch.zeros(n_tasks)
    labels_std = torch.zeros(n_tasks)
    for t in range(n_tasks):
        labels_mean[t] = float(np.mean(task_values[t]))
        labels_std[t] = float(np.std(task_values[t]))

    return labels_mean, labels_std

def collate(data):
    """Batching a list of datapoints for dataloader in training GNNs.

    Returns
    -------
    smiles: list
        List of smiles
    bg: DGLGraph
        DGLGraph for a batch of graphs
    labels: Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks: Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))
    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def load_model(exp_configure):
    
    if exp_configure['model'] == 'AttentiveFP':
        if exp_configure['mode'] == 'parallel':
            model_class = AttentiveFPRegressor
        else:
            model_class = AttentiveFPRegressorBypass
        model = model_class(in_node_feats=exp_configure['in_node_feats'],
                            in_edge_feats=exp_configure['in_edge_feats'],
                            gnn_out_feats=exp_configure['graph_feat_size'],
                            num_layers=exp_configure['num_layers'],
                            num_timesteps=exp_configure['num_timesteps'],
                            n_tasks=exp_configure['n_tasks'],
                            regressor_hidden_feats=exp_configure['regressor_hidden_feats'],
                            dropout=exp_configure['dropout'])
    return model

def init_featurizer(args):
    """Initialize node/edge featurizer

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['atom_featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['atom_featurizer_type']))


    if args['model'] in ['AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer()
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer()
    else:
        args['edge_featurizer'] = None

    return args

def load_dataset(args, df,name):
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph,num_virtual_nodes=0),
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'],
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['result_path'] +'/'+ str(name)+'_graph.bin',
                                 task_names=args['task'],
                                 n_jobs=args['num_workers'],
                                 load=False
                                )

    return dataset

def get_configure(model, args=None):
    """Query for the manually specified configuration

    Parameters
    ----------
    model : str
        Model type

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    if args == None:
            with open('model_configures/{}.json'.format(model), 'r') as f:
                config = json.load(f)
            return config
    else:
            with open('./model_configures/{}.json'.format(model), 'r') as f:
                config = json.load(f)
            return config

def mkdir_p(path):
    """Create a folder for the given path.

    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def init_trial_path(args):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def collate_molgraphs_unlabeled(data):
    """Batching a list of datapoints without labels

    Parameters
    ----------
    data : list of 2-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    """
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg
    


def predict(args, model, bg):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    elif args['bond_featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
     
        return model(bg, node_feats, edge_feats,get_node_weight=True)
    else:
        node_feats = bg.ndata['h'].to(args['device'])
        edge_feats = bg.edata['e'].to(args['device'])
        with bg.local_scope():
            pka1_model = Pka_acidic_view(node_feat_size = 74,
                    edge_feat_size = 12,
                    output_size = 1,
                    num_layers= 6,
                    graph_feat_size=200,
                    dropout=0).to(args['device'])

            pka1_model.eval()
            with torch.no_grad():
                
                pka1_model.load_state_dict(torch.load('Trained_model/site_acidic.pkl',map_location='cpu'))
                prediction,pka1_atom_list = pka1_model(bg,bg.ndata['h'], bg.edata['e'])

        with bg.local_scope():
            pka2_model = Pka_basic_view(node_feat_size = 74,
                    edge_feat_size = 12,
                    output_size = 1,
                    num_layers= 6,
                    graph_feat_size=200,
                    dropout=0).to(args['device'])
           # print('basic load!')
            pka2_model.eval()
            with torch.no_grad():
                
                pka2_model.load_state_dict(torch.load('Trained_model/site_basic.pkl',map_location='cpu'))
                prediction,pka2_atom_list = pka2_model(bg,bg.ndata['h'], bg.edata['e'])

        pka1_atom_list=np.array(pka1_atom_list)
        pka1_atom_list[np.isinf(pka1_atom_list)]=15
        pka2_atom_list=np.array(pka2_atom_list)
        pka2_atom_list[np.isinf(pka2_atom_list)]=0         

        pka1_feature = torch.Tensor(pka1_atom_list/11).to(args['device'])
        pka2_feature = torch.Tensor(pka2_atom_list/11).to(args['device'])

        pka1_feature=pka1_feature.unsqueeze(-1)
        pka2_feature=pka2_feature.unsqueeze(-1)

        node_feats = torch.cat([node_feats,pka1_feature,pka2_feature],dim = 1)

        return model(bg, node_feats, edge_feats)
def split_dataset(args, dataset):
    """Split the dataset
    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance
    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold_decompose':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='decompose')
    elif args['split'] == 'scaffold_smiles':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set