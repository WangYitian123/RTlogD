from hyperopt import hp

common_hyperparameters = {
    'lr': hp.uniform('lr', low=1e-4, high=3e-1),
    'weight_decay': hp.uniform('weight_decay', low=0, high=3e-3),
    'patience': hp.choice('patience', [100]),
    'batch_size': hp.choice('batch_size', [128, 256, 512]),
}

attentivefp_hyperparameters = {
    'num_layers': hp.choice('num_layers', [3]),
    'num_timesteps': hp.choice('num_timesteps', [1]),
    'graph_feat_size': hp.choice('graph_feat_size', [300]),
    'dropout': hp.uniform('dropout', low=0., high=0.6),
    'regressor_hidden_feats':hp.choice('regressor_hidden_feats', [128]),
}

def init_hyper_space(model):
    """Initialize the hyperparameter search space

    Parameters
    ----------
    model : str
        Model for searching hyperparameters

    Returns
    -------
    dict
        Mapping hyperparameter names to the associated search spaces
    """
    candidate_hypers = dict()
    candidate_hypers.update(common_hyperparameters)
    if model == 'AttentiveFP':
        candidate_hypers.update(attentivefp_hyperparameters)
    return candidate_hypers