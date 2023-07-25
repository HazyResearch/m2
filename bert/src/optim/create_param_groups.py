def create_param_groups(cfg, model):
    '''Create sets of parameter groups based on whether parameter has `_optim` attribute.'''
    if not any(hasattr(p, '_optim') for p in model.parameters()):
        return model.parameters()
    
    special_params = set()
    other_params = set()
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # In case of parameter sharing, some parameters show up here but are not in
            # param_dict.keys()
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if hasattr(p, '_optim'):
                special_params.add(fpn)
            else:
                other_params.add(fpn)
    
    param_groups = [
        {"params": [param_dict[pn] for pn in other_params]}
    ]

    # Add parameters with special hyperparameters
    # Unique dicts
    hps = [
        dict(s)
        for s in set(frozenset(param_dict[pn]._optim.items())
                     for pn in special_params)
    ]
    for hp in hps:
        params = [
            param_dict[pn]
            for pn in sorted(list(special_params)) if param_dict[pn]._optim == hp
        ]
        param_groups.append({"params": params, **hp})
    
    return param_groups