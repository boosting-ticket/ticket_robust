import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import prune_rate, arg_nonzero_min


def weight_prune(m, pruning_perc, verbose=False):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    layer_weights = []
    masks = []
    
    #if isinstance(m, nn.Module):
        #model = m
    #else:
    model = m.model
    
    all_layers = []
    '''
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                all_layers.append(i)
        else:
            all_layers.append(layer)
    '''
    for layer in model.modules():
        all_layers.append(layer)
            
    for layer in all_layers:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            p = layer.weight
            w = p.cpu().data.abs().numpy()
            if np.sum(np.abs(w))>0:
                all_weights += list(np.ravel(w/np.mean(w))) #normalization (important)
            layer_weights.append(w/np.mean(w))

    threshold = np.percentile(np.array(all_weights), pruning_perc)
    
    prev_mask = None
    index = 0
    all_index = 0
    for i in range(len(all_layers)):
        # prune all layers including the last one.
        if isinstance(all_layers[i], (nn.Conv2d, nn.Linear)):
            pruned_inds = layer_weights[index] > threshold
            prev_mask = pruned_inds.astype('float32')
            masks.append(prev_mask)
            index += 1
            all_index += 1
        if isinstance(all_layers[i], (nn.BatchNorm2d, nn.BatchNorm1d)):
            masks.append((np.sum(prev_mask, tuple(range(1, len(prev_mask.shape)))) > 0).astype('float32'))
            all_index += 1
    #print("all_index:", all_index)
    m.set_masks(masks)
    #print("apply success")
    current_pruning_perc = prune_rate(model, verbose=False)
    if verbose:
        print('{:.2f} pruned'.format(current_pruning_perc))
    return masks


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature/neuron map by the scaled l1-norm of 
    kernel weights. In case of FC layer, the filter is replaced by neurons.
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True
    
    all_layers = []
    
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                all_layers.append(i)
        else:
            all_layers.append(layer)
            
    values = []
    
    for i in model.parameters():
        pass
    n_classes = i.data.size()[0] ## hack to get the number of classes
    
    for layer in all_layers:
        if isinstance(layer, nn.Conv2d):
            p = layer.weight
            p_np = p.data.cpu().numpy()
            
            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.abs(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important) only if there is a non-zero filter.
            if np.sum(np.abs(value_this_layer))>0:
                value_this_layer = value_this_layer / \
                (np.abs(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])
        
        if isinstance(layer, nn.Linear):
            p = layer.weight
            p_np = p.data.cpu().numpy()
            
            if len(p_np) != n_classes:  #don't prune the last layer.
                # construct masks if there is not
                if NO_MASKS:
                    masks.append(np.ones(p_np.shape).astype('float32'))
                # find the scaled l2 norm for each filter this layer
                value_this_layer = np.abs(p_np).sum(axis=1)/(p_np.shape[1])
                # normalization
                if np.sum(np.abs(value_this_layer))>0:
                    value_this_layer = value_this_layer / (np.abs(value_this_layer).sum())
                min_value, min_ind = arg_nonzero_min(list(value_this_layer))
                values.append([min_value, min_ind])
            
            if len(p_np) == n_classes:
                # Add the all_ones mask for the last fc layer, which is not pruned.
                if NO_MASKS:
                    masks.append(np.ones(p_np.shape).astype('float32'))
        
        if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            p_np = layer.weight.data.cpu().numpy()
            if NO_MASKS:
                    masks.append(np.ones(len(p_np)).astype('float32'))
    
    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    
    prev_mask = None
    index = 0
    index_mask = 0
    for i in range(len(all_layers)):
        if isinstance(all_layers[i], (nn.Conv2d, nn.Linear)):
            if index == to_prune_layer_ind:
                masks[index_mask][to_prune_filter_ind] = 0.                
                # Respectively prune the next batch_norm layer (if exist) too. 
                if isinstance(all_layers[i+1], (nn.BatchNorm2d, nn.BatchNorm1d)):
                    masks[index_mask+1][to_prune_filter_ind] = 0.
            index += 1
        if isinstance(all_layers[i], (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            index_mask += 1
                
    #print('Prune filter #{} in layer #{}'.format(
    #    to_prune_filter_ind, 
    #    to_prune_layer_ind))

    return masks


def filter_prune(m, masks, pruning_perc, verbose=False):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    if isinstance(m, nn.Module):
        model = m
    else:
        model = m.model
        
    current_pruning_perc = prune_rate(model, verbose=False)
    if current_pruning_perc == 0:
        current_pruning_perc = -0.001
        
    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        m.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        if verbose:
            print('{:.2f} pruned'.format(current_pruning_perc))

    return masks