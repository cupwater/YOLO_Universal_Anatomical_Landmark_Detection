'''
Author: Peng Bo
Date: 2022-05-23 09:11:16
LastEditTime: 2022-05-23 09:21:02
Description: 

'''
import torch
import argparse
import collections

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def main(args):
    checkpoints = torch.load(args.checkpoints, map_location='cpu')
    new_checkpoints = torch.load(args.checkpoints, map_location='cpu')
    new_checkpoints['model_state_dict'] = collections.OrderedDict()
    for k in list( checkpoints['model_state_dict'].keys() ):
        if 'module' in k:
            # print(checkpoints['model_state_dict'][k])
            new_checkpoints['model_state_dict'][k.replace('module.', '')] = checkpoints['model_state_dict'][k]

    torch.save(new_checkpoints, args.checkpoints + '.process')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract the weights of logits layer.")
    parser.add_argument("-c", "--checkpoints", type=str, required=True)
    args = parser.parse_args()
    main(args)
