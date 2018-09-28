import torch
import torch.nn as nn
from model.build_model import DarkNet

def weight_init(model, path='./model/yolov3.pytorch', state=None):
    ''' path .pytorch state_dict '''

    if state is None:
        state = torch.load(path)

    for n, m in model.named_modules():
        
        state_n  = n

        if isinstance(m, nn.Conv2d):
            if m.out_channels == state[state_n+'.weight'].shape[0]:
                m.weight.data.copy_(state[state_n+'.weight'])
                
                if m.bias is not None:
                    m.bias.data.copy_(state[state_n+'.bias'])
            else:
                # print(m.weight.data.shape,  m.bias.data.shape)
                # print(m.out_channels, state[n+'.weight'].shape)
                pass

        elif isinstance(m, nn.BatchNorm2d):
            m.running_mean.data.copy_(state[state_n+'.running_mean'])
            m.running_var.data.copy_(state[state_n+'.running_var'])

        elif isinstance(m, nn.Linear):
            pass
            
    return model


def resume(model, optimizer=None, scheduler=None, state_path=''):
    '''
    state is dict, keys epoch model scheduler
    '''
    state = torch.load(state_path)
    model.load_state_dict(state['model'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])


if __name__ == '__main__':

    model = DarkNet('./_model/yolov3.cfg', cls_num=20)
    print(model)

    weight_init(model)