import torch
import torch.nn as nn

import torchvision.models as models 
import distiller
import math
import numpy as np
from utils import yaml_ordered_load
from utils import get_dataloader


def train(train_loader, model, criterion, optimizer, epoch, compression_scheduler):
    '''
    '''
    steps_per_epoch = len(train_loader) // 10

    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        compression_scheduler.on_minibatch_begin(epoch, i, steps_per_epoch, optimizer)

        output = model(inputs)
        loss = criterion(output, target)

        agg_loss = compression_scheduler.before_backward_pass(epoch, i, steps_per_epoch, loss, optimizer=optimizer, return_loss_components=True)
        loss = agg_loss.overall_loss

        optimizer.zero_grad()
        loss.backward()

        compression_scheduler.before_parameter_optimization(epoch, i, steps_per_epoch, optimizer)
        optimizer.step()

        compression_scheduler.on_minibatch_end(epoch, i, steps_per_epoch, optimizer)



if __name__ == '__main__':


    model = models.alexnet(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    compress = './config/alex_pruning.yaml'
    train_loader = get_dataloader()
    compression_scheduler = distiller.file_config(model, optimizer, compress, None, None)

    print('summary...')
    distiller.model_summary(model, 'compute', shape=(1, 3, 224, 224)) 

    print('sensitivity...')
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sparsities = np.arange(0, 0.6, 5)
    sensitivity = distiller.perform_sensitivity_analysis(model, which_params, sparsities, test_func=None, group='filter')
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')


    # distiller.sparsity()
    # distiller.density()

    
    c += 1

    for epoch in range(0, 100):
        compression_scheduler.on_epoch_begin(epoch, metrics=(vloss if (epoch != 0) else 10**6))
        train(train_loader, model, criterion, optimizer, epoch, compression_scheduler)
        compression_scheduler.on_epoch_end(epoch, optimizer)


