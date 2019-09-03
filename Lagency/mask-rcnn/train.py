import torch
import torch.utils.data as data
import torch.optim as optim
from references.detection import engine
from references.detection import utils

from data.dataset import PennFudanPed, collate_fn
import core.models as models

dataset = PennFudanPed('./data/PennFudanPed')
indices = torch.randperm(len(dataset)).tolist()

train_dataset = data.Subset(dataset, indices=indices[:-50])
test_dataset = data.Subset(dataset, indices=indices[-50:])

train_dataloader = data.DataLoader(train_dataset, batch_size=3, num_workers=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = data.DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=True, collate_fn=collate_fn)

model = models.mask()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epoches = 10
device = torch.device('cuda:1')

model.to(device)

for epoch in range(num_epoches):
    engine.train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    lr_scheduler.step()
    engine.evaluate(model, test_dataloader, device=device)

print('done...')


