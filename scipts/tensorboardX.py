import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data 
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


from data.BEAR import DatasetX
from model.network import RESNET


# torch.initial_seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)


NAME = 'bear'


NUM_CLASSES = 8

INIT_LR = 0.001
LOGS_STEP = 30

EPOCHS = 30
LR_STEP_SIZE = 10
SAVE_STEP = 10


TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 5

TRAIN_GLOBAL_STEP = 0
EVAL_GLOBAL_STEP = 0



TRAIN_DATA_FILE_PATH = './data/train.txt'
TEST_DATa_FILE_PATH = './data/test.txt'


writer = SummaryWriter()

GPU_ID = 0


model = RESNET(num_classes=NUM_CLASSES)
# model.load_state_dict( torch.load('./outputs/patch_with_context_resid_70.pth') )
print(model)

if torch.cuda.is_available():
    model = model.cuda(GPU_ID)

criteria = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.1)


def train(data_loader, e):

    model.train()

    for i, (x,y) in enumerate(data_loader):
        
        
        if torch.cuda.is_available():

            x = Variable(x.cuda(GPU_ID))
            y = Variable(y.cuda(GPU_ID))


        out = model(x)
        loss = criteria(out, y)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # ACCURACY
        pred = np.argmax(out.cpu().data.numpy(), axis=1)
        acc = np.sum(pred == y.cpu().data.numpy())*1.0 / x.size()[0]

        accs = _accuracy(out, y, topk=(1,))

        # LOGS
        if i%LOGS_STEP == 0:
            
            lll =  'epoch/iters {:0>2}/{:0>4} loss: {:.4} accuracy: {:.4}'.format(e, i, loss.cpu().data.numpy()[0], acc)
            print( lll )

            global TRAIN_GLOBAL_STEP
            TRAIN_GLOBAL_STEP += 1

            writer.add_scalar('/train/acc1', acc, global_step=TRAIN_GLOBAL_STEP)
            writer.add_scalar('/train/loss', loss, global_step=TRAIN_GLOBAL_STEP)
            writer.add_text('Text', lll, global_step=TRAIN_GLOBAL_STEP)
            
        if i%(LOGS_STEP*50) == 0:
            
            writer.add_image('/train/input_image', vutils.make_grid(x.cpu().data, normalize=True, scale_each=True), global_step=TRAIN_GLOBAL_STEP)
            


def validate(data_loader):
    model.eval()

    acc_num = 0
    n = 0

    for i, (x, y) in enumerate(data_loader):

        if torch.cuda.is_available():

            x = Variable(x.cuda(GPU_ID))
            y = Variable(y.cuda(GPU_ID))

        out = model(x)

        pred = np.argmax(out.cpu().data.numpy(), axis=1)
        acc_num += np.sum(pred == y.cpu().data.numpy())
        n += x.size()[0]

    print( '\n\n eval... accuracy: {} \n\n'.format(acc_num*1.0 / n) )

    global EVAL_GLOBAL_STEP
    EVAL_GLOBAL_STEP += 1

    writer.add_scalar('/eval/accc', acc_num*1.0 / n, global_step=EVAL_GLOBAL_STEP)
    writer.add_image('/eval/input', vutils.make_grid(x.cpu().data, normalize=True, scale_each=True), global_step=EVAL_GLOBAL_STEP)


def _accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == "__main__":


    dataset_train = DatasetX(TRAIN_DATA_FILE_PATH, is_training=True)
    dataset_test = DatasetX(TEST_DATa_FILE_PATH, is_training=False)

    loader_train = data.DataLoader(dataset_train, shuffle=True, batch_size=TRAIN_BATCH_SIZE, num_workers=5)
    loader_test = data.DataLoader(dataset_test, shuffle=False, batch_size=EVAL_BATCH_SIZE, num_workers=3)


    for e in range(EPOCHS):
        
        scheduler.step()
        train(loader_train, e)
        validate(loader_test)

        if e % SAVE_STEP == 0:

            model.cpu()
            torch.save(model.state_dict(), './outputs/{}_{:0>2}.pth'.format(NAME, e))
            model.cuda(GPU_ID)
    
    torch.save(model.cpu().state_dict(), './outputs/{}_final.pth'.format(NAME))

    writer.close()

    
    
    