import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from core.datasets import dataset_factory
from core.models import crit_losses as crit
from core.models import pose_dla_dcn_origin as pose_dla_dcn 

from config import test_config as cfg
from PIL import Image
from PIL import ImageDraw
import numpy as np
import time
import glob
import random
print(cfg.dataset)

device = torch.device('cuda:1')

if __name__ == '__main__':

    dataset = dataset_factory.get_dataset(cfg.dataset)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    mm = pose_dla_dcn.get_pose_net('34', cfg.network['heads'], 1)
    print(mm)
    mm.load_state_dict(torch.load('./tmp/state-ckpt-epoch-00110')['model'])
    mm = mm.to(device)
    mm.eval()

    paths = glob.glob('../../../tmp/images/1/*.jpg')
    random.shuffle(paths)

    with torch.no_grad():
        for i, path in enumerate(paths[:10]):
            im = Image.open(path).resize((cfg.width, cfg.height))
            draw = ImageDraw.Draw(im)
            data = (np.array(im)/255. - cfg.mean) / cfg.std
            data = transforms.ToTensor()(data)
            data = data.unsqueeze(0).to(device, dtype=torch.float32)

            output = mm(data)
            heatmap = output['hm'].sigmoid()

            hm = heatmap.cpu().data.numpy()[0, 0]
            _im = np.floor(hm * 255)
            _im = Image.fromarray(_im).convert('L')
            # _im = _im.resize((cfg.width, cfg.height))
            _im.save(f'./tmp/{i}_hm.jpg')
            
            for jj, ii in zip(*np.where(hm > 0.5)):
                cx = cfg.stride * (ii + output['off'][0, 0, jj, ii].sigmoid()).item()
                cy = cfg.stride * (jj + output['off'][0, 1, jj, ii].sigmoid()).item()
                w =  cfg.stride * (output['wh'][0, 0, jj, ii]).item()
                h =  cfg.stride * (output['wh'][0, 1, jj, ii]).item()
                draw.rectangle((cx-w/2, cy-h/2, cx+w/2, cy+h/2), outline='red')

            im.save(f'tmp/{i}_im.jpg')

