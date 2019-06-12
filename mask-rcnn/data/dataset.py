import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import sys

from PIL import Image


class PennFudanPed(data.Dataset):
    def __init__(self, root,):
        self.root = root
        self.imgs = sorted(glob.glob(os.path.join(self.root, 'PNGImages', '*.png')))
        self.msks = sorted(glob.glob(os.path.join(self.root, 'PedMasks', '*.png')))
        self.totensor = transforms.ToTensor()

    def __len__(self, ):
        return len(self.imgs)

    def __getitem__(self, i):

        img = Image.open(self.imgs[i]).convert('RGB')
        msk = Image.open(self.msks[i])
        msk = np.array(msk)

        ids = np.unique(msk, )
        ids = ids[1:]
        masks = msk == ids[:, None, None]
        
        bboxes = []
        for j in range(len(ids)):
            pos = np.where(masks[j])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])

            bboxes += [[xmin, ymin, xmax, ymax], ]

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.ones((len(ids)), dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([i])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((len(ids), ), dtype=torch.long)

        target = {}
        target['bboxes'] = bboxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return self.totensor(img), target


def collate_fn(batch):
    '''
    '''
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


if __name__ == '__main__':
    
    dataset = PennFudanPed('./data/PennFudanPed')

    dataloader = data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0, collate_fn=collate_fn)

    for i, (images, targets) in enumerate(dataloader):

        # print(img)
        # print(len(img))
        # print(targets)
        print(targets[0].keys())
        for im in images:
            print(i, im.shape)
            
        break
