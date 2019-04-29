from .detdataset import DetDataset
from .anno_parser import labelme_parser
from .data_augmentor import bbox_augmentation

def get_dataset(cfg):
    '''
    '''
    if cfg['task'] == 'detection':
        dataset = DetDataset
    
    if cfg['type'] == 'bbox' and cfg['augmentor']:
        augmentor = bbox_augmentation(cfg['augmentor'])

    if cfg['anno_format'] == 'labelme':
        parser = labelme_parser
    
    return dataset(cfg, parser=parser, augmentor=augmentor)
