import yaml
from collections import OrderedDict
import torch
import torch.utils.data as data

__all__ = ['yaml_ordered_load', 'get_dataloader']


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Function to load YAML file using an OrderedDict
    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

    return yaml.load(stream, OrderedLoader)



class DummyDataset(data.Dataset):
    def __init__(self, ):
        pass
        
    def __len__(self, ):
        return 1000

    def __getitem__(self, i):
        data = torch.randn(3, 224, 224)
        target = torch.randint(0, 10, (1,))
        return data, target[0]

def get_dataloader():
    '''
    '''
    dataset = DummyDataset()
    dataloader = data.DataLoader(dataset, batch_size=10)
    return dataloader



if __name__ == '__main__':

    with open('./config/alex_pruning.yaml', 'r') as f:
        data = yaml_ordered_load(f)
        print(data['version'])
        