from PIL import Image
from PIL import ImageDraw


import time


def show_array(arr, name=None, save='True'):
    '''
    '''
    _im = Image.fromarray(arr)
    if save:
        name = name if name else time.time()
        _im.save(f'{name}.png')
    else:
        _im.show()
