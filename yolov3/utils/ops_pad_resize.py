from PIL import Image
import numpy as np

def pad_resize(img, bbox=None, size=None):
    '''
    pad and resize image and corresponding bbox
    
    img: PIL 
    bbox: numpy
    '''

    w, h = img.size

    if size is None:
        size = [max(w, h)] * 2
    
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(scale * w), int(scale * h)
    img = img.resize((nw, nh), Image.BICUBIC)

    pad = (size[0] - nw) // 2, (size[1] - nh) // 2
    new_img = Image.new('RGB', size=size, color=(128, 128, 128))
    new_img.paste(img, pad)

    if bbox is not None:
        bbox = np.array(bbox)
        bbox[:, 0] = bbox[:, 0] * scale + pad[0]
        bbox[:, 1] = bbox[:, 1] * scale + pad[1]
        bbox[:, 2] = bbox[:, 2] * scale + pad[0]
        bbox[:, 3] = bbox[:, 3] * scale + pad[1]
        
        return new_img, bbox
    
    return new_img
    

if __name__ == '__main__':

    pass