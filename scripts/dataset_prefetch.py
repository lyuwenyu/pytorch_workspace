import random
from multiprocessing import Pool, Queue, Process
import json
import numpy as np 

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# raise IOError("image file is truncated")

import Augmentor

label_map = {
    "L_JSBTC 355ml": 0,
    "P_FJBPJ 330ml": 1,
    "P_FKGHXM 330ml": 2,
    "P_JSBBC 330ml": 3,
    "P_JSBMSPGQJ 330ml": 4,
    "P_JSBTC 330ml": 5,
    "P_JSBTC 640ml": 6,
    "P_JSBYLXD 330ml": 7,
    "P_KX1664BP 250ml": 8,
    "P_KX1664BP 330ml": 9,
    "P_KX1664FG 250ml": 10,
    "P_KX1664HP 250ml": 11,
    "P_KX1664JP 250ml": 12,
    "P_KX1664MG 250ml": 13,
    "P_KX1664SM 250ml": 14,
    "P_KX1664TH 330ml": 15,
    "P_LB 330ml": 16,
    "P_QD11DXP 330ml": 17,
    "P_XL 330ml": 18,
    "P_YJJPPJ11D 300ml": 19,
    "T_HEBCS 500ml": 20,
    "T_JSBPJ 500ml": 21,
    "T_JSBTC 500ml": 22,
    "T_KX1664HP 500ml": 23,
    "T_LB 330ml": 24,
    "T_QD11DYZ 330ml": 25,
    "T_QDAGT12 500ml": 26,
    "T_QDPJ 500ml": 27,
    "T_XL 330ml": 28,
    "T_YJ12DYJBP 500ml": 29,
    "T_YJPJ 330ml": 30,
    "T_YJXP 500ml": 31,
    "qita": 32
}



class Prefech(Process):
    
    def __init__(self, path, queue, is_training, label_map, batch_size, out_size):
        
        super(Prefech, self).__init__()

        with open(path, 'r') as f:
            raw_data = json.load(f)

        self.paths = raw_data['paths']
        self.classes = raw_data['classes']

        self.batch_size = batch_size
        self.out_size = out_size
        self.label_map = label_map
        self.is_training = is_training
        self.p = self._get_pipeline()
        self.queue = queue

        self._cur = 0
        self._perm = np.random.permutation(np.arange(len(self.paths)))
        
    def get_next_batch(self, ):
        
        if self._cur + self.batch_size >= len(self.paths):
            self._cur = 0
            self._perm = np.random.permutation(np.arange(len(self.paths))) 

        images = []
        labels = []

        for i in self._perm[self._cur: self._cur + self.batch_size]:

            image = Image.open(self.paths[i]).convert('RGBA')
            img_rgb = image.convert('RGB')
            img_msk = image.split()[-1]

            label = self.classes[i]
            if self.label_map is not None:
                label = self.label_map[label]

            if self.is_training:
                img_rgb, img_msk = self._sample_images_augmentor([img_rgb, img_msk], self.p)
                
            else:
                pass

            img_rgb = self._image_process(img_rgb)
            img_msk = img_msk.resize([self.out_size, self.out_size])
            img_msk = np.array(img_msk)
            img_msk[img_msk > 0] = 0
            
            image = np.concatenate([img_rgb, img_msk[:,:,np.newaxis]], axis=-1)
            
            images += [image]
            labels += [label]

        images = np.array(images)
        labels = np.array(labels)
        blob = {'images': images, 'labels': labels}

        return blob

    def _image_process(self, image):
        
        if isinstance(image, Image.Image):
            image = image.resize([self.out_size, self.out_size])
            image = np.array(image)
        image = (image - [123.0, 117.0, 104.0]) / 255.0
        
        return image

    def _get_pipeline(self):
        p = Augmentor.Pipeline()
        p.flip_left_right(probability=0.5)
        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
        p.random_distortion(probability=0.3, grid_height=5, grid_width=5, magnitude=2)
        p.skew(probability=0.5)
        # p.resize(probability=1.0, width=self.out_size, height=self.out_size)
        return p

    def _sample_images_augmentor(self, images, p, seed=-1, random_order=False):

        if seed > -1:
            p.set_seed(seed)

        if not isinstance(images, list):
            images = [images]

        if random_order:
            random.shuffle(p.operations)

        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        return images

    def run(self, ):
        while True:
            blob = self.get_next_batch()
            self.queue.put(blob)


class Dataset(object):

    def __init__(self, path, num_workers=5, is_training=True, batch_size=32, out_size=224):

        self.queue = Queue(num_workers)
    
        fetch_process = []
        for i in range(num_workers):
            fetch_process += [Prefech(path, self.queue, is_training=is_training, label_map=label_map, batch_size=batch_size, out_size=out_size)]
            fetch_process[i].start()

        def clean_up():
            for item in fetch_process:
                item.terminate()
                item.join()
        import atexit
        atexit.register(clean_up)

    def get_data(self, ):

        return self.queue.get()




if __name__ == '__main__':

    test = Dataset(path='beer_patch_with_mask_data_train.json', num_workers=5, batch_size=4, out_size=224)

    for i in range(100):
        blob = test.get_data()
        print(blob['images'].shape)

    
