import Augmentor
from multiprocessing import Pool
import random




def func(n, seed=0):

    p = Augmentor.Pipeline('/home/wenyu/Desktop/test/')
    
    # p.set_seed(seed)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.random_erasing(probability=0.5, rectangle_area=0.3)
    p.skew(probability=0.5)
    p.crop_random(probability=0.5, percentage_area=0.9)
    p.flip_left_right(probability=0.5)
    
    random.shuffle(p.operations)

    p.sample(n)


if __name__ == '__main__':


    # pool = Pool(3)
    # pool.map(func, [10, 10, 10])
    # pool.close()
    # pool.join()

    func(20)