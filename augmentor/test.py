import Augmentor
from multiprocessing import Pool




def func(n):


    p = Augmentor.Pipeline('/home/wenyu/Desktop/test/')

    p.rotate(probability=0.7, max_left_rotation=12, max_right_rotation=12)
    p.flip_top_bottom(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.random_erasing(probability=0.7, rectangle_area=0.5)

    p.sample(n)


if __name__ == '__main__':


    pool = Pool(8)


    pool.map(func, [10, 10, 60])

    
    pool.close()
    pool.join()