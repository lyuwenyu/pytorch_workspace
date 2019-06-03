import psutil
import ray
import numpy as np
import scipy.signal
import time

num_cpus = psutil.cpu_count(logical=False)

print(num_cpus)


ray.init(num_cpus=num_cpus)

@ray.remote
def f1(x):
    '''
    '''
    time.sleep(1)
    return x ** 2


@ray.remote
def create_martix(size):
    '''
    '''
    return np.random.normal(size=size)

@ray.remote
def f2(x, y):
    '''
    '''
    return np.dot(x, y)

if __name__ == '__main__':

    result_ids = []
    for i in [2, 3, 5, 1]:
        result_ids.append(f1.remote(i))
    results = ray.get(result_ids)
    print(results)


    x_id = create_martix.remote([100, 100])
    # x_id = ray.put(np.random.rand(100, 100))
    y_id = create_martix.remote([100, 200])
    z_id = f2.remote(x_id, y_id)
    z = ray.get(z_id)
    print(z.shape)

