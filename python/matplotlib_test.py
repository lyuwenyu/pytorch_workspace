import matplotlib.pyplot as plt
import numpy as np 
import random



def func1():

    times = [10, 20, 30, 50, 50, 60]
    x = list(range(1, len(times)+1))

    fig, ax = plt.subplots()

    # plt.plot(list(range(0, len(times))), times, label=str(i))
    plt.bar(x, times, width=0.45, align='center', color='c')
    plt.bar(x, times, width=0.45, align='center', color='g', bottom=times)

    for a, b in zip(x, times):
        plt.text(a+0.2, b+0.1, str(int(b)), ha='center', va='bottom', fontsize=8) # horizontal / verhical alignment

    plt.title('time consuming of n perspective skew images.')
    plt.xlabel('n_augmentation')
    plt.ylabel('ms')
    
    plt.xticks(x, ['a', 'b', 'c', 'd', 'e', 'f'], rotation=30)
    plt.yticks(np.arange(0, 200, 10))
    # ax.set_yticks()

    plt.ylim([0, 200])
    plt.xlim([0, 8])
    # ax.set_xlim()

    plt.legend(['a', 'b'], loc='upper right', fontsize=10)

    plt.grid()
    plt.show()



if __name__ == '__main__':

    func1()