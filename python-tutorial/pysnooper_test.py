import pysnooper

@pysnooper.snoop('./logs.txt')
def func(x):
    '''
    '''
    y = x + x

    if y > 10:
        y += 10
    else:
        y += 20

    return y


def func_p(x):
    '''
    '''
    y = x + x

    with pysnooper.snoop():
        if y > 10:
            y += 10
        else:
            y += 20

    return y



if __name__ == '__main__':

    x = 10
    z = func_p(x)

    print(z)