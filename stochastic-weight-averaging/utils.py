import torch


def moving_average(model_swa, model_current, alpha=1.):

    for p1, p2 in zip(model_swa.parameters(), model_current.parameter()):
        p1 *= (1.0 - alpha)
        p1 += p2 * alpha
    
