import torch as T

def convert_arrays_to_tensors(array, device):
    tensors = []
    for arr in array:
        tensors.append(T.tensor(arr).to(device))
    return tensors

def clip_rpm_prediction(rpm):
    if rpm>15:
        return 1
    elif rpm< -1:
        return -1
    else: 
        return rpm
    



