import torch as T

def convert_arrays_to_tensors(array, device):
    tensors = []
    for arr in array:
        tensors.append(T.tensor(arr).to(device))
    return tensors

def clip_reward(r):
    if r>1:
        return 1
    elif r< -1:
        return -1
    else: 
        return r
    



