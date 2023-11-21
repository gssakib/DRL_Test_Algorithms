import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(errors, filename, x=None, window=5):   
    N = len(errors)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(errors[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Error')       
    plt.xlabel('Epoch')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)
