import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, x=None, window=500):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Epoch')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

def plotRPMandPrediction(rpms,predictions,filename, x=None, window=100):
    if x is None:
        x= range(len(rpms))
    
    #Thin the data by plotting every 100th point 
    thin_factor = 100
    x_thinned = x[::thin_factor]
    rpms_thinned = rpms[::thin_factor]
    predictions_thinned = predictions[::thin_factor]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rpms, label='Actual RPM', color='blue')
    plt.plot(predictions, label='Predicted RPM', color='red', alpha = 0.5)

    # Adding title and labels
    plt.title('RPM vs Predicted RPM')
    plt.xlabel('Time')
    plt.ylabel('RPM')
    plt.plot(rpms, predictions)
    plt.savefig(filename)

#def calculateAccuracy(correctPredictions, totalPredictions):
