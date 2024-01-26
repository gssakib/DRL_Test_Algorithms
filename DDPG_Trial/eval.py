import time
import pandas as py

import data
import ddpg_tf
from ddpg_tf import Agent

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from data import Import_Data
from utils import plotLearning, plotRPMandPrediction
from keras.utils import to_categorical

from train3 import y_predicted

print(y_predicted[3])











