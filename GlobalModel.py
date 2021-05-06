# Collaborative Distributed Machine Learning Meets Optimization for  Intelligence Edge Offloading in 5G O-RAN
# Author: Anselme
# Python 3.6.4
########################################################################################################################
# Needed packages
from __future__ import division
import matplotlib
import math
import time
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Activation, Flatten, ReLU, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings("ignore")
import scipy.stats as stats
from scipy import special
import matplotlib.ticker as mtick
from matplotlib.pyplot import subplots, show
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

# Starting time
start_time = time.time()
np.random.seed(100)
# Loading the dataset
dataset = pd.read_csv('dataset/User_traffic_data.csv', usecols=[8], engine='python')
dataset2 = pd.read_csv('dataset/User_traffic_data.csv', engine='python')
dataset2.drop(dataset.filter(regex="Unname"),axis=1, inplace=True)

print(dataset.columns)
print(dataset.head(5))
#find absolute value of z-score for each observation
z = np.abs(stats.zscore(dataset))

#only keep rows in dataframe with all z-scores less than absolute value of 3
dataset = dataset[(z<3).all(axis=1)]

#find Q1, Q3, and interquartile range for each column
Q1 = dataset.quantile(q=.25)
Q3 = dataset.quantile(q=.75)
IQR = dataset.apply(stats.iqr)

#only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
dataset  = dataset[~((dataset < (Q1-1.5*IQR)) | (dataset > (Q3+1.5*IQR))).any(axis=1)]
dataset_test_unmodified = dataset
print("dataset_test_unmodified", dataset_test_unmodified.shape)
plt.plot(dataset[:240])
plt.ylabel('Video Traffic (Mbps)')
plt.xlabel('Time (Minutes)')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
plt.show()
#abel='Actual network traffic'
df = dataset2.rename(columns={'video_app_flux': 'Monthly video traffic', 'video_app_flux_day': 'Daily video traffic'})
plot_cols = ['Monthly video traffic', 'Daily video traffic']
plot_features = df[plot_cols]
plot_features.index = sorted(df['Time'])
_ = plot_features.plot(subplots=True)
plt.ylabel('Video Traffic (Mbps)')
plt.show()
plot_features = df[plot_cols][:480]
plot_features.index = sorted(df['Time'][:480])
_ = plot_features.plot(subplots=True)
plt.ylabel('Video Traffic (Mbps)')
plt.show()
# 9/4/2019  12:00:00 PM to 9/5/2019  3:57:00 AM
plt.hist2d(dataset2['video_app_flux'], dataset2['video_app_flux_day'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Monthly video traffic(Mbps)')
plt.ylabel('Daily video traffic(Mbps)')
plt.show()
dataset = dataset.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split data into train and test data sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1


look_back = 60
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
# Using LSTM
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model.save('C:/Users/anselme/Documents/5G_network/Dataset/673423_1184752_bundle_archive/GMLModel.h5', overwrite=True,
           include_optimizer=True)
history = model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)
# list all data in history
print(history.history.keys())
scores = model.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# Source code https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Testing  data
X_test = []
for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i + look_back), 0]
    X_test.append(a)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
predicted_network_traffic = model.predict(X_test)
predicted_network_traffic = scaler.inverse_transform(predicted_network_traffic)
predicted_traffic = predicted_network_traffic

i = 0
future_traffic = []
for x in range(len(predicted_traffic)):
    t = predicted_traffic[x]
    t = np.asarray(t)
    t = t[0]
    # t = float("{0:.2f}".format(t))
    future_traffic.append(t)

# print(len(predicted_traffic))
append_size = len(dataset_test_unmodified)-len(future_traffic)
for n in range(append_size):
    future_traffic.insert(0, 0)
#print(future_traffic)
#print(len(future_traffic))

##############################

# Using ARIMA
dataset_ARIMA = dataset_test_unmodified['video_app_flux_day']
dataset_ARIMA = dataset_test_unmodified.iloc[:240].values
print("dataset_ARIMA ", dataset_ARIMA)
history_ARIMA = []
for x in range(len(dataset_ARIMA)):
    y = dataset_ARIMA[x]
    y = np.asarray(y)
    y = y[0]
    # y = float("{0:.2f}".format(y))
    history_ARIMA.append(y)
predictions_ARIMA = list()
for t in range(len(dataset_ARIMA)):
    model2 = ARIMA(history_ARIMA, order=(5, 1, 0))
    model_fit = model2.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions_ARIMA.append(yhat)
    obs = dataset_ARIMA[t]
    history_ARIMA.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(dataset_ARIMA, predictions_ARIMA)
mse = ((predictions_ARIMA - dataset_ARIMA) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('Test MSE: %.3f' % error)
dataset_test_unmodified['PredictedTraffic'] = future_traffic
FinalData = pd.merge(dataset2, dataset_test_unmodified, on="video_app_flux_day")
FinalData.to_csv('network_traffic_prediction.csv')

# Gaussian Processes regression
#########################################################################################################################
# Code source: https://towardsdatascience.com/getting-started-with-gaussian-process-regression-modeling-47e7982b534d#_
def f(x):
    """The function to predict."""
    return x * np.sin(x)
X = dataset_ARIMA
# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
#plt.figure()
#plt.plot(trainPredictPlot[:240], f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
#plt.plot(trainPredictPlot[:240], y, 'r.', markersize=10, label='Observations')
#plt.plot(trainPredictPlot[:240], y_pred, 'b-', label='Prediction')
#plt.xlabel('$x$')
#plt.ylabel('$f(x)$')
#plt.legend(loc='upper left')
#plt.show()


#######################################################################################################################
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset[:240]),label='Actual Network Traffic')
plt.plot(trainPredictPlot[:240], markerfacecolor='red', label="Predicted Network Traffic")
plt.legend(loc="upper left")
plt.ylabel('Network traffic (Mbps)')
plt.xlabel('Time (Minutes)')
plt.grid(color='gray', linestyle='dashed')
plt.show()

# For plotting
predictions_ARIMA0 = predictions_ARIMA[look_back:240]
predictions_ARIMA = np.empty_like(dataset_ARIMA)
predictions_ARIMA[:, :] = np.nan
predictions_ARIMA[look_back:len(predictions_ARIMA)+look_back, :] = predictions_ARIMA0

predictions_GPR = y_pred

plt.plot(scaler.inverse_transform(dataset[:240]), linewidth=2.0,  label='Actual network traffic')
plt.plot(trainPredictPlot[:240], linewidth=2.0, color='yellowgreen', label='Predicted network traffic (GAIM)')
plt.plot(predictions_ARIMA[:240], linewidth=2.0, color='red', label='Predicted network traffic (ARIMA)')
plt.plot(predictions_GPR[:240], linewidth=2.0, color='black', label='Predicted network traffic (GPR)')
plt.grid(color='gray')
plt.xlabel('Time (Minutes)')
plt.ylabel('Network traffic (Mbps)')
plt.legend(title='')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
#plt.ylim(0, 290)
plt.show()
