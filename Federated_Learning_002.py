# Joint Federated Learning and Optimization for Intelligent Task Offloading
# Author: Anselme Ndikumana
# Python 3.8
#############################################################################################
import collections
import functools
import os
import time
import numpy
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import scipy.stats as stats
from scipy import special
import seaborn as sns
import numpy as np
import pandas
import matplotlib.pyplot as plt
import tensorflow_federated as tff
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(0)
# https://stackoverflow.com/questions/58965488/how-to-create-federated-dataset-from-a-csv-file
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

# Starting time
start_time = time.time()
np.random.seed(100)
# Loading the dataset
dataset = pd.read_csv('User_traffic_data.csv', usecols=[7], engine='python')
dataset2 = pd.read_csv('User_traffic_data.csv', engine='python')
dataset2.drop(dataset.filter(regex="Unname"), axis=1, inplace=True)

print(dataset.columns)
print(dataset.head(5))

# We have 21118 users
total_users = dataset2['user_id'].nunique()
# We have 319 areas, we assume that each area is served by one O-RU
total_O_RU = dataset2['area_id'].nunique()
print("Number of Area", total_O_RU)

# We have 10  edge clouds, we assume that each edge cloud hosts near_RT_RIC and O-DU
Near_RT_RIC = dataset2['edge_cloud'].nunique()
print("Number of Near_RT_RIC ", Near_RT_RIC)
# We have 1 regional cloud, we assume that each regional cloud hosts nNo_Rear_RT_RIC
No_Rear_RT_RIC = 1

########################################################################################################################
# Start making global model
# find absolute value of z-score for each observation
# Standard score, https://en.wikipedia.org/wiki/Standard_score
# find absolute value of z-score for each observation
z = np.abs(stats.zscore(dataset))

# only keep rows in dataframe with all z-scores less than absolute value of 3
dataset = dataset[(z < 3).all(axis=1)]

# find Q1, Q3, and interquartile range for each column
Q1 = dataset.quantile(q=.25)
Q3 = dataset.quantile(q=.75)
IQR = dataset.apply(stats.iqr)

# only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
dataset = dataset[~((dataset < (Q1-1.5*IQR)) | (dataset > (Q3+1.5*IQR))).any(axis=1)]
dataset_test_unmodified = dataset


dataset_test_unmodified = dataset
print("dataset_test_unmodified", dataset_test_unmodified.shape)
plt.plot(dataset[:240])
plt.ylabel('Video Traffic (Mbps)')
plt.xlabel('Time (Minutes)')
plt.grid()
plt.show()

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
plt.grid()
plt.show()

# 9/4/2020  12:00:00 PM to 9/5/2020  3:57:00 AM
plt.hist2d(df['Monthly video traffic'], df['Daily video traffic'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Monthly video traffic(Mbps)')
plt.ylabel('Daily video traffic(Mbps)')
plt.show()

fft = tf.signal.rfft(df['Monthly video traffic'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['Monthly video traffic'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 3000000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
plt.ylabel('Video traffic variation (Mbps)')
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


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 60

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


logdir1 = "/tmp/logs/scalars/training/fd1"
# To run TensorBoard, use the command
# tensorboard --logdir=/tmp/logs/scalars/training/fd1"

GAIM =  tf.keras.models.Sequential([
      tf.keras.layers.LSTM(32, input_dim=look_back),
      tf.keras.layers.Dense(1, activation='sigmoid')
])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir1, histogram_freq=1)
GAIM.compile(loss='mean_squared_error', optimizer='adam')
GAIM.summary()
history = GAIM.fit(trainX, trainY, epochs=200, verbose=2, callbacks=[tensorboard_callback])
GAIM.evaluate(testX, testY, verbose=2)
summary_writer = tf.summary.create_file_writer(logdir1)

# list all data in history
print(history.history.keys())
scores = GAIM.evaluate(trainX, trainY, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# Source code https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
model_json = GAIM.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
GAIM.save_weights("model.h5")
print("Saved model to disk")
trainPredict =GAIM.predict(trainX)
testPredict = GAIM.predict(testX)
print("trainX, trainY, testY, testY", trainX.shape, trainY.shape, testX.shape, testY.shape)
print("trainPredict, testPredict", trainPredict.shape, testPredict.shape)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY2 = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY2 = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY2[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY2[0], testPredict[:,0]))
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
X_test2 = []
for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i + look_back), 0]
    X_test2.append(a)
X_test2 = np.array(X_test2)
X_test2 = np.reshape(X_test2, (X_test2.shape[0], 1, X_test2.shape[1]))
predicted_network_traffic = GAIM.predict(X_test2)
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
FinalData.to_csv('dataset/network_traffic_prediction.csv')


#######################################################################################################################
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset[:240]),label='Actual Network Traffic')
plt.plot(trainPredictPlot[:240], markerfacecolor='red', label="Predicted Network Traffic (GMLM)")
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

#predictions_GPR = y_pred

plt.plot(scaler.inverse_transform(dataset[:240]), linewidth=2.0,  label='Actual network traffic')
plt.plot(trainPredictPlot[:240], linewidth=2.0, color='yellowgreen', label='Predicted network traffic')
#plt.plot(predictions_ARIMA[:240], linewidth=2.0, color='red', label='Predicted network traffic (ARIMA)')
#plt.plot(predictions_GPR[:240], linewidth=2.0, color='black', label='Predicted network traffic (GPR)')
plt.grid(color='gray')
plt.xlabel('Time (Minutes)')
plt.ylabel('Network traffic (Mbps)')
plt.legend(title='')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
plt.show()
loss_gloabal = history.history['loss']
plt.plot(history.history['loss'],linewidth=3)
plt.ylabel("Mean Squared Error Function")
plt.xlabel("Number of Epoch")
plt.grid(which='both')
plt.grid(color='gray', linestyle='dashed')
plt.show()

########################################################################################################################
# Generating a decentralized data for federated Learning
########################################################################################################################
# Between  Near_RT_RIC (Clients) and  Non_Near_RT_RIC (Server)

T_Near_RT_RIC = Near_RT_RIC
Near_RT_RIC_SAMPLE_SIZE = len(trainX)/T_Near_RT_RIC
step = len(trainX)/T_Near_RT_RIC

data_fed = [tf.data.Dataset.from_tensor_slices({"value": trainX[int(i*step):int((i+1)*step)],
                                                "label":trainY[int(i*step):int((i+1)*step)]})
            for i in range(T_Near_RT_RIC)]

# client dataset can be accesed as data_fed[ CLIENT_ID ]
example_dataset = data_fed[0]
#   def preprocess(data_fed):
#   def batch_format_fn(ele):

example_element = next(iter(example_dataset))
# example_element[0] refers to X val
print("example_element value", example_element["value"].numpy())
# example_element[1] refers to X val
print("example_element value", example_element["label"].numpy())



# preprocess

Near_RT_RIC = T_Near_RT_RIC
NUM_EPOCHS = 5
BATCH_SIZE = int(Near_RT_RIC_SAMPLE_SIZE // 2)
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(fl_dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(x=element["value"], y=element["label"])
    return fl_dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(
        PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data[x])
            for x in client_ids]

sample_batch["x"].shape
print("sample_batch.shape", sample_batch["x"].shape)
sample_clients = [x for x in range(Near_RT_RIC)]
federated_train_data1 = make_federated_data(data_fed, sample_clients)

print('Number of edge cloud datasets: {l}'.format(l=len(federated_train_data1)))
print('First dataset: {d}'.format(d=federated_train_data1[0]))

# Number of examples per layer for a sample of clients
f = plt.figure(figsize=(18, 13))
f.suptitle('Distribution of the dataset in some Near_RT_RICs')
for i in range(3):
    client_dataset = data_fed[i]
    client_dataset = next(iter(client_dataset))
    plot_data = client_dataset["value"].numpy()
    plot_data=plot_data[0]
    plt.subplot(3, 3, i+1)
    plt.title('Near RT RIC {}'.format(i))
    plt.ylabel("Normalized Network Traffic")
    plt.xlabel('Time (Minutes)')
    plt.grid(which='both')
    plt.plot(plot_data)
plt.show()


def model_fn():
    keras_model1 = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(32, input_dim=look_back),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
    return tff.learning.from_keras_model(keras_model1, input_spec=preprocessed_example_dataset.element_spec,
                                         loss=tf.keras.losses.MeanSquaredError(),
                                         metrics=[tf.keras.metrics.MeanSquaredError()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

str(iterative_process.initialize.type_signature)

state = iterative_process.initialize()
LAIM = tf.keras.models.clone_model(GAIM)
LAIM.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=[tf.keras.metrics.MeanSquaredError()])

loss_EC = []
def keras_evaluate1(state, round_num):
  state.model.assign_weights_to(LAIM)
  loss, accuracy = LAIM.evaluate(testX, testY, steps=2, verbose=2)
  print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
  loss_EC.append(loss)




NUM_ROUNDS=201
for round_num in range(1, NUM_ROUNDS):
    Near_RT_RICs = list(range(1, Near_RT_RIC))
    federated_train_data1 = make_federated_data(data_fed, Near_RT_RICs)
    print("ROUND ", round_num)
    keras_evaluate1(state, NUM_ROUNDS + 1)
    state, metrics = iterative_process.next(state, federated_train_data1)
    print('\tTrain: loss={}, accuracy={} '.format(round_num, metrics["train"]["loss"],
                                                  metrics["train"]["mean_squared_error"]))

keras_evaluate1(state, NUM_ROUNDS + 1)

# Federated Learning
state.model.assign_weights_to(LAIM)
Y_pred_tff = LAIM.predict(testX).ravel()
#################################################
# make predictions
trainPredict_tff_1 = LAIM.predict(trainX)
testPredict = LAIM.predict(testX)
# invert predictions
trainPredict_tff_1 = scaler.inverse_transform(trainPredict_tff_1)
trainY_tff_1 = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_tff_1 = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_tff_1[0], trainPredict_tff_1[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_tff_1[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredict_tff_1Plot = numpy.empty_like(dataset)
trainPredict_tff_1Plot[:, :] = numpy.nan
trainPredict_tff_1Plot[look_back:len(trainPredict_tff_1)+look_back, :] = trainPredict_tff_1
# shift test predictions for plotting
testPredictPlot1 = numpy.empty_like(dataset)
testPredictPlot1[:, :] = numpy.nan
testPredictPlot1[len(trainPredict_tff_1)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset[:240]),linewidth=2.0,  label='Actual network traffic')
plt.plot(trainPredict_tff_1Plot[:240],linewidth=2.0,  label='Predicted network traffic')
plt.plot(testPredictPlot1[:240], linewidth=2.0,  label='Predicted network traffic')
plt.show()


##################################


# To run TensorBoard, use the command
# tensorboard --logdir=/tmp/logs/scalars/training/fd2"
logdir2 = "/tmp/logs/scalars/training/fd2"
summary_writer = tf.summary.create_file_writer(logdir2)
state = iterative_process.initialize()

with summary_writer.as_default():
    for round_num in range(1, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data1)
        for name, value in metrics['train'].items():
            tf.summary.scalar(name, value, step=round_num)



########################################################################################################################

# Between  Near_RT_RIC (server) and  edge devices (clients)
# We consider one device in each area served by 0-RU

T_Edge_device = int(total_O_RU//10)
Edge_device_SAMPLE_SIZE = len(trainX)/T_Edge_device
step = len(trainX)/T_Edge_device

data_fed_edge_device = [tf.data.Dataset.from_tensor_slices({"value": trainX[int(i*step):int((i+1)*step)],
                                                            "label":trainY[int(i*step):int((i+1)*step)]})
            for i in range(T_Edge_device)]

# client dataset can be accesed as data_fed_edge_device[ CLIENT_ID ]
example_dataset_edge_device = data_fed_edge_device[0]
#   def preprocess(data_fed_edge_device):
#   def batch_format_fn(ele):

example_element_edge_device = next(iter(example_dataset_edge_device))
# example_element[0] refers to X val
print("example_element value at edge device", example_element_edge_device["value"].numpy())
# example_element[1] refers to X val
print("example_element value at edge device", example_element_edge_device["label"].numpy())

# preprocess

Edge_device = T_Edge_device
NUM_EPOCHS = 5
BATCH_SIZE = int(Edge_device_SAMPLE_SIZE // 2)
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess_edge_device(fl_dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(x=element["value"], y=element["label"])
    return fl_dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(
        PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess_edge_device(example_dataset_edge_device)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))


def make_federated_data_edge_device(client_data, client_ids):
    return [preprocess(client_data[x])
            for x in client_ids]

sample_batch["x"].shape
print("sample_batch.shape", sample_batch["x"].shape)
sample__edge_devices = [x for x in range(Edge_device)]
federated_train_data2 = make_federated_data_edge_device(data_fed_edge_device, sample__edge_devices)

print('Number of edge devices: {l}'.format(l=len(federated_train_data2)))
print('First dataset: {d}'.format(d=federated_train_data2[0]))


def model_fn2():
    keras_model2 = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(32, input_dim=look_back),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
    return tff.learning.from_keras_model(keras_model2, input_spec=preprocessed_example_dataset.element_spec,
                                         loss=tf.keras.losses.MeanSquaredError(),
                                         metrics=[tf.keras.metrics.MeanSquaredError()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn2,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

str(iterative_process.initialize.type_signature)

state = iterative_process.initialize()
OAIM = tf.keras.models.clone_model(LAIM)
OAIM.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=[tf.keras.metrics.MeanSquaredError()])

loss_edge_devices = []
def keras_evaluate2(state, round_num):
  state.model.assign_weights_to(OAIM)
  loss, accuracy = OAIM.evaluate(testX, testY, steps=2, verbose=2)
  print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
  loss_edge_devices.append(loss)


NUM_ROUNDS = 201
for round_num in range(1, NUM_ROUNDS):
    Edge_devices = list(range(1, Edge_device))
    federated_train_data2 = make_federated_data_edge_device(data_fed_edge_device, Edge_devices)
    print("ROUND ", round_num)
    keras_evaluate2(state, NUM_ROUNDS + 1)
    state, metrics = iterative_process.next(state, federated_train_data2)
    print('\tTrain: loss={}, accuracy={} '.format(round_num, metrics["train"]["loss"],
                                                  metrics["train"]["mean_squared_error"]))

keras_evaluate2(state, NUM_ROUNDS + 1)

# Federated Learning
state.model.assign_weights_to(OAIM)
Y_pred_tff = OAIM.predict(testX).ravel()
Y_pred_tff.reshape(-1,1)
#################################################
# make predictions
trainPredict_tff_2 = OAIM.predict(trainX)
testPredict = OAIM.predict(testX)
# invert predictions
trainPredict_tff_2 = scaler.inverse_transform(trainPredict_tff_2)
trainY_tff_2 = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_tff_2 = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_tff_2[0], trainPredict_tff_2[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_tff_2[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredict_tff_2Plot = numpy.empty_like(dataset)
trainPredict_tff_2Plot[:, :] = numpy.nan
trainPredict_tff_2Plot[look_back:len(trainPredict_tff_2)+look_back, :] = trainPredict_tff_2
# shift test predictions for plotting
testPredictPlot2 = numpy.empty_like(dataset)
testPredictPlot2[:, :] = numpy.nan
testPredictPlot2[len(trainPredict_tff_2)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset[:240]),linewidth=2.0,  label='Actual network traffic')
plt.plot(trainPredict_tff_2Plot[:240],linewidth=2.0,  label='Predicted network traffic')
plt.xlabel('Time (Minutes)')
plt.ylabel('Network traffic (Mbps)')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
plt.show()

plt.plot(trainPredictPlot[:240], linewidth=2.0 , label='Predicted network traffic (GMLM)')
plt.plot(trainPredict_tff_1Plot[:240], linewidth=2.0,  label='Predicted network traffic (LMLM)')
plt.plot(trainPredict_tff_2Plot[:240], linewidth=2.0,  label='Predicted network traffic (OMLM)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Network traffic (Mbps)')
plt.grid(which='both')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
plt.show()

plt.plot(trainPredict_tff_1Plot[:240], linewidth=2.0,  label='Predicted network traffic (LMLM)')
plt.plot(trainPredict_tff_2Plot[:240], linewidth=2.0,  label='Predicted network traffic (OMLM)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Network traffic (Mbps)')
plt.grid(which='both')
plt.grid(color='gray', linestyle='dashed')
plt.legend(loc="upper left")
plt.show()


plt.plot(loss_gloabal, linewidth=2.0,  label='GMLM')
plt.plot(loss_EC, linewidth=2.0,  label='LMLM')
plt.plot(loss_edge_devices,linewidth=2.0,  label='OMLM')
plt.ylabel("Mean Squared Error")
plt.legend(loc="upper left")
plt.xlabel("Number of Epoch")
plt.grid(which='both')
plt.grid(color='gray', linestyle='dashed')
plt.show()
##################################
# To run TensorBoard, use the command
# tensorboard --logdir=/tmp/logs/scalars/training/fd3"
logdir3 = "/tmp/logs/scalars/training/fd3"
summary_writer = tf.summary.create_file_writer(logdir3)
state = iterative_process.initialize()

with summary_writer.as_default():
    for round_num in range(1, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data2)
        for name, value in metrics['train'].items():
            tf.summary.scalar(name, value, step=round_num)

end_time = time.time()
End_learning_time = end_time - start_time
print("End of HFL", End_learning_time)
