import collections
import functools
import os
import time
import datetime
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import scipy.stats as stats
from scipy import special
import seaborn as sns
import numpy as np
import pandas
# import cvxpy as cvx
# import dmcp
# from scipy.interpolate import interp1d
# from dmcp.fix import fix
# from dmcp.find_set import find_minimal_sets
# from dmcp.bcd import is_dmcp
import matplotlib.pyplot as plt
import tensorflow_federated as tff
from pandas import DataFrame
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import tensorflow as tf
# https://stackoverflow.com/questions/58965488/how-to-create-federated-dataset-from-a-csv-file
mpl.rcParams['axes.grid'] = True


# Starting time
start_time = time.time()
np.random.seed(100)
# Loading the dataset
dataset = pd.read_csv('User_traffic_data.csv', usecols=[6], engine='python')
dataset2 = pd.read_csv('User_traffic_data.csv', engine='python')
dataset2 = dataset2.loc[:, ~dataset2.columns.str.contains('^Unnamed')]



print(dataset.columns)
print(dataset.head(5))
print(dataset2.columns)
print(dataset2.head(5))

# We have 21118 users
total_users = dataset2['user_id'].nunique()
# We have 319 areas, we assume that each area is served by one O-RU
total_O_RU= dataset2['area_id'].nunique()
print("Number of Area", total_O_RU)

# We have 10  edge clouds, we assume that each edge cloud hosts near_RT_RIC and O-DU
Near_RT_RIC = dataset2['edge_cloud'].nunique()
print("Number of Near_RT_RIC ", Near_RT_RIC)
# We have 1 regional cloud, we assume that each regional cloud hosts nNo_Rear_RT_RIC
No_Rear_RT_RIC = 1

########################################################################################################################
# Start making global model
# find absolute value of z-score for each observation


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
plt.grid()
plt.show()

df = df[5::6]
df['Time'] = pd.to_datetime(df['Time'], infer_datetime_format=True)

date_time = pd.to_datetime(df.pop('Time'), format='%d.%m.%Y %H:%M:%S')


timestamp_s = date_time.map(datetime.datetime.timestamp)
# 9/4/2019  12:00:00 PM to 9/5/2019  3:57:00 AM
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

plt.hist2d(df['Monthly video traffic'], df['Daily video traffic'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Monthly video traffic(Mbps)')
plt.ylabel('Daily video traffic(Mbps)')
plt.show()

plot_cols = ['Monthly video traffic', 'Daily video traffic']
plot_features = df[plot_cols]
plot_features.index = date_time
plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

df = df.drop(['area_id', 'user_id'], axis=1)
print(df.head())
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
print(n)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

train_df = np.absolute(train_df)
val_df = np.absolute(val_df)
test_df = np.absolute(test_df)

print("train_df", len(train_df))
print("val_df ", len(val_df))
print("test_df", len(test_df))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split data into train and test data sets
train_size = int(len(dataset) * 0.63)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset),:]

train=np.absolute(train)
test= np.absolute(test)
print(len(train), len(test))


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['Daily video traffic'])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels


WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')


w2.example = example_inputs, example_labels


def plot(self, model=None, plot_col='Daily video traffic', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col}')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')


WindowGenerator.plot = plot
w2.plot()
plt.show()

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)
  ds = ds.map(self.split_window)
  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair
w2.train.element_spec


for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')



single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['Daily video traffic'])
single_step_window


class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]


baseline = Baseline(label_index=column_indices['Daily video traffic'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)


wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['Daily video traffic'])

wide_window

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(baseline)

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)


MAX_EPOCHS = 20

history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


wide_window.plot(linear)


dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

#val_performance['Dense'] = dense.evaluate(single_step_window.val)
#performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['Daily video traffic'])

conv_window

conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")
plt.show()

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
history = compile_and_fit(multi_step_dense, conv_window)


val_performance['MS dense'] = multi_step_dense.evaluate(conv_window.val)
performance['MS dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)


LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['Daily video traffic'])

wide_conv_window

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)
plt.show()
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)
history = compile_and_fit(lstm_model, wide_window)


val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm_model)

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]
plt.show()
plt.ylabel('MAE for video traffic prediction, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation dataset')
plt.bar(x + 0.17, test_mae, width, label='Test dataset')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()


########################################################################################################################