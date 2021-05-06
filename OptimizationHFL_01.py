# Joint Federated Learning and Optimization for Intelligent Task Offloading
# Author: Anselme Ndikumana
# Python 3.8
# Please install cvxpy==1.1.0 for DMCP

"""
Optimization approach: Disciplined Multi-Convex Programming
"""
import matplotlib
import numpy as np
import random
from scipy.interpolate import interp1d
import pandas as pd
from pandas import DataFrame
from random_words import RandomWords
import math as mt
import math
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import time
import lfucache.lfu_cache as lfu_cache  # LFU replacement policy from https://github.com/laurentluce/lfu-cache
np.set_printoptions(precision=10)
from scipy import special
from keras.models import model_from_json
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt;
from matplotlib.pyplot import subplots, show
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cvx
import dmcp
from scipy.interpolate import interp1d
from dmcp.fix import fix
from dmcp.find_set import find_minimal_sets
from dmcp.bcd import is_dmcp
# We use seed to reproduce the same results
np.random.seed(1000)
from scipy.interpolate import make_interp_spline, BSpline
import warnings
warnings.filterwarnings("ignore")


np.random.seed(100)
# Loading the dataset
start_learning_time = time.time()
dataset = pd.read_csv('User_traffic_data.csv', usecols=[7], engine='python')
dataset2 = pd.read_csv('User_traffic_data.csv', engine='python')
dataset2.drop(dataset.filter(regex="Unname"),axis=1, inplace=True)

print(dataset.columns)

#find absolute value of z-score for each observation
z = np.abs(stats.zscore(dataset))

#only keep rows in dataframe with all z-scores less than absolute value of 3
dataset = dataset[(z<3).all(axis=1)]

#find Q1, Q3, and interquartile range for each column
Q1 = dataset.quantile(q=.25)
Q3 = dataset.quantile(q=.75)
IQR = dataset.apply(stats.iqr)

# only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
dataset = dataset[~((dataset < (Q1-1.5*IQR)) | (dataset > (Q3+1.5*IQR))).any(axis=1)]
dataset_test_unmodified = dataset


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
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


# loop back  of 60 min
look_back = 60
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])
scores = loaded_model.evaluate(trainX, trainY, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
# Accuracy

trainPredict = loaded_model.predict(trainX)
testPredict = loaded_model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# Calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
theta = testScore
accuracy_threshold = 90
#if accuracy_threshold>=theta:



print('Accuracy', theta)
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
predicted_network_traffic = loaded_model.predict(X_test)
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

dataset_test_unmodified['PredictedTraffic'] = future_traffic
FinalData = pd.merge(dataset2, dataset_test_unmodified, on="video_app_flux_day")
FinalData.to_csv('C:/Users/anselme/Documents/5G_network/Dataset/673423_1184752_bundle_archive/network_traffic_prediction.csv')
End_learning_time = time.time()
Total_learning_Time = End_learning_time-start_learning_time
print('Total_learning_Time', Total_learning_Time)

# To get predicted traffic
predicted_traffic = pd.read_csv('C:/Users/anselme/Documents/5G_network/Dataset/673423_1184752_bundle_archive/'
                                'network_traffic_prediction.csv', engine='python')
predicted_traffic = predicted_traffic.reset_index(drop=True)
predicted_traffic.drop(predicted_traffic.filter(regex="Unname"), axis=1, inplace=True)
predicted_traffic.replace([np.inf, -np.inf], np.nan)
predicted_traffic.dropna(inplace=True)
df = DataFrame(predicted_traffic, columns=['Time', 'area_id', 'user_id', 'edge_cloud', 'PredictedTraffic'])

# At Regional Cloud
total_number_regional_cloud = 1  # The total number of regional cloud
p_d = random.randint(2500000000,
                     3900000000) # Computation capacity at regional cloud in terms of Cycles Per Second or Hertz  # (from 2.5 GHz to 3.9 GHz)
c_d = random.uniform(1000000, 5000000)  # Storage capacity at regional cloud in terms of Megabyte
cache_capacity_allocation_dc = []  # Cache  allocation  at DC
computation_capacity_allocation_dc = []  # Computation allocation at DC
DC_caching_decision_variable = []  # Cache decision variable
DC_computation_decision_variable = []  # Computation decision variable
execution_latency_dc_vector = []  # Execution latency_dc

# At Edge Cloud
EdgeCloud_RegionalCloud = df.groupby(['edge_cloud'])['PredictedTraffic'].agg('sum').reset_index()
count_EdgeCloud = EdgeCloud_RegionalCloud.shape[0]  # To get number of edge cloud
# print("Number of edge cloud", count_EdgeCloud)
cache_capacity_EC = []  # Cache capacity of edge cloud
computation_capacity_EC = []  # Computation capacity of edge cloud
communication_EC_RC = []  # Communication capacity of edge cloud
percentage_radio_spectrum_vector = []  # Fraction of communication resource allocation
cached_content = lfu_cache.Cache()  # Contents need to be cached at EC
# Collaboration space variales
EC_m_EC_n_offloading_variable = []
EC_m_computation_allocation_variable = []
EC_n_computation_allocation_variable = []
EC_m_cache_allocation_variable = []
EC_n_cache_allocation_variable = []
EC_server_maximum_capacity_caching = []
computation_capacity_allocation_EC = []
EC_server_maximum_capacity_computation = []
total_executing_ec_n_array = []
total_executing_ec_m_array = []
EC_m_cloud_offloading_variable = []
cache_capacity_allocation_EC = []
transm_delay_between_ru = []
Bandwidth_EC_m_EC_n = random.randint(3000, 3500)  # Bandwidth  between EC m and EC n in terms of Mbps

for i in range(count_EdgeCloud):
    cache_capacity_EC.append(random.uniform(100000, 500000))  # Caching capacity of edge cloud (EC) in Megabyte
    computation_capacity_EC.append(
        random.uniform(2500000, 39000000))  # Computation capacity of edge cloud (EC) in Megabyte
    communication_EC_RC.append(random.uniform(3000, 3500))  # Overall Bandwith between EC m and RC n in terms of Mbps

# At RUs

RU_EdgeCloud = df.groupby(['area_id'])['PredictedTraffic'].agg('sum').reset_index()
count_RU = RU_EdgeCloud.shape[0]  # Find the number of 0-RU
# print("Number of RU", count_RU)

Fiber_Fronthaul_RU_EC = []
for i in range(count_RU):
    Fiber_Fronthaul_RU_EC.append(random.uniform(2000, 2500))  # Overall fronthaaul Bandwith between O-RU m and 0-DU
    # in terms of Mbps

# At end-user device
User_RU = df.groupby(['user_id'])['PredictedTraffic'].agg('sum').reset_index()
count_User = User_RU.shape[0]  # Find the total number of users
ContentName = User_RU['user_id'].values
User_RU = User_RU['PredictedTraffic'].values  # Predicted network traffic from users

User_RU = User_RU.astype(int)  # Make sure that the traffic that reaches 0-RU
# print(User_RU, type(User_RU))
# print("Number of users", count_User)
# Wireless Communication Model
estimated_users_per_ru = int(count_User / count_RU)  # The total numbers of users per O-RU
count_User = estimated_users_per_ru
User_ORU_decision_variable = 1  # We assume that each user is connected to at least one O-RU

# Initialize the wireless environment and some other veriables.

N_0 = mt.pow(10, ((-169 / 3) / 10))  # Noise power spectrum density is -169dBm/Hz;
eta_los = 1  # Loss corresponding to the LoS connections
eta_nlos = 20  # Loss corresponding to the NLoS connections
path_loss_factor = eta_los - eta_nlos  #
C = 20 * np.log10(4 * np.pi * 9 / 3) + eta_nlos  #  \carrier frequncy is # 900Mhz=900*10^6,
# and light speed is c=3*10^8; then one has f/c=9/3;
transmission_power = 5 * mt.pow(10, 5)  # maximum uplink transimission power of one GT is 5mW;


# Get users' locations and calculate communication resources
ground_distance = 40  # Estimated distance between edge device and O-RU in terms of metres


def get_ORU_location():
    random_theta = np.random.uniform(0.0, 1.0, size=120) * 2 * np.pi
    random_radius = ground_distance * np.sqrt(np.random.uniform(0.0, 1.0, size=120))
    x_radius = random_radius * np.cos(random_theta)
    y_radius = random_radius * np.sin(random_theta)
    uniform_distance_points = [(x_radius[i], y_radius[i]) for i in range(120)]
    select_location = np.random.randint(100, 120)
    x_uniform = uniform_distance_points[select_location][0]
    y_uniform = uniform_distance_points[select_location][1]
    user_location = math.sqrt(x_uniform ** 2 + y_uniform ** 2)
    return user_location


def communication_resources_User_oru():
    distance_User_ORU = get_ORU_location()
    # http://www.raymaps.com/index.php/lte-data-rate-calculation/
    # 1 Time slot = 0.5 ms(i.e 1 Sub-frame = 1 ms)
    # 1 Time slot = 7 Modulation Symbols
    # Modulation Symbol = 6 bits if 64 QAM is used as modulation scheme
    Number_symbols_per_time_slot = 7
    Bits_per_symbol = 6
    Duration_of_time = 0.5e-3
    Data_rate_single_carrier = Number_symbols_per_time_slot * Bits_per_symbol/ Duration_of_time
    Data_rate_all_carrier = Data_rate_single_carrier*1200 # if we consider 100RBs
    # If we  assume 4×4 MIMO is used then the capacity would increase four fold
    wireless_bandwidth= Data_rate_all_carrier
    # Fading
    sigma = 7  # Standard  deviation[dB]
    mu = 0  # Zero mean
    tempx = np.random.normal(mu, sigma, estimated_users_per_ru)
    x_com = np.mean(tempx)  # In term of dBm
    PL_0_dBm = 34  # In terms of dBm;
    PL_dBm = PL_0_dBm + 10 * path_loss_factor * math.log10(distance_User_ORU / ground_distance) + x_com
    path_loss = 10 ** (PL_dBm / 10)  # [milli - Watts]
    channel_gain = transmission_power - path_loss
    channel_gain = float(channel_gain)
    spectrum_efficiency = User_ORU_decision_variable * wireless_bandwidth * \
                          math.log(1 + (transmission_power * channel_gain ** 2)/(wireless_bandwidth + N_0))
    return channel_gain, wireless_bandwidth, spectrum_efficiency


channel_gain, wireless_bandwidth, spectrum_efficiency = communication_resources_User_oru()

active_user_list = []
storage_user = []
communication_user = []
number_user_application = []
user_cache_allocation_variable = []
cpu_energy = []
computation_user_deadline = []
cpu_arc_parameter = 2  # constant parameter that is related to CPU hardware architecture
cache_capacity_user = []
computation_capacity_user = []
content_name_user = []
user_offloading_variable = []
cache_capacity_allocation_user = []
instantaneous_data_vector = []
transm_delay = []
computation_capacity_allocation_user = []
local_computation_cost = []
user_execution_latency_array = []
input_data_vector = []
output_data = []
video_format0 = [".avi", ".mpg"]
video_format_user = []
end_user_device_stutus = []
computation_user_requirement = []
for i in range(count_User):
    computation_capacity_user.append(
        random.randint(15000, 110000))   # Computation capacity at each end-user in terms of Cycles Per Second/Hz

    computation_user_requirement.append(
        random.uniform(4525, 7375))   # computation requirement for end-users in terms of cycles/byte
    cache_capacity_user.append(
        random.uniform(1000, 64000))  # Storage capacity at each end-user device in terms of Megabyte
    cpu_energy.append(random.randint(3, 4))  # Energy at end-user device in terms W/GhZ
    computation_user_deadline.append(random.uniform(0.0005, 1))  # computation deadline in terms of seconds
    number_user_application.append(1)  # Number of applications at each end-user device


#######################################################################
# For regional cloud, we consider data center as regional cloud

class DataCenter(object):
    def __init__(self, number_rc_server, end_user, input_data, computation_deadline, computation_requirement_user
                 , p_d, c_d):
        self.number_rc_server = number_rc_server
        self.p_d = p_d
        self.c_d = c_d
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user = computation_requirement_user
        self.end_user = end_user

    def cache_dc(self, end_user, input_data, output_data0, c_d, Name_offloaded_data):
        """The  caching processes takes users input data and computation output, then cache them"""
        print("Caching at Data center is done at %d%%" % (random.randint(50, 99)))
        c_kd = end_user * (input_data + output_data0)
        cache_capacity_allocation_dc.append(c_kd)
        cached_content.insert(Name_offloaded_data, output_data0)
        DC_caching_decision_variable.append(1)
        return cached_content, DC_caching_decision_variable, cache_capacity_allocation_dc

    def compute_dc(self, end_user, input_data, computation_requirement_user, transm_delay_EC_dc, Name_offloaded_data):
        """The  computation processes takes users input and process it"""
        pkd = p_d * (computation_requirement_user / (end_user * computation_requirement_user))
        execution_latency = (input_data * computation_requirement_user) / (end_user * pkd)
        print("Computation at Data center is done at %d%%" % (random.randint(50, 99)))
        output_data0 = self.input_data * (70/100)  # We assume that the output data is small than input data
        total_exucution_dc = transm_delay_EC_dc + execution_latency
        execution_latency_dc_vector.append(total_exucution_dc)
        computation_capacity_allocation_dc.append(pkd)
        output_data.append(output_data0)
        cached_content, DC_caching_decision_variable, cache_capacity_allocation_dc = \
            DataCenter.cache_dc(self, end_user, input_data, output_data0, c_d, Name_offloaded_data)
        DC_computation_decision_variable.append(1)

        return output_data, execution_latency_dc_vector, DC_computation_decision_variable, \
               computation_capacity_allocation_dc, cached_content, DC_caching_decision_variable, cache_capacity_allocation_dc


#######################################################################################################################
# For edge clouds


class EC(object):
    """
    Each EC has a limited number of servers to allocate computation and caching resources in parallel
       End-user has to request resources and once got it, it has to start the computation and wait for it to finish.
       The input and output of computation are cached for being reused later
    """

    # Offloading task and corresponding data to EC network

    def __init__(self, number_EC_server, end_user, input_data, computation_deadline,
                 computation_requirement_user, p_EC, c_EC, RC_EC_capacity):
        self.number_EC_server = number_EC_server
        self.p_EC = p_EC
        self.c_EC = c_EC
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user = computation_requirement_user
        self.end_user = end_user
        self.RC_EC_capacity = RC_EC_capacity

    def cache(self, end_user, input_data, output_data0, c_EC, Name_offloaded_data):
        """The  caching processes takes users input data and computation output, then cache them"""

        print("Caching at EC server is done at %d%% " % (random.randint(50, 99)))
        c_km = c_EC/(1+((end_user-1) * (input_data + output_data0)))
        cache_capacity_allocation_EC.append(c_km)
        cached_content.insert(Name_offloaded_data, output_data0)
        return cached_content, cache_capacity_allocation_EC

    def compute(self, end_user, input_data, computation_deadline, p_ec, c_ec, RC_EC_capacity,
                computation_requirement_user, transm_delay_user_ru, Name_offloaded_data):
        """The  computation processes takes users input and process it"""
        global execution_latency_dc_vector  # Global need to be declared in the beginning of the function
        global EC_m_cache_allocation_variable
        global cache_capacity_allocation_EC
        global DC_caching_decision_variable
        global DC_computation_decision_variable
        global output_data

        pkm = p_ec * (computation_requirement_user / ((end_user - 1) * computation_requirement_user))
        p_EC_n = max(computation_capacity_EC)
        pkn = p_EC_n * (computation_requirement_user / ((end_user - 1) * computation_requirement_user))
        execution_latency_ec_n = (input_data * computation_requirement_user) / ((end_user - 1) * p_ec)
        execution_latency_ec_m = (input_data * computation_requirement_user) / ((end_user - 1) * p_ec)
        if computation_requirement_user <= pkm and execution_latency_ec_m <= computation_deadline:
            execution_latency_mc = (input_data * computation_requirement_user) / ((end_user - 1) * p_ec)
            total_executing_ec_m = execution_latency_mc
            total_executing_ec_m_array.append(total_executing_ec_m)
            total_executing_ec_n_array.append(0)
            EC_m_EC_n_offloading_variable.append(0)
            EC_n_cache_allocation_variable.append(0)
            print("  Computation at EC is done at %d%%" % (random.randint(50, 99))) # We just print to see where the
            # task is execute
            EC_m_computation_allocation_variable.append(1)
            EC_n_computation_allocation_variable.append(0)
            output_data0 = self.input_data * (70 / 100)  # We assume that after computation, the input can be reduced
            # 30%
            EC_m_cache_allocation_variable.append(1)
            cached_content, cache_capacity_allocation_EC = \
            EC.cache(self, end_user, input_data, output_data0, c_ec, Name_offloaded_data)
            computation_capacity_allocation_EC.append(pkm)
            execution_latency_dc_vector.append(0)
            cache_capacity_allocation_dc = []
            computation_capacity_allocation_dc = []
            DC_computation_decision_variable.append(0)
            computation_capacity_allocation_dc.append(0)
            DC_caching_decision_variable.append(0)
            cache_capacity_allocation_dc.append(0)
            output_data.append(output_data0)
            transm_delay_BS_m= input_data / Fiber_Fronthaul_RU_EC[end_user]
            transm_delay_between_ru.append(transm_delay_BS_m)
            return transm_delay_between_ru, cached_content, EC_m_cloud_offloading_variable, \
                       EC_m_computation_allocation_variable, EC_n_computation_allocation_variable, \
                       total_executing_ec_m_array, total_executing_ec_n_array, \
                       computation_capacity_allocation_EC, cache_capacity_allocation_EC, \
                       EC_m_cache_allocation_variable, EC_n_cache_allocation_variable, \
                       EC_m_EC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                       DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                       cache_capacity_allocation_dc
        elif computation_requirement_user <= pkn and execution_latency_ec_n <= computation_deadline:
            EC_n_cache_allocation_variable.append(1)
            EC_m_EC_n_offloading_variable.append(1)
            EC_m_cache_allocation_variable.append(0)
            EC_n_computation_allocation_variable.append(1)
            EC_m_computation_allocation_variable.append(0)
            # Offloading delay between EC m and EC n
            transm_delay_ECm_ECn = input_data / Bandwidth_EC_m_EC_n
            total_executing_ec_n = execution_latency_ec_n
            print("  Computation at EC  n is done at %d%%" % (random.randint(50, 99)))
            output_data0 = input_data * (40 / 1000)
            cached_content, cache_capacity_allocation_EC = EC.cache(
                self, end_user, input_data, output_data0, c_ec, Name_offloaded_data)
            computation_capacity_allocation_EC.append(0)
            execution_latency_dc_vector.append(0)
            cache_capacity_allocation_dc = []
            output_data.append(output_data0)
            computation_capacity_allocation_dc = []
            DC_computation_decision_variable = []
            DC_computation_decision_variable.append(0)
            computation_capacity_allocation_dc.append(0)
            DC_caching_decision_variable.append(0)
            cache_capacity_allocation_dc.append(0)
            EC_m_cloud_offloading_variable.append(0)
            total_executing_ec_n_array.append(total_executing_ec_n)
            total_executing_ec_m_array.append(0)
            transm_delay_between_ru.append(transm_delay_ECm_ECn)
            return transm_delay_between_ru, cached_content, EC_m_cloud_offloading_variable, \
                       EC_m_computation_allocation_variable, EC_n_computation_allocation_variable, \
                       total_executing_ec_m_array, total_executing_ec_n_array, \
                       computation_capacity_allocation_EC, cache_capacity_allocation_EC, \
                       EC_m_cache_allocation_variable, EC_n_cache_allocation_variable, \
                       EC_m_EC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                       DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                       cache_capacity_allocation_dc
        else:
            print("No available resources at EC server, task is offloaded to Data center")
            EC_n_computation_allocation_variable.append(0)
            EC_m_computation_allocation_variable.append(0)
            EC_m_cloud_offloading_variable.append(1)
            EC_m_cache_allocation_variable.append(0)
            cache_capacity_allocation_EC.append(0)
            EC_n_cache_allocation_variable.append(0)
            EC_m_EC_n_offloading_variable.append(0)
            transm_delay_EC_dc = input_data / RC_EC_capacity
            # Request Resource at Data center
            datacenter = DataCenter(total_number_regional_cloud, computation_deadline, input_data, computation_deadline,
                                    computation_requirement_user, p_d, c_d)
            output_data, execution_latency_dc_vector, DC_computation_decision_variable, \
            computation_capacity_allocation_dc, cached_content, DC_caching_decision_variable, \
            cache_capacity_allocation_dc = datacenter.compute_dc(end_user, input_data, computation_requirement_user,
                                                                     transm_delay_EC_dc, Name_offloaded_data)

            computation_capacity_allocation_EC.append(0)
            total_executing_ec_m_array.append(0)
            total_executing_ec_n_array.append(0)
            transm_delay_bsm_dc = input_data / RC_EC_capacity
            transm_delay_between_ru.append(transm_delay_bsm_dc)

            return transm_delay_between_ru, cached_content, EC_m_cloud_offloading_variable, \
                       EC_m_computation_allocation_variable, EC_n_computation_allocation_variable, \
                       total_executing_ec_m_array, total_executing_ec_n_array, \
                       computation_capacity_allocation_EC, cache_capacity_allocation_EC, \
                       EC_m_cache_allocation_variable, EC_n_cache_allocation_variable, \
                       EC_m_EC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                       DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                       cache_capacity_allocation_dc

#######################################################################################################################


class EndUser(object):
    def __init__(self, end_user_id, input_data, computation_deadline, computation_requirement_user, p_k, c_k):
        self.p_k = p_k
        self.c_k = c_k
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user = computation_requirement_user
        self.end_user = end_user_id

    def cache_end_user(self, end_user, input_data, output_data0, c_k, computation_requirement_user,
                       Name_offloaded_data):
        """The  caching processes takes users input data and computation output, then cache them"""
        print("computation at end-user device is done at %d%% of %s's task." %
              (random.randint(50, 99), end_user))
        print("Caching at end-user device is done at %d%% of %s's task." % (random.randint(50, 99), end_user))
        c_ki = c_k * (input_data + output_data0) / (number_user_application[end_user] * computation_requirement_user)
        cached_content.insert(Name_offloaded_data, output_data0)
        cache_capacity_allocation_user.append(c_ki)
        return cached_content, cache_capacity_allocation_user

    def compute_end_user(self, end_user, input_data, computation_deadline, p_k, computation_requirement_user, c_k, p_ec,
                         c_ec, RC_EC_capacity, Name_offloaded_data):
        """Each end-user demand arrives at edge cloud and requests resources, where each end-user has identification (end_user_id).
          It  starts  computation process, waits for it to finish """
        global total_executing_ec_m_array
        global total_executing_ec_n_array
        global EC_m_cloud_offloading_variable
        global EC_m_computation_allocation_variable
        global EC_n_computation_allocation_variable
        global output_data
        global transm_delay_between_ru
        global cache_capacity_allocation_user

        # Convert
        # cycle_per_second = cycle_per_byte *  byte_per_second
        # https: // crypto.stackexchange.com / questions / 8405 / how - to - calculate - cycles - per - byte
        # each end-user device has a CPU peak bandwidth of $16$-bit values per cycle
        computation_requirement_user * 16
        pki = p_k * (computation_requirement_user / (end_user * computation_requirement_user))

        execution_latency_user = (input_data * computation_requirement_user) / (1 + pki)
        energy_consumption = cpu_arc_parameter * input_data * computation_requirement_user * pki ** 2
        active_user_list.append(end_user)
        if execution_latency_user >= computation_deadline or energy_consumption >= cpu_energy[
            end_user] or computation_requirement_user >= pki:
            print("No available resources at end-user device, task is offloaded to EC server")
            user_cache_allocation_variable.append(0)
            user_offloading_variable.append(1)
            # Radio resource revenue
            percentage_radio_spectrum = random.random()
            spectrum_efficiency_user = spectrum_efficiency  # End user needs communication resource for offloading
            instantaneous_data = np.multiply(1, (percentage_radio_spectrum *
                                                 spectrum_efficiency_user * wireless_bandwidth))
            transm_delay_user_ru = np.multiply(user_offloading_variable, (input_data / 1 + instantaneous_data))
            instantaneous_data_vector.append(instantaneous_data)
            input_data0 = []
            input_data0.append(input_data)
            percentage_radio_spectrum_vector.append(percentage_radio_spectrum)
            EC_m_cloud_offloading_variable.append(0)

            # Offload to EC network

            ec = EC(count_EdgeCloud, end_user, input_data, computation_deadline, computation_requirement_user, p_ec,
                    c_ec, RC_EC_capacity)

            transm_delay_between_ru, cached_content, EC_m_cloud_offloading_variable, \
            EC_m_computation_allocation_variable, EC_n_computation_allocation_variable, \
            total_executing_ec_m_array, total_executing_ec_n_array, \
            computation_capacity_allocation_EC, cache_capacity_allocation_EC, \
            EC_m_cache_allocation_variable, EC_n_cache_allocation_variable, \
            EC_m_EC_n_offloading_variable, output_data, execution_latency_dc_vector, \
            DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
            cache_capacity_allocation_dc = ec.compute(end_user, input_data, computation_deadline, p_ec, c_ec,
                                                      RC_EC_capacity,
                                                      computation_requirement_user, transm_delay_user_ru,
                                                      Name_offloaded_data)
            transm_delay.append(transm_delay_user_ru)
            computation_capacity_allocation_user.append(pki)
            local_computation_cost.append(0)
            user_execution_latency_array.append(0)
            input_data_vector.append(Name_offloaded_data)
            cache_capacity_allocation_user.append(0)
            return transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
                   user_execution_latency_array, computation_capacity_allocation_user, cache_capacity_allocation_user, \
                   local_computation_cost, user_cache_allocation_variable, user_offloading_variable, \
                   input_data_vector, active_user_list, transm_delay_between_ru, cached_content, \
                   EC_m_cloud_offloading_variable, EC_m_computation_allocation_variable, \
                   EC_n_computation_allocation_variable, total_executing_ec_m_array, total_executing_ec_n_array, \
                   computation_capacity_allocation_EC, cache_capacity_allocation_EC, EC_m_cache_allocation_variable, \
                   EC_n_cache_allocation_variable, EC_m_EC_n_offloading_variable, output_data, \
                   execution_latency_dc_vector, DC_computation_decision_variable, computation_capacity_allocation_dc, \
                   DC_caching_decision_variable, cache_capacity_allocation_dc
        else:
            end_user_device_stutus.append(1)
            user_execution_latency_array.append(execution_latency_user)
            user_cache_allocation_variable.append(1)
            total_executing_ec_m_array.append(0)
            total_executing_ec_n_array.append(0)
            EC_m_cloud_offloading_variable.append(0)
            EC_m_computation_allocation_variable.append(0)
            EC_n_computation_allocation_variable.append(0)
            percentage_radio_spectrum_vector.append(0)
            computation_capacity_allocation_EC = []
            computation_capacity_allocation_EC.append(0)
            user_offloading_variable.append(0)  # Local computation
            execution_latency_dc_vector = []
            execution_latency_dc_vector.append(0)
            instantaneous_data_vector.append(0)
            computation_capacity_allocation_user.append(pki)
            local_computation_cost.append(execution_latency_user)
            cache_capacity_allocation_EC = []
            cache_capacity_allocation_EC.append(0)
            EC_m_EC_n_offloading_variable = []
            EC_m_EC_n_offloading_variable.append(0)
            output_data0 = self.input_data * (70 / 100)
            output_data.append(output_data0)
            transm_delay_between_ru.append(0)
            cached_content, cache_capacity_allocation_user = \
                EndUser.cache_end_user(self, end_user, input_data, output_data0, c_k, computation_requirement_user,
                                       Name_offloaded_data)
            computation_capacity_allocation_EC.append(0)
            EC_m_cache_allocation_variable = []
            EC_n_cache_allocation_variable = []
            DC_computation_decision_variable = []
            computation_capacity_allocation_dc = []
            DC_caching_decision_variable = []
            cache_capacity_allocation_dc = []
            EC_m_cache_allocation_variable.append(0)
            EC_n_cache_allocation_variable.append(0)
            DC_computation_decision_variable.append(0)
            computation_capacity_allocation_dc.append(0)
            DC_caching_decision_variable.append(0)
            cache_capacity_allocation_dc.append(0)
            input_data_vector.append(Name_offloaded_data)
            transm_delay.append(0)

            return transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
                   user_execution_latency_array, computation_capacity_allocation_user, cache_capacity_allocation_user, \
                   local_computation_cost, user_cache_allocation_variable, user_offloading_variable, \
                   input_data_vector, active_user_list, transm_delay_between_ru, cached_content, \
                   EC_m_cloud_offloading_variable, EC_m_computation_allocation_variable, \
                   EC_n_computation_allocation_variable, total_executing_ec_m_array, total_executing_ec_n_array, \
                   computation_capacity_allocation_EC, cache_capacity_allocation_EC, EC_m_cache_allocation_variable, \
                   EC_n_cache_allocation_variable, EC_m_EC_n_offloading_variable, output_data, \
                   execution_latency_dc_vector, DC_computation_decision_variable, computation_capacity_allocation_dc, \
                   DC_caching_decision_variable, cache_capacity_allocation_dc
########################################################################################################################
# Start the simulation


for i in range(2, estimated_users_per_ru):
    secure_random = random.SystemRandom()
    input_data = int(User_RU[i])
    computation_requirement_user = computation_user_requirement[i]  # computation requirement for end-users in terms of
    RC_EC_capacity = random.choice(communication_EC_RC)
    p_k = computation_capacity_user[i]
    c_k = cache_capacity_user[i]
    Name_offloaded_data = ContentName[i]
    p_ec = random.choice(cache_capacity_EC)
    c_ec = random.choice(computation_capacity_EC)
    EC_server_maximum_capacity_computation.append(p_ec)
    EC_server_maximum_capacity_caching.append(c_ec)
    user_deadline = computation_user_deadline[i]  # computation deadline in terms of seconds
    users = EndUser(i, input_data, user_deadline, computation_requirement_user, p_k, c_k)

    transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
    user_execution_latency_array, computation_capacity_allocation_user, cache_capacity_allocation_user, \
    local_computation_cost, user_cache_allocation_variable, user_offloading_variable, \
    input_data_vector, active_user_list, transm_delay_between_ru, cached_content, \
    EC_m_cloud_offloading_variable, EC_m_computation_allocation_variable, \
    EC_n_computation_allocation_variable, total_executing_ec_m_array, total_executing_ec_n_array, \
    computation_capacity_allocation_EC, cache_capacity_allocation_EC, EC_m_cache_allocation_variable, \
    EC_n_cache_allocation_variable, EC_m_EC_n_offloading_variable, output_data, \
    execution_latency_dc_vector, DC_computation_decision_variable, computation_capacity_allocation_dc, \
    DC_caching_decision_variable, cache_capacity_allocation_dc = users.compute_end_user(i, input_data, user_deadline,
                                                                                        p_k,
                                                                                        computation_requirement_user,
                                                                                        c_k,
                                                                                        p_ec, c_ec, RC_EC_capacity,
                                                                                        Name_offloaded_data)

########################################################################################################################


transm_delay = np.array(transm_delay)
percentage_radio_spectrum_vector = np.array(percentage_radio_spectrum_vector)
percentage_radio_spectrum_vector /= percentage_radio_spectrum_vector.sum()
instantaneous_data_vector = np.array(instantaneous_data_vector)
user_execution_latency_array = np.array(user_execution_latency_array)
computation_capacity_allocation_user = np.array(computation_capacity_allocation_user)
cache_capacity_allocation_user = np.array(cache_capacity_allocation_user)
local_computation_cost = np.array(local_computation_cost)
user_cache_allocation_variable = np.array(user_cache_allocation_variable)
user_offloading_variable = np.array(user_offloading_variable)
input_data_vector = np.array(input_data_vector)
active_user_list = np.array(active_user_list)
transm_delay_between_ru = np.array(transm_delay_between_ru)
computation_capacity_allocation_EC = np.array(computation_capacity_allocation_EC)
computation_capacity_allocation_EC.resize(active_user_list.shape)
cache_capacity_allocation_EC = np.array(cache_capacity_allocation_EC)
cache_capacity_allocation_EC.resize(active_user_list.shape)
EC_m_cache_allocation_variable = np.array(EC_m_cache_allocation_variable)
EC_m_cache_allocation_variable.resize(active_user_list.shape)
EC_n_cache_allocation_variable = np.array(EC_n_cache_allocation_variable)
EC_n_cache_allocation_variable.resize(active_user_list.shape)
EC_m_EC_n_offloading_variable = np.array(EC_m_EC_n_offloading_variable)
EC_m_EC_n_offloading_variable.resize(active_user_list.shape)
execution_latency_dc_vector = np.array(execution_latency_dc_vector)
execution_latency_dc_vector.resize(active_user_list.shape)
DC_computation_decision_variable = np.array(DC_computation_decision_variable)
DC_computation_decision_variable.resize(active_user_list.shape)
computation_capacity_allocation_dc = np.array(computation_capacity_allocation_dc)
computation_capacity_allocation_dc.resize(active_user_list.shape)
DC_caching_decision_variable = np.array(DC_caching_decision_variable)
DC_caching_decision_variable.resize(active_user_list.shape)
cache_capacity_allocation_dc = np.array(cache_capacity_allocation_dc)
cache_capacity_allocation_dc.resize(active_user_list.shape)
########################################################################################################################

#  local Computation cost
array_one = np.ones(len(user_offloading_variable))
error_checking_variable = np.ones(len(user_offloading_variable))
number_demand = len(user_offloading_variable)

computation_cost10 = np.subtract(array_one, user_offloading_variable)
local_computation_cost_update = [a*b for a, b in zip(computation_cost10, user_execution_latency_array)]
local_computation_cost = [a*b for a, b in zip(local_computation_cost_update, local_computation_cost)]


# Offloading and computation latency
offloading_computation_cost20 = [a*b for a, b in zip(EC_m_computation_allocation_variable, total_executing_ec_m_array)]
offloading_computation_cost21 = [a*b for a, b in zip(EC_n_computation_allocation_variable, total_executing_ec_n_array)]
offloading_computation_cost22 = [a*b for a, b in zip(EC_m_cloud_offloading_variable, execution_latency_dc_vector)]
offloading_computation_cost23 = [a+b for a, b in zip(offloading_computation_cost20, offloading_computation_cost21)]
offloading_computation = [a + b for a, b in zip(offloading_computation_cost22, offloading_computation_cost23)]

Offloading_computation_latency = [a + b for a, b in zip(execution_latency_dc_vector, transm_delay_between_ru)]
Offloading_computation_latency = [a + b for a, b in zip(Offloading_computation_latency, transm_delay_between_ru)]
from scipy.interpolate import make_interp_spline, BSpline
index1 = len(local_computation_cost)
x_list = list(range(0, index1))
x_list = np.array(x_list)
xnew = np.linspace(x_list.min(), x_list.max(), index1)
spl1 = make_interp_spline(x_list, local_computation_cost, k=7)
local_computation_cost_smooth = spl1(xnew)


spl2 = make_interp_spline(x_list, offloading_computation_cost23, k=7)
offloading_computation_cost23_smooth = spl2(xnew)

spl3 = make_interp_spline(x_list, Offloading_computation_latency, k=7)
Offloading_computation_latency_smooth = spl3(xnew)

# Create smooth line chart
fig000, ax000 = plt.subplots(figsize=(9, 6))
local_computation, = plt.plot(xnew, sorted(local_computation_cost_smooth), linewidth=3, color='r', linestyle='dashed')
edge_cloud_computation, = plt.plot(xnew, sorted(offloading_computation_cost23_smooth),linewidth=3,color='g', linestyle='solid')
regional_computation,= plt.plot(xnew, sorted(Offloading_computation_latency_smooth),linewidth=3, color='b', linestyle='dotted')
plt.legend([local_computation, edge_cloud_computation, regional_computation], ['Local Computation', 'EC computation', 'DC computation'], fancybox=True)
plt.xlabel('Number of users')
plt.ylabel('Total delay (Second)')
plt.xticks()
plt.yticks()
plt.grid(color='gray', linestyle='dashed')
plt.ylim(bottom=0.)
plt.show()

spl11 = make_interp_spline(x_list, computation_capacity_allocation_user, k=7)
computation_capacity_allocation_user_smooth = spl11(xnew)


spl22 = make_interp_spline(x_list, computation_capacity_allocation_EC, k=7)
computation_capacity_allocation_EC_smooth = spl22(xnew)

spl33 = make_interp_spline(x_list, computation_capacity_allocation_dc, k=7)
computation_capacity_allocation_dc_smooth = spl33(xnew)

fig06, ax06 = plt.subplots(figsize=(9, 6))
local_computation, = plt.plot(sorted(computation_capacity_allocation_user_smooth), color='r', linewidth=3,linestyle='dashed')# plotting by columns
edge_cloud_computation, = plt.plot(sorted(computation_capacity_allocation_EC_smooth), color='g', linewidth=3,linestyle='solid') # plotting by columns
regional_computation, = plt.plot(sorted(computation_capacity_allocation_dc_smooth), color='b',linewidth=3, linestyle='dotted')# plotting by columns
plt.grid(color='gray', linestyle='dashed')
plt.legend([local_computation, edge_cloud_computation, regional_computation], ['Local computation allocation',
                                                                               'EC computation allocation',
                                                                               'RC computation allocation'],
           fancybox= True,loc='upper right')
plt.xlabel('Number of users')
plt.ylabel('Computation resource allocation')
plt.xticks()
plt.yticks()
plt.ylim(bottom=0.)
plt.show()

# print("local_computation_cost", local_computation_cost)
# print("offloading_computation", offloading_computation)
# computation delay cost
computation_delay_cost = [a+b for a, b in zip(local_computation_cost, offloading_computation)]
computation = computation_capacity_allocation_EC
total_delay = np.array(computation_delay_cost)

total_delay = abs(total_delay)
objective_function = np.array(total_delay)
# Get initial values
epsilon = 1e-12  # Epsilon Convergence condition
varrho = 1e-06
lambd = 1e-02
HFL_threshold = 10
k = 1

#  Disciplined Multi-Convex Programming
# Papers: https://web.stanford.edu/~boyd/papers/dmcp.html
# Python code: https://github.com/cvxgrp/dmcp
from cvxpy import * # Please do not change the this code because if you put on top it start conflicting with other package
########################################################################################################################

opt_val_varrho = []
# Sub_problem 1
def initialize_parameter1():
    t = 0  # iteration
    # Initial optimal variable at 0-th iteration  to zeros
    opt_x, opt_g,opt_varrho = np.zeros([number_demand, ]), np.zeros([number_demand, ]), np.zeros([number_demand, ])

    # Array of optimal variable and approximation error
    opt_val_x = []
    opt_val_g = []
    approx_err = []

    # Objective function
    opt_val_x.append(user_offloading_variable + (varrho / 2) *
                     np.power(np.linalg.norm(user_offloading_variable - opt_x), 2))
    opt_val_g.append(user_offloading_variable + (varrho / 2) *
                     np.power(np.linalg.norm(error_checking_variable - opt_g), 2))
    opt_val_varrho.append(np.dot(total_delay, user_offloading_variable) + (varrho / 2) *
                          np.power(np.linalg.norm(EC_m_cache_allocation_variable - opt_varrho), 2))

    # Initial approximate error
    approx_err.append(np.Inf)
    return t, opt_x, opt_g, opt_val_x, opt_val_g, opt_varrho, opt_val_varrho, approx_err


def obj_dmcp(i, opt_x, opt_g, opt_val_varrho):
    x = cvx.Variable(number_demand, nonneg=True)
    g = cvx.Variable(number_demand, nonneg=True)
    # initial point
    x.value = np.random.uniform(low=0.1, high=1.0, size=(number_demand,))
    g.value = np.random.uniform(low=0.1, high=1.0, size=(number_demand,))

    x_xo_norm = 0
    g_go_norm = 0
    for t in range(number_demand):
        if t == i:
            x_xo_norm += np.linalg.norm(user_offloading_variable[i] - opt_x[t]) ** 2
            g_go_norm += np.linalg.norm(error_checking_variable[i] - opt_g[t]) ** 2
    objective1 = cvx.Minimize(abs(cvx.sum(x * (Total_learning_Time * g)) - k * math.log(1/theta) -
                              ((1 / (lambd * 2)) * x_xo_norm) - ((1 / (lambd * 2)) * g_go_norm)))
    constraint1 = [cvx.sum(percentage_radio_spectrum_vector[t]) * x <= 1,
               x*Total_learning_Time <= HFL_threshold, x <= g, x >= 0, g >= 0, x <= 1, g <= 1]
    myprob1 = cvx.Problem(objective1, constraint1)
    print("minimal sets:", dmcp.find_minimal_sets(myprob1))   # find all minimal sets
    print("problem is DCP:", myprob1.is_dcp())
    print("problem is DMCP:", dmcp.is_dmcp(myprob1))
    result = myprob1.solve(method='bcd')
    opt_x = x.value
    opt_g = x.value
    opt_val_varrho.append(myprob1.value)
    return opt_x, opt_g,  opt_val_varrho


########################################################################################################################

# BCD sing Cyclic coordinate selection rule
t, opt_x, opt_g, opt_val_x, opt_val_g, opt_varrho, opt_val_varrho, approx_err = initialize_parameter1()


for i in range(0,  number_demand):
    opt_x, opt_g,  opt_val_cy = obj_dmcp(i, opt_x, opt_g, opt_val_varrho)
    opt_val_cyc = opt_val_cy
    t += 1
opt_val_cyc = opt_val_cy
########################################################################################################################

# BCD using Randomized coordinate selection rule
t, opt_x, opt_g, opt_val_x, opt_val_g, opt_varrho, opt_val_varrho, approx_err = initialize_parameter1()

for i in range(0, number_demand):
    i = np.random.randint(0, number_demand)
    opt_x, opt_g, opt_val_rand = obj_dmcp(i, opt_x, opt_g, opt_val_varrho)
    opt_val_ran = opt_val_rand
    t += 1
opt_val_ran = opt_val_rand
########################################################################################################################


# BCD using Gauss-Southwell coordinate selection rule

t, opt_x, opt_g, opt_val_x, opt_val_g, opt_varrho, opt_val_varrho, approx_err = initialize_parameter1()

for i in range(0, number_demand):
    i = np.argmax(np.abs(total_delay + varrho / 2 * (user_offloading_variable - np.array(opt_x))))
    opt_x, opt_g, opt_val_gou = obj_dmcp(i, opt_x, opt_g, opt_val_varrho)
    opt_val_gso = opt_val_gou
    t += 1
opt_val_gso = opt_val_gou

print("opt_val_cyc", opt_val_cyc)
print("opt_val_gso", opt_val_gso)
print("opt_val_ran", opt_val_ran)
fig222, ax222 = plt.subplots(figsize=(9, 6))
cyc, = plt.plot(opt_val_cyc)
gso, = plt.plot(opt_val_gso)
ran, = plt.plot(opt_val_ran)
plt.xlabel('Iterations')
plt.ylabel('Total delay minimization(Second)')
plt.legend([cyc, gso, ran], ['Our Proposal with Cyclic', 'Our Proposal with Gauss-Southwell', 'Our Proposal with Randomized'])
plt.show()

########################################################################################################################

# Sub_problem 2

"""
delay_requirement = 10
def initialize_parameter2():
    # Initial optimal variable at 0-th iteration  to zeros
    opt_x, opt_y = np.zeros([number_demand, ]), np.zeros([number_demand, ])
    # Array of optimal variable and approximation error
    opt_val_x = []
    opt_val_y = []
    approx_err = []

    # Objective function
    opt_val_x.append(user_offloading_variable + (varrho / 2) *
                     np.power(np.linalg.norm(user_offloading_variable - opt_x), 2))
    opt_val_y.append(EC_m_computation_allocation_variable + (varrho / 2) *
                     np.power(np.linalg.norm( EC_m_computation_allocation_variable - opt_y), 2))
    # Initial approximate error
    approx_err.append(np.Inf)
    return opt_x, opt_y, opt_val_x, opt_val_y, approx_err


x = cvx.Variable(number_demand, nonneg=True)
y = cvx.Variable(number_demand, nonneg=True)

x.value = np.random.uniform(low=0.1, high=1.0, size=(number_demand,))
y.value = np.random.uniform(low=0.1, high=1.0, size=(number_demand,))

x_xo_norm = 0
y_yo_norm = 0

opt_x, opt_y, opt_val_x, opt_val_y, approx_err = initialize_parameter2()
for t in range(number_demand):
    x_xo_norm += np.linalg.norm(user_offloading_variable[t] - opt_x[t]) ** 2
    y_yo_norm += np.linalg.norm(EC_m_computation_allocation_variable[t] - opt_y[t]) ** 2

objective2 = cvx.Minimize(abs(cvx.sum((1-x) * local_computation_cost) +
                              cvx.sum((x * y) * offloading_computation) - k * math.log(1/theta) +
                              ((1 / (lambd * 2)) * x_xo_norm) - ((1 / (lambd * 2)) * y_yo_norm)))
constraint2 = [cvx.sum(x * y) * computation <= np.sum(computation_capacity_EC),
               (1-x) + x * (y*total_delay) <= delay_requirement, (1-x) + (x * y) == 1,
               y <= x, x >= 0, y >= 0, x <= 1, y <= 1]
myprob2 = cvx.Problem(objective2, constraint2)
print("minimal sets:", dmcp.find_minimal_sets(myprob2))   # find all minimal sets
print("problem is DCP:", myprob2.is_dcp()) 
print("problem is DMCP:", dmcp.is_dmcp(myprob2)) 
result = myprob2.solve(method='bcd')
"""
