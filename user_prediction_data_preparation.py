import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Source of data https://www.kaggle.com/liukunxin/dataset/activity
df_sample = pd.read_csv('dataset/sample.csv')
# We ignore the sample data because it does not have meaningful data
# Because it does not have useful information for our problem
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
print(df_train.shape, df_test.shape, df_sample.shape)


# Only consider 5G networks and we ignore 3G and 4G network
df_train = df_train[(df_train["is_5g_base_cover"] == 1)
                                    & (df_train["is_work_5g_cover"] == 1)
                                    & (df_train["is_home_5g_cover"] == 1) &
                                    (df_train["is_work_5g_cover_l01"] == 1)
                                    & (df_train["is_home_5g_cover_l01"] == 1)
                                    & (df_train["is_work_5g_cover_l02"] == 1) &
                                    (df_train["is_home_5g_cover_l02"] == 1)]

# Select area_id, user_id, innet_months, total_times, video_app_flux
df_train = df_train[['area_id', 'user_id', 'innet_months', 'total_times', 'video_app_flux']]
df_train.to_csv('dataset/train_updated.csv')

df_test = df_test[(df_test["is_5g_base_cover"] == 1)
                                    & (df_test["is_work_5g_cover"] == 1)
                                    & (df_test["is_home_5g_cover"] == 1) &
                                    (df_test["is_work_5g_cover_l01"] == 1)
                                    & (df_test["is_home_5g_cover_l01"] == 1)
                                    & (df_test["is_work_5g_cover_l02"] == 1) &
                                    (df_test["is_home_5g_cover_l02"] == 1)]

# Select area_id, user_id, innet_months, total_times, video_app_flux
df_test = df_test[['area_id', 'user_id', 'innet_months', 'total_times', 'video_app_flux']]
df_test.to_csv('dataset/test_updated.csv')
# We combine two dataset
df = pd.concat([df_train, df_test])
df = df.to_csv('dataset/dataset_updated.csv')
