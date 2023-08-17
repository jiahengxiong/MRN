import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# Load the Training Set

def load_data(path):
    basic_train = pd.read_csv(path)

    dataset = basic_train.drop(['User_Satisfaction', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    ground_truth = basic_train.loc[:, 'User_Satisfaction'].copy()

    print('Train Data:', dataset.shape)
    print('Train Target:', ground_truth.shape)
    print('% of Class 1 Users (Dissatisfied) in Training : ', len(dataset[ground_truth == 1]) / len(dataset),
          '\n% of Class 0 Users (Satisfied) in Training: ', len(dataset[ground_truth == 0]) / len(dataset))



    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    x = dataset.Cumulative_Full_Service_Time_LTE
    tmp_0 = ax.hist(x, bins=100)
    ax.set_ylabel('Bincount')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    plt.grid(True)
    plt.xlabel('[s]')
    plt.show()


    total_service_time_UMTS = dataset['Cumulative_Full_Service_Time_UMTS'] + dataset['Cumulative_No_Service_Time_UMTS'] + \
                              dataset['Cumulative_Lim_Service_Time_UMTS']

    total_service_time_LTE = dataset['Cumulative_Full_Service_Time_LTE'] + dataset['Cumulative_No_Service_Time_LTE'] + \
                              dataset['Cumulative_Lim_Service_Time_LTE']

    total_youtube_time = dataset['Cumulative_YoutubeSess_LTE_DL_Time'] + dataset['Cumulative_YoutubeSess_UMTS_DL_Time']

    dataset['Cumulative_Full_Service_Time_percent_UMTS'] = dataset['Cumulative_Full_Service_Time_UMTS'] / total_service_time_UMTS
    dataset['Cumulative_Lim_Service_Time_percent_UMTS'] = dataset['Cumulative_Lim_Service_Time_UMTS'] / total_service_time_UMTS
    dataset['Cumulative_No_Service_Time_percent_UMTS'] = dataset['Cumulative_No_Service_Time_UMTS'] / total_service_time_UMTS
    dataset['Cumulative_Full_Service_Time_percent_LTE'] = dataset['Cumulative_Full_Service_Time_LTE'] / total_service_time_LTE
    dataset['Cumulative_No_Service_Time_percent_LTE'] = dataset['Cumulative_No_Service_Time_LTE'] / total_service_time_LTE
    dataset['Cumulative_Lim_Service_Time_percent_LTE'] = dataset['Cumulative_Lim_Service_Time_LTE'] / total_service_time_LTE
    dataset['Cumulative_YoutubeSess_LTE_DL_Time_percent'] = dataset['Cumulative_YoutubeSess_LTE_DL_Time'] / total_youtube_time
    dataset['Cumulative_YoutubeSess_UMTS_DL_Time_percent'] = dataset['Cumulative_YoutubeSess_UMTS_DL_Time'] / total_youtube_time

    return dataset, ground_truth


if __name__ == '__main__':
    path = 'D:\\project\\MRN\\dataset\\BasicDataset_Training_MRN.csv'  # PUT YOUR FILE PATH
    load_data(path)