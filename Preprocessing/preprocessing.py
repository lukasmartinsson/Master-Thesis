import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch

import matplotlib.pyplot as plt

def Preprocessing(df: pd.DataFrame, lag:int = 1, dif_all:bool = True) -> pd.DataFrame:    
    
    df = df.drop(['symbol', 'timestamp'], axis=1)

    # Check for NaN values
    if df.isnull().values.any():
        print("Error in dataframe, missing value")


    # Add the difference between values (will be used as label)
    res_df = [df.iloc[i+lag]['open'] - df.iloc[i]['open'] for i in range(1,len(df)-lag)]

    #Save colummn names for later
    col_names = list(df.columns) + ['results']

    #Diff all values if TRUE
    if dif_all:
        df = pd.DataFrame([df.values[i] - df.values[i-1] for i in range(1,len(df)-lag)])
    else: df = df[1:-lag]
    
    df['results'] = res_df

    ##Check diff before scaling to make sure that label isn't warped
    diff_one_zero = (sum(y > 0 for y in res_df)-sum(y < 0 for y in res_df))/len(res_df)
    print('The diff of one and zero prior to scaling is is: '+'{:.2%}'.format(diff_one_zero))

    # Scale data using robust scaler
    df = RobustScaler().fit_transform(df.values)
    df = pd.DataFrame(df,columns=col_names)
    
    # Extract features and labels
    features = df[['open', 'high', 'low', 'close', 'volume', 'trade_count','vwap']].values
    labels = df['results'].values

    # Check diff post scaling
    diff_one_zero = (sum(y > 0 for y in labels)-sum(y < 0 for y in labels))/len(labels)
    print('The diff of one and zero post scaling is: '+'{:.2%}'.format(diff_one_zero))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle = False)
    

    # Convert the scaled data into tensors
    X_train = torch.tensor(X_train).float()
    X_train = torch.reshape(X_train,   (X_train.shape[0], 1, X_train.shape[1]))
    
    X_test = torch.tensor(X_test).float()
    X_test = torch.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1])) 

    y_train = torch.tensor(y_train).float()
    y_train = y_train.view(-1, 1)

    y_test = torch.tensor(y_test).float()
    y_test = y_test.view(-1, 1)

    return X_train, X_test, y_train, y_test


'''def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset.values[i] - dataset.values[i - interval]
        diff.append(value)
    return diff

def scale(dataset):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(dataset)
    return pd.DataFrame(scaled)


def add_lag(df, lag=1):
    columns = [df[0].shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df = df.drop(0)
    df.columns=['open_t','open_t+1','high','low','close','volume_t', 'trade_count_t','vmap']
    print(df)
    return df

def split_data(df, test_size = 0.2):
    X = df.drop('open_t+1', axis=1).values
    Y = df['open_t+1'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return  X_train, X_test, y_train, y_test

def process_data(df):
    return split_data(add_lag(scale(difference(df))))'''