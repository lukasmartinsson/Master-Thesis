import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils
import numpy as np

def preprocessing(df: pd.DataFrame, lag:int = 1, sequence_length:int = 128, dif_all:bool = True, train_size:int=0.9, TSAI = False, CLF = False) -> tuple:

    # Extract week and day
    time_data = pd.to_datetime(df['timestamp'])
    weekday = time_data.dt.weekday.values[1:]
    hour = time_data.dt.hour.values[1:]

    # Drop symbol and timestamp from df
    df = df.drop(['symbol', 'timestamp'], axis=1)

    # Check for NaN values
    if df.isnull().values.any():
        print("Error in dataframe, missing value")

    # Calculates the change in close price
    change = (df['close'] - df['close'].shift(lag)).dropna().reset_index(drop=True)

    # Calculates the difference in columns if set to true
    df = df.diff().dropna() if dif_all else df[1:] #Might be more efficient to use iloc so we dont have to loop through everything?

    #Assign week and day after, since they are features rather than timeseries, and hence shouldnt' be differenced
    df = df.assign(change = change,
              weekday = weekday,
              hour = hour )

    #Shift function goes backwards, we wont have the "real" results for the last "lag" instances
    df = df[:-lag]

    
    # Split the dataset into test and train data --> 
    df_train, df_test = df[:int(len(df)*train_size)], df[int(len(df)*train_size)+lag:]

    # Scales the data using robustscaler
    scaler = RobustScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), index = df_train.index, columns = df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), index = df_test.index, columns = df_test.columns)

    if CLF: 
        df_train["change"] = np.where(df_train["change"] < 0, 0, 1)
        df_test["change"] = np.where(df_test["change"] < 0, 0, 1)

    if TSAI:
        train_sequence = create_sequences_2(df_train, 'change', sequence_length)
        test_sequence = create_sequences_2(df_test, 'change', sequence_length)
    
    else:
        train_sequence = create_sequences(df_train, 'change', sequence_length)
        test_sequence = create_sequences(df_test, 'change', sequence_length)

    return train_sequence, test_sequence, scaler

def create_sequences(df: pd.DataFrame, prediction:str, sequence_length:int):

    # Get the data as a PyTorch tensor
    data = torch.tensor(df.drop(['change'],axis=1).values, dtype=torch.float)

    # Create all the sequences at once using PyTorch's tensor slicing
    sequences = torch.stack([data[i:i+sequence_length] for i in range(len(data) - sequence_length + 1)]) #Changed to so that label and feature have the same position

    # Get the labels as a separate PyTorch tensor using pandas' .iloc method
    labels = torch.tensor(df.iloc[sequence_length-1:][prediction].values, dtype=torch.float) #Changed to so that label and feature have the same position

    return sequences, labels


def create_sequences_2(df: pd.DataFrame, prediction: str, sequence_length: int):
    # Get the data as a PyTorch tensor
    data = torch.tensor(df.drop([prediction], axis=1).values, dtype=torch.float)

    # Create all the sequences at once using PyTorch's tensor slicing
    sequences = torch.zeros((len(data) - sequence_length + 1, data.shape[1], sequence_length), dtype=torch.float)
    
    for i in range(len(data) - sequence_length + 1): sequences[i] = data[i:i+sequence_length].T

    # Get the labels as a separate PyTorch tensor using pandas' .iloc method
    labels = torch.tensor(df.iloc[sequence_length-1:][prediction].values, dtype=torch.float)

    return sequences, labels
