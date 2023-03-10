import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils
import numpy as np

def preprocessing(df: pd.DataFrame, lag:int = 1, sequence_length:int = 128, dif_all:bool = True, 
                  train_size:int=0.9, TSAI:bool = False, CLF:bool = False, index:str = None, data:str = "alpacca", buckets:int=1) -> tuple:

    #Check if the input data comes from alpacca or twelve, and if the spy index should be added as features
    if data == "alpacca":
        if index is not None: #min, hour, day depending on which data
            df_index = pd.read_csv('Data\index\SPY_'+index).add_suffix('_new').rename(columns={'timestamp_new':'timestamp'})
            df = pd.merge(df,df_index, on = 'timestamp', how='left').fillna(method='ffill').drop(['symbol_new'], axis=1)
        time_data = pd.to_datetime(df['timestamp'])
        df = df.drop(['symbol', 'timestamp'], axis=1) # Drop symbol and timestamp from df

    elif data == "twelve":
        if index is not None: #min, hour, day depending on which data
            df_index = pd.read_csv('Data/twelve_data/SPY_'+index).add_suffix('_new').rename(columns={'datetime_new':'datetime'})
            df = pd.merge(df,df_index, on = 'datetime', how='left').fillna(method='ffill')
        time_data = pd.to_datetime(df['datetime'])
        df = df.drop(['datetime'], axis=1)

    # Extract week and day
    
    weekday = time_data.dt.weekday.values[1:]
    hour = time_data.dt.hour.values[1:]

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

    #Need to be before split to ensure that buckets have the same range
    if CLF: 

        # Create a new column with absolute values
        df['abs_column'] = df['change'].abs()

        # Divide the positive values into n buckets with custom labels
        df.loc[df['change'] >= 0, 'positive_bucket'] = pd.qcut(df.loc[df['change'] >= 0, 'abs_column'], q=buckets, labels=False)+1

        # Divide the negative values into n buckets with custom labels
        df.loc[df['change'] < 0, 'negative_bucket'] = pd.qcut(df.loc[df['change'] < 0, 'abs_column'], q=buckets, labels=False)

        # Fill 
        df['positive_bucket'] = df['positive_bucket'].fillna(-df['negative_bucket'])+buckets-1
        print(df.groupby('positive_bucket')['change'].agg(['min', 'max']))
        
        #Labels need to be 0-->2n-1
        df['change'] = df['positive_bucket']
        df = df.drop(['abs_column','negative_bucket','positive_bucket'], axis=1)

        # print the resulting dataframe
        print('Split full dataset')
        print(df['change'].value_counts())

    # Split the dataset into test and train data --> 
    df_train, df_test = df[:int(len(df)*train_size)], df[int(len(df)*train_size)+lag:]

    # Scales the data using robustscaler
    scaler = RobustScaler().fit(df_train)
    df_train_scaled = pd.DataFrame(scaler.transform(df_train), index = df_train.index, columns = df_train.columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), index = df_test.index, columns = df_test.columns)

    if CLF:
        df_train_scaled['change'] = df_train['change']
        df_test_scaled['change'] = df_test['change']
        print(df_train["change"].value_counts())
        print(df_test["change"].value_counts())

    if TSAI:
        train_sequence = create_sequences_2(df_train_scaled, 'change', sequence_length)
        test_sequence = create_sequences_2(df_test_scaled, 'change', sequence_length)
    
    else:
        train_sequence = create_sequences(df_train_scaled, 'change', sequence_length)
        test_sequence = create_sequences(df_test_scaled, 'change', sequence_length)

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
