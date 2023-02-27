import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils

def Preprocessing(df: pd.DataFrame, lag:int = 1, batch_size:int = 32, dif_all:bool = True, num_workers:int = 4) -> tuple:    
    
    time_data = pd.to_datetime(df['timestamp'])
    weekday = time_data.dt.weekday.values[1:-lag]
    hour = time_data.dt.hour.values[1:-lag]

    df = df.drop(['symbol', 'timestamp'], axis=1)

    # Check for NaN values
    if df.isnull().values.any():
        print("Error in dataframe, missing value")


    # Add the difference between values (will be used as label)
    results = df.iloc[lag:]['open'].values - df.iloc[:-lag]['open'].values

    #Diff all values if TRUE
    df = pd.DataFrame((df.values[1:-lag] - df.values[:-(lag+1)] ) if dif_all else df[1:-lag])
   
   #can make nicer if needed
    df['results'] = results[1:]
    df['weekday'] = weekday
    df['hour'] = hour
    
    #Check diff before scaling to make sure that label isn't warped
    diff_one_zero = (sum(y > 0 for y in df['results'])-sum(y < 0 for y in df['results']))/len(df['results'])
    print('The diff of one and zero prior to scaling is is: '+'{:.2%}'.format(diff_one_zero))

    # Scale data using robust scaler
    df = RobustScaler().fit_transform(df.values)
    
    features, labels = df[:,:7], df[:,-1:]

    # Check diff post scaling
    diff_one_zero = (sum(y > 0 for y in labels)-sum(y < 0 for y in labels))/len(labels)
    print('The diff of one and zero post scaling is: '+'{:.2%}'.format(diff_one_zero[0]))

    # Convert features and labels to tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
 
    # Split the features and labels into a training set and a test set# Split the features and labels into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

    # Convert the training data into a PyTorch tensor and create a data loader for it
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)#, num_workers = num_workers)

    # Convert the test data into a PyTorch tensor and create a data loader for it
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#, num_workers = num_workers)

    return train_dataloader, test_dataloader
'''
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
    
    return X_train, X_test[lag+1:], y_train, y_test[lag+1:]
'''

def preprocessing_improved(df: pd.DataFrame, lag:int = 1, sequence_length:int = 128, dif_all:bool = True, train_size:int=0.9) -> tuple:

    #Extract week and day
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
    df = df.diff().dropna() if dif_all else df.dropna() #Might be more efficient to use iloc so we dont have to loop through everything?

    #Assign week and day after, since they are features rather than timeseries, and hence shouldnt' be differenced
    df = df.assign(change = change,
                   weekday = weekday,
                   hour = hour )

    #Shift function goes backwards, we wont have the "real" results for the last "lag" instances
    df = df[:-lag]


    # Split the dataset into test and train data --> 
    df_train, df_test = df[:int(len(df)*train_size)], df[int(len(df)*train_size)+lag:]

    print("Pre", df_test[sequence_length:]['change'].values)


    # Scales the data using robustscaler
    scaler = RobustScaler().fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), index = df_train.index, columns = df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), index = df_test.index, columns = df_test.columns)

    print("Post", df_test[sequence_length:]['change'].values)
    
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
