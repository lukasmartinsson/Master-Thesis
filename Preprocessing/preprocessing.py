import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch

def Preprocessing(df: pd.DataFrame) -> pd.DataFrame:    
    
    # Drop irrelevant values
    #df = df.drop(columns=['symbol', 'timestamp'])

    # Check for NaN values
    if df.isnull().values.any():
        print("Error in dataframe, missing value")

    # Add the difference between values (will be used as label)
    df['results'] = [df.iloc[i]['vwap'] - df.iloc[i-1]['vwap'] for i in range(len(df))]
    
    # Resets the index and removes the first value that will have the wrong results
    df = df[1:].reset_index()

    features = df[['open', 'high', 'low', 'close', 'volume', 'trade_count','vwap']]
    labels = df['results']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle = False)
    

    # Apply RobustScaler to the training set
    scaler = RobustScaler().fit(X_train)

    # Apply the scaling to the sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled data into tensors
    X_train = torch.tensor(X_train_scaled).float()
    X_train = torch.reshape(X_train,   (X_train.shape[0], 1, X_train.shape[1]))
    
    X_test = torch.tensor(X_test_scaled).float()
    X_test = torch.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1])) 

    y_train = torch.tensor(y_train.values).float()
    y_train = y_train.view(-1, 1)

    y_test = torch.tensor(y_test.values).float()
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