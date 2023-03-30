import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volume import ChaikinMoneyFlowIndicator

def preprocessing(df: pd.DataFrame, lag:int = 1, sequence_length:int = 128, dif_all:bool = True, 
                  train_size:int=0.9, TSAI:bool = False, CLF:bool = False, index:str = None, 
                  data:str = "alpacca", buckets:int=1, print_info:bool=False, TI = False) -> tuple:
    
    #add technical indicators
    if TI: df = add_TI(df.copy()).dropna().reset_index(drop=True)

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
        print(df.dropna(how='all').dropna(how='all', axis=1))

    # Calculates the change in close price to use for classification

    if CLF: change = (df['close'] - df['close'].shift(lag)).dropna().reset_index(drop=True)
    
    # Get price lag steps into the future
    else: change = df['close'][lag:].reset_index(drop=True)
    
    # Calculates the difference in columns if set to true
    df = df.diff().dropna() if dif_all else df[1:]
    
    #Assign week and day after, since they are features rather than timeseries, and hence shouldnt' be differenced
    df = df.assign(change = change, weekday = weekday, hour = hour )

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

        # Used for df_info
        bucket_stats = df.groupby('positive_bucket')['change'].agg(['min', 'max'])
        
        #Labels need to be 0-->2n-1
        df['change'] = df['positive_bucket']
        df = df.drop(['abs_column','negative_bucket','positive_bucket'], axis=1)

    # Split the dataset into test and train data (only for visualization)
    df_train, df_test = df[:int(len(df)*train_size)], df[int(len(df)*train_size)+lag:]

    # Scales the data using robustscaler
    scaler = RobustScaler().fit(df_train)
    df_scaled = pd.DataFrame(scaler.transform(df), index = df.index, columns = df.columns)

    if CLF:
        df_scaled['change'] = df['change']
        

        # Create a new dataframe for info
        df_info = pd.DataFrame({'Label': df['change'].value_counts().sort_index(ascending=True).index, 
                                'Count': df['change'].value_counts().sort_index(ascending=True).values, 
                                'Train count': df_train["change"].value_counts().sort_index(ascending=True).values,
                                'Test count': df_test["change"].value_counts().sort_index(ascending=True).values,  
                                'Bucket min': bucket_stats['min'].values, 
                                'Bucket max': bucket_stats['max'].values})

        if print_info:
            df_info.loc[len(df_info)] = ['Total', df_info['Count'].sum(), 
                                        df_info['Train count'].sum(), 
                                        df_info['Test count'].sum(), 
                                        df_info['Bucket min'].min(), 
                                        df_info['Bucket max'].max()]
            
            print(df_info)

    if TSAI:
        df_sequence = create_sequences_2(df_scaled, 'change', sequence_length)
    
    else:
        df_sequence = create_sequences(df_scaled, 'change', sequence_length)

    # Split the dataset into test and train data --> 
    X_train_sequence, X_test_sequence, Y_train_sequence, Y_test_sequence = train_test_split(df_sequence[0], df_sequence[1], test_size=1-train_size, random_state=42, shuffle = False)

    #train_sequence, test_sequence = df_sequence[[:int(len(df[0])*train_size)][:int(len(df[0])*train_size)]

    return (X_train_sequence, Y_train_sequence), (X_test_sequence, Y_test_sequence), scaler

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

def add_TI(df):
        #Moving Average Convergence Divergence --> Trend
        indicator_MACD = MACD(close=df["close"], window_slow= 26, window_fast= 12, window_sign= 9)
        df['MACD_line'] = indicator_MACD.macd()
        df['MACD_diff'] = indicator_MACD.macd_diff()
        df['MACD_signal'] = indicator_MACD.macd_signal()

        #Relative strength index --> Momentum
        df['RSI'] = RSIIndicator(close=df["close"],window=14).rsi()

        #Bollinger bands --> Volatility
        indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()
        df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
        df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

        #Volume --> Chaikin Money Flow
        #df['CMF'] = ChaikinMoneyFlowIndicator(high=df["high"], low = df["low"], close = df["close"], volume=df["volume"], window=20).chaikin_money_flow()

        return df
