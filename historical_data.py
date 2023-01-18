from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import  (CryptoBarsRequest, StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
    StockLatestBarRequest)
from alpaca.data.timeframe import TimeFrame

def Historical_data(type: str, stock: list[str], timeframe: TimeFrame, start: str, end: str):

    if type == 'Crypto':
        return Crypto(stock, timeframe, start, end)
   # elif type == 'StockBars':
   # elif type == 'StockQuotes':
   # elif type == 'StockTrades':
   # elif type == 'StockLatestTrade':
   # elif type == 'StockLatestQuote':
   # elif type == 'StockSnapshot':
   # elif type == 'StockLatestBar':
    

def Crypto(stock, timeframe, start, end):
    
    # Create a client for cryptohistoricaldata
    client = CryptoHistoricalDataClient()

    # Creating request object
    request_params = CryptoBarsRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    # Retrieve daily bars for Bitcoin in a DataFrame and printing it
    btc_bars = client.get_crypto_bars(request_params)

    # Convert to dataframe and return
    return btc_bars.df