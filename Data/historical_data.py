from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import  (CryptoBarsRequest, StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
    StockLatestBarRequest)
from alpaca.data.timeframe import TimeFrame

def Historical_data(s_type: str, stock: list[str], timeframe: TimeFrame, start: str, end: str, client: StockHistoricalDataClient):

    if s_type == 'Crypto': return Crypto(stock, timeframe, start, end)
    elif s_type == 'StockBars': return StockBars(stock, timeframe, start, end, client)
    elif s_type == 'StockQuotes': return StockQuotes(stock, timeframe, start, end, client)
    elif s_type == 'StockTrades': return StockTrades(stock, timeframe, start, end, client)
    elif s_type == 'StockLatestTrade': return StockLatestTrade(stock, timeframe, start, end, client)
    elif s_type == 'StockLatestQuote': return StockLatestQuote(stock, timeframe, start, end, client)
    elif s_type == 'StockSnapshot': return StockSnapshot(stock, timeframe, start, end, client)
    elif s_type == 'StockLatestBar': return StockLatestBar(stock, timeframe, start, end, client)   

def Crypto(stock, timeframe, start, end):

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

def StockBars(stock, timeframe, start, end, client):

    # Creating request object
    request_params = StockBarsRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_bars = client.get_stock_bars(request_params)

    # Convert to dataframe and return
    return stock_bars.df

def StockQuotes(stock, timeframe, start, end, client):

    # Creating request object
    request_params = StockQuotesRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_quotes = client.get_stock_quotes(request_params)

    # Convert to dataframe and return
    return stock_quotes.df

def StockTrades(stock, timeframe, start, end, client):

    # Creating request object
    request_params = StockTradesRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_trades = client.get_stock_trades(request_params)

    # Convert to dataframe and return
    return stock_trades.df

def StockLatestTrade(stock, timeframe, start, end, client):

    # Creating request object
    request_params = StockLatestTradeRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_latest_trade = client.get_stock_latest_trade(request_params)

    # Convert to dataframe and return
    return stock_latest_trade.df

def StockLatestQuote(stock, timeframe, start, end, client):
    
    # Creating request object
    request_params = StockLatestQuoteRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_latest_quote = client.get_stock_latest_quote(request_params)

    # Convert to dataframe and return
    return stock_latest_quote.df

def StockSnapshot(stock, timeframe, start, end, client):
    
    # Creating request object
    request_params = StockSnapshotRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_quotes = client.get_stock_snapshot(request_params)

    # Convert to dataframe and return
    return stock_quotes.df

def StockLatestBar(stock, timeframe, start, end, client):
    
    # Creating request object
    request_params = StockLatestBarRequest(
                            symbol_or_symbols=stock,
                            timeframe=timeframe,
                            start=start,
                            end=end
                            )

    stock_latest_bar = client.get_stock_latest_bar(request_params)

    # Convert to dataframe and return
    return stock_latest_bar.df