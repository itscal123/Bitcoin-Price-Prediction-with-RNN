import requests
import pandas as pd
import time


#    Note that the apikey parameter in the url string should be replaced with your own api key which can be obtained for free
#    at https://www.alphavantage.co/support/

def get_exchange_rates(apikey, symbol="BTC"):
    """
    Downloads daily historical time series for Bitcoin (BTC) traded in the USD market, refreshed daily at midnight (UTC).
    params: apikey (str), symbol (str)
    returns: dataframe
    """
    url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={}&market=USD&apikey={}'.format(symbol, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient="index").sort_index(axis=1)
    df = df.rename(columns={ '1a. open (USD)': 'Open (USD)', '2a. high (USD)': 'High (USD)', '3a. low (USD)': 'Low (USD)', '4a. close (USD)': 'Close (USD)', '5. volume': 'Volume', '6. market cap (USD)': 'Market Cap (USD)'})
    df = df[['Open (USD)', 'High (USD)', 'Low (USD)', 'Close (USD)', 'Volume', 'Market Cap (USD)']]
    df.to_csv("prices.csv")
    return df


def get_SMA(apikey, symbol="BTC", time_period=50):
    """
    Downloads the daily simple moving average (SMA) values for Bitcoin in USD. Since SMA is considered to react relatively
    slow in price changes, we use the time period of 50 days. Additionally, since SMA is usually calculated using closing prices
    we set the series type parameter to close.
    params: apikey (str), symbol (str), time_period (positive int)
    returns dataframe
    """
    url = 'https://www.alphavantage.co/query?function=SMA&symbol={}USD&interval=daily&time_period={}&series_type=close&apikey={}'.format(symbol, time_period, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data['Technical Analysis: SMA'], orient="index").sort_index(ascending=False)
    df.to_csv("sma.csv")
    return df


def get_EMA(apikey, symbol="BTC", time_period=20):
    """
    Downloads the daily exponential moving average (EMA) values for Bitcoin in USD. Since SMA is considered to be a shorter indicator, we use the time period of 20 days.
    Additionally, since SMA is usually calculated using closing prices we also use closing prices for the EMA series type parameter.
    params: apikey (str), symbol (str), time_period (positive int)
    returns dataframe
    """
    url = 'https://www.alphavantage.co/query?function=EMA&symbol={}USD&interval=daily&time_period={}&series_type=close&apikey={}'.format(symbol, time_period, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data['Technical Analysis: EMA'], orient="index").sort_index(ascending=False)
    df.to_csv("ema.csv")
    return df


def get_RSI(apikey, symbol="BTC", time_period=14):
    """
    Downloads the daily relative strength index (RSI) values for Bitcoin. Popular value for time period of the 
    indicator is 14 which we set as the default value
    params: apikey (str), symbol (str), time_period (positive int) 
    returns dataframe
    """
    url = 'https://www.alphavantage.co/query?function=RSI&symbol={}USD&interval=daily&time_period={}&series_type=close&apikey={}'.format(symbol, time_period, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Technical Analysis: RSI"], orient="index").sort_index(ascending=False)
    df.to_csv("rsi.csv")
    return df


def get_BBANDS(apikey, symbol="BTC", time_period=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    Downloads the daily Bollinger Bands values for Bitcoin. Here we use the standard Bollinger Band formula where we
    set the centerline as a 20 day simple moving average (SMA) and use a 2x multiplier for the upper and lower bands.
    Hence, time_period is 20, nbdevup and nbdevdn are both 2, and matype is 0 where 0 signifies SMA. Check alpha vantage
    documentation for more information.
    params: apikey (str), symbol (str), time_period (positive int), nbdevup(positive int)
            nbdevdn (postive int), matype (int [0,8])
    returns: df
    """
    url = 'https://www.alphavantage.co/query?function=BBANDS&symbol={}USD&interval=daily&time_period=20&series_type=close&nbdevup={}&nbdevdn={}&matype={}&apikey={}'.format(symbol, time_period, nbdevup, nbdevdn, matype, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Technical Analysis: BBANDS"], orient="index").sort_index(axis=1)
    df.to_csv("bbands.csv")
    return df


def get_MACD(apikey, symbol="BTC", fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Downloads the moving average convergence / divergence (MACD) values. The MACD represents a trend
    following indicator that highlights the short-term price momentum and whether it follows the direction
    of the long-term price momentum or if a trend is near. The indicator uses the difference between
    a slow period EMA and fast period EMA which is popularly set to 12 and 26, respectively. Likewise, 
    there is a signal line which is generally defined by a 9 period EMA. 
    params: apikey (str), symbol (str), fastperiod (positive int), slowperiod (positive int), signalperiod (positive int)
    returns: dataframe
    """
    url = 'https://www.alphavantage.co/query?function=MACD&symbol={}USD&interval=daily&series_type=close&fastperiod={}&slowperiod{}&signalperiod={}&apikey={}'.format(symbol, fastperiod, slowperiod, signalperiod, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Technical Analysis: MACD"], orient="index").sort_index(axis=1)
    df.to_csv("macd.csv")
    return df


def get_STOCH(apikey, symbol="BTC", fastkperiod=14, slowkperiod=3, slowdperiod=3, slowkmatype=0, slowdmatype=0):
    """
    Downloads the daily stochastic oscillator (STOCH) values. The indicator shows momentum by comparing the 
    closing price with a range of its prices over a certain period of time. Generally uses simple moving average
    hence the default values of slowkmatype and slowdmatype. Additional parameters are the fastkperiod, 
    slowkperiod, and slowdperiod which are commonly set to 14 for the fast parameter and 3 for the slow parameters.
    params: apikey (str), symbol (str), fastperiod (positive int), slowkperiod (positive int), slowdperiod (positive int),
            slowkmatype (int [0,8]) slowdmatype (int [0,8])
    returns: dataframe
    """
    url = 'https://www.alphavantage.co/query?function=STOCH&symbol={}USD&interval=daily&fastkperiod={}&slowkperiod={}&slowdperiod={}&slowkmatype={}&slowdmatype={}&apikey={}'.format(symbol, fastkperiod, slowkperiod, slowdperiod, slowkmatype, slowdmatype, apikey)
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Technical Analysis: STOCH"], orient="index").sort_index(axis=1)
    df.to_csv("stoch.csv")
    return df


def get_data(apikey):
    """
    Calls the get_ functions to retrieve the necessary data. Since we are using the free api which is limited
    to 5 calls/minute we need to implement a timer to split the api calls so that we don't go over the api call 
    limit. Then we merge the data into a single dataframe using outer union logic which we write to the current 
    directory as csv file
    params: apikey (str)
    returns: dataframe
    """
    exchange_rates = get_exchange_rates(apikey)
    sma = get_SMA(apikey)
    ema = get_EMA(apikey)
    rsi = get_RSI(apikey)
    bbands = get_BBANDS(apikey)
    time.sleep(60) # Wait a minute before using the API again
    macd = get_MACD(apikey)
    stoch = get_STOCH(apikey)

    datasets = [exchange_rates, sma, ema, rsi, bbands, macd, stoch]
    data = pd.concat(datasets, axis=1)
    data.to_csv("data.csv")
    return data


if __name__ == "__main__":
    apikey = "S8YIUGVLMYAG3S4E"
    data = get_data(apikey)
