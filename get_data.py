import requests
import pandas as pd
import os


#    Note that the apikey parameter in the url string should be replaced with your own api key which can be obtained for free
#    at https://www.alphavantage.co/support/

def get_exchange_rates():
    """
    Downloads daily historical time series for Bitcoin (BTC) traded in the USD market, refreshed daily at midnight (UTC).
    params: None
    returns: dataframe
    """
    url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=S8YIUGVLMYAG3S4E' # Replace apikey with your own key from https://www.alphavantage.co/support/
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient="index").sort_index(axis=1)
    df = df.rename(columns={ '1a. open (USD)': 'Open (USD)', '2a. high (USD)': 'High (USD)', '3a. low (USD)': 'Low (USD)', '4a. close (USD)': 'Close (USD)', '5. volume': 'Volume', '6. market cap (USD)': 'Market Cap (USD)'})
    df = df[['Open (USD)', 'High (USD)', 'Low (USD)', 'Close (USD)', 'Volume', 'Market Cap (USD)']]
    return df


def get_SMA():
    """
    Downloads the daily simple moving average (SMA) values for Bitcoin in USD. Since SMA is considered to react relatively
    slow in price changes, we use the time period of 50 days. Additionally, since SMA is usually calculated using closing prices
    we set the series type parameter to close.
    params: None
    returns dataframe
    """
    url = 'https://www.alphavantage.co/query?function=SMA&symbol=BTCUSD&interval=daily&time_period=50&series_type=close&apikey=S8YIUGVLMYAG3S4E'
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data['Technical Analysis: SMA'], orient="index").sort_index(ascending=False)
    return df


def get_EMA():
    """
    Downloads the daily exponential moving average (EMA) values for Bitcoin in USD. Since SMA is considered to be a shorter indicator, we use the time period of 20 days.
    Additionally, since SMA is usually calculated using closing prices we also use closing prices for the EMA series type parameter.
    params: None
    returns dataframe
    """
    url = 'https://www.alphavantage.co/query?function=EMA&symbol=BTCUSD&interval=daily&time_period=20&series_type=close&apikey=S8YIUGVLMYAG3S4E'
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame.from_dict(data['Technical Analysis: EMA'], orient="index").sort_index(ascending=False)
    return df


def merge_data(exchange_rates, sma, ema):
    """
    Merges the the different dataframes that were collected by the API. Concatenated using outer union logic. Writes the dataframe in the 
    current directory as a csv file.
    params: exchange_rates (dataframe), sma (dataframe), ema (dataframe)
    returns: dataframe
    """
    data = pd.concat([exchange_rates, sma, ema], axis=1)
    data.to_csv("data.csv")
    return data


if __name__ == "__main__":
    exchange_rates = get_exchange_rates()
    sma = get_SMA()
    ema = get_EMA()
    data = merge_data(exchange_rates, sma, ema)
    print(data.isnull().sum())