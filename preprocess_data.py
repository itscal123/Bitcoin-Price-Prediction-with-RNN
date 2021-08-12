from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd


df = pd.read_csv("data/data.csv")

df = pd.DataFrame(MinMaxScaler().fit_transform(df))
print(pd.head())