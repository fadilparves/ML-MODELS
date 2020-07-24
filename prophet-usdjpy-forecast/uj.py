import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from fbprophet import Prophet

def Predict(data_url, N):
    start = dt.datetime(2017,9,21)
    end = dt.datetime(2018,9,21)
    uj_df = web.DataReader('DEXJPUS', 'fred', start, end)
    # uj_df.head()
    print(uj_df.tail())
    uj = uj_df.reset_index()
    uj = uj.rename(columns={"DATE": "ds", "DEXJPUS": "y"})
    uj.set_index("ds").y.plot()
    m = Prophet()
    m.fit(uj)
    future = m.make_future_dataframe(periods=N)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()
    print(forecast)

Predict('uj.csv', 1)