import os
import datetime
import pandas as pd
import numpy as np


def str_datetime(file_name):
    # Returns the unix timestamp of the date of the file
    date = file_name.split(".")
    date = list(map(int, date))
    date = datetime.datetime(*date)
    return int(date.timestamp())

def load_data(path,n_channels= 8):
    # Loads the data from the dataset in a Dataframe
    cols = ["b1_ch1","b2_ch2","b3_ch3","b4_ch4"]
    if n_channels == 8:
        cols = ["b1_ch1","b1_ch2","b2_ch3","b2_ch4","b3_ch5","b3_ch6","b4_ch7","b4_ch8"]
    df = pd.read_csv(path,delimiter = "\t")
    df.columns = cols
    _,file_name = os.path.split(path)
    init_date = str_datetime(file_name)
    df = df.astype('float32')
    df["timestamp"] = np.arange(init_date,init_date + df.shape[0])
    df = df.set_index("timestamp")
    return df
