# Script for normalizing the data form data set 1 from 8 columns to just 4
import pandas as pd
import numpy as np
import re
from pathlib import Path
def sum_col(val1,val2):
    return np.sqrt(val1**2+val2**2)
features_data_path = Path("../data/feature_data")
print(features_data_path.exists())
features_paths = []
for i in features_data_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*.pkl"):
            features_paths.append(j)

data1 = []
for data in features_paths:
    if "1" in re.findall(r"\d+",str(data)):
        data1.append(data)
    else:
        pass
data1.sort()

for i in data1:
    df = pd.read_pickle(i)
    print(i.stem)
    df["b1_ch1"] = sum_col(df.b1_ch1,df.b1_ch2)
    df["b2_ch2"] = sum_col(df.b2_ch3,df.b2_ch4)
    df["b3_ch3"] = sum_col(df.b3_ch5,df.b3_ch6)
    df["b4_ch4"] = sum_col(df.b4_ch7,df.b4_ch8)
    df.drop(columns = ["b1_ch2"	,"b2_ch3",	"b2_ch4","b3_ch5","b3_ch6","b4_ch7","b4_ch8"],inplace = True)
    fname = str(i)
    fname = fname[:-4]
    fname = fname +"_n.pkl"
    df.to_pickle(fname)