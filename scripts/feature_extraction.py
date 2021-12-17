# author: Jaime Bowen Varela
# Gets the features of the raw data sets 
import pandas as pd
import numpy as np
from pathlib import Path 

def get_absmean(values):
    # returns the absolute mean
    return np.sum(np.abs(values)) / len(values)

def get_shapefactor(values):
    # returns the shape factor
    # More info at https://en.wikipedia.org/wiki/Shape_parameter
    shapefactor = get_rms(values) / get_absmean(values)
    return shapefactor

def get_rms(values):
    # return the Root Mean Square
    # More info at https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(np.sum(values ** 2) / len(values))

features_data_path = Path("../data/feature_data")

if not features_data_path.exists():
    features_data_path.mkdir()

data_set_paths = [Path("../data/1st_test_full.pkl"), 
Path("../data/2nd_test_full.pkl") ,Path("../data/3th_test_full.pkl")]

features_to_extract = ["mean_abs","rms","sfactor","kurt","skew","rolling_avg"]
for index,df_path in enumerate(data_set_paths):
    for feature in features_to_extract:
        print(f"Feature:{feature} \t Dataset:{str(df_path.name)}\n")
        feature_path = features_data_path/feature
        if not feature_path.exists():
            feature_path.mkdir()
        df = pd.read_pickle(str(df_path))
        if feature == "rms":
            df1 = df.groupby(["timestamp"]).apply(get_rms)
        elif feature == "mean_abs":
            df = df.abs()
            df1 = df.groupby(["timestamp"]).mean()
        elif feature == "kurt":
            df1 = df.groupby(["timestamp"]).apply(pd.DataFrame.kurt)
        elif feature == "skew":
            df1 = df.groupby(["timestamp"]).apply(pd.DataFrame.skew)
        elif feature == "sfactor":
            df1 = df.groupby(["timestamp"]).apply(get_shapefactor)
        elif feature == "rolling_avg":
            break
        else:
            print("Fin blucle")
            break
        try:
            df1.drop(columns=["timestamp"], inplace=True)
        except: 
            pass
        df1.index = pd.to_datetime(df1.index, unit="s")
        df1.to_pickle(feature_path/f"{feature}_{index+1}.pkl")