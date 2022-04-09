import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Load the features paths
feauture_path = Path("../data/feature_data")
features_paths = []
for i in input_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*"):
            features_paths.append(j)
data1 = []
data2 = []
data3 = []
for data in features_paths:
    if "1" in re.findall(r"\d+",str(data)):
        data1.append(data)
    elif "2" in re.findall(r"\d+",str(data)):
        data2.append(data)
    else:
        data3.append(data)
data1.sort()
data2.sort()
data3.sort()

labels_paths = {"pca":Path("../data/pca_results/results_pca.csv"),
        "lin_auto":Path("../data/lineal_autoencoder_results/lineal_autoencoder_results.csv"),
        "conv_auto":Path("../data/conv_autoencoder_results/conv1d_autoencoder_results.csv")]

for name, data_path in labels_paths.keys():
    df = pd.read_csv(data_path)
    if not name == "pca":
        df.index = df.Label
        df.rename(columns= {"detection_day":"first_detection"},inplace = True)
    else:
        df.index = df.name
    
    
