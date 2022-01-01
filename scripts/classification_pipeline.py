"""
TO-DO:
    1. Get all features data sets
    2. Add labels to the data sets
    3. Obtain results of the training ( of all three algos)
    4. Obtain the plots and confussion matrix
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix,
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

kneigh = KNeighborsClassifier()
dtree = DecisionTreeClassifier()
forest  = RandomForestClassifier()
algos = [kneigh,dtree,forest]
RANDOM_STATE = 42
scaler = MinMaxScaler()
input_labels = Path("../data/labels")
input_path = Path("../data/feature_data")
output_path = Path("../data/classification_results")
if not output_path.exists():
    output_path.mkdir()

# 1. Loading the features

features_paths = []
for i in input_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*.pkl"):
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
labels = [pd.read_pickle(i) for i in input_labels.glob("*.pkl")]
# 2. Add labels to the dataset and train
for index, dir_paths in enumerate([data1,data2,data3]):
    #index +=1
    training_results = pd.DataFrame()
    for data_paths in dir_paths:
        data_name = data_paths.stem
        df = pd.read_pickle(data_paths)
        df = df.join(labels[index])
        df.dropna(inplace = True) # delete nans in values
        X = df[["b1_ch1","b2_ch2","b3_ch3","b4_ch4"]].values 
        y = df[["labels"]].values
        y = np.ravel(y)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=RANDOM_STATE)
        # 3. Train algos
        for algo in algos:
            algo.fit(X_train,y_train)
            # add scoring methods



