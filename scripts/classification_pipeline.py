"""
TO-DO:
    1. Get all features data sets [X]
    2. Add labels to the data sets[X]
    3. Obtain results of the training ( of all three algos)[X]
    4. Obtain the plots and confussion matrix[]
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import metrics
from sklearn.metrics import roc_auc_score,confusion_matrix, precision_score,f1_score,recall_score,accuracy_score,RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import font_manager
from sklearn.model_selection import KFold


def label_dist(label):
    one_n = np.count_nonzero(label==1)
    zero_n = np.count_nonzero(label==0)
    return print(f"Unos:{one_n}\t Ceros:{zero_n}")

kf = KFold(n_splits=5)
font_dirs = ['../otros/lmr']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Latin Modern Roman'
def plot_roc_curve(y_test,y_pred,fname):
    with plt.style.context("seaborn-paper"):
        fig,ax = plt.subplots(figsize = (5,5))
        roc = RocCurveDisplay.from_predictions(y_test,y_pred)
        roc.plot(ax = ax)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.savefig(fname,dpi = 300)

def gen_score(y_test,y_pred,feature_name,algo_name,panda_index = 0):
    accuracy = accuracy_score(y_test,y_pred)
    precission = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    try:
        roc_score = roc_auc_score(y_test,y_pred)
    except:
        roc_score =0
    results = pd.DataFrame({"feature_name":feature_name,"algo":algo_name,"accuracy":accuracy,"precission":precission,
        "f1":f1,"recall":recall,"roc_score":roc_score},index = [panda_index])
    return results



RANDOM_STATE = 42
kneigh = KNeighborsClassifier()
dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
forest  = RandomForestClassifier(random_state=RANDOM_STATE)
algos = {"kneigh":kneigh,"dtree":dtree,"forest":forest}

scaler = MinMaxScaler()
input_labels ={"pca":Path("../data/labels/labels_pca"),
        "lin_auto":Path("../data/labels/labels_lineal"),
        "conv_auto":Path("../data/labels/labels_conv")}
input_path = Path("../data/feature_data")
output_path = Path("../data/con_auto_classification_results")
output_rocplots_path = output_path/"pca_roc_plots"

if not output_path.exists():
    output_path.mkdir()
if not output_rocplots_path.exists():
    output_rocplots_path.mkdir()
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

labels_paths = [i for i in input_labels["conv_auto"].glob("*.pkl")]
labels_paths.sort()
labels = [pd.read_pickle(i) for i in labels_paths]

#labels_paths_lin= [i for i in input_labels["lin_auto"].glob("*.pkl")]
#labels_paths_conv = [i for i in input_labels["conv_auto"].glob("*.pkl")]
# 2. Add labels to the dataset and train
outer_index = 0
for index, dir_paths in enumerate([data1,data2,data3]):
    training_results = pd.DataFrame()
    for data_paths in dir_paths:
        data_name = data_paths.stem
        print(f"Data set: {data_name}\n")
        df = pd.read_pickle(data_paths)
        df = df.join(labels[index])

        df.dropna(inplace = True) 
        df = df.sample(int(len(df))) # it is need to shuffle the entire database to obtain good trainign resulst
        X = df[["b1_ch1","b2_ch2","b3_ch3","b4_ch4"]].values 
        print(len(X))
        y = df[["labels"]].values
        y = np.ravel(y)
        X = scaler.fit_transform(X)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
        # 3. Train algos
        for name,algo in algos.items():
            fold_index =0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print("Train")
                label_dist(y_train)
                print("Test")
                label_dist(y_test)
                algo.fit(X_train,y_train)
                y_pred = algo.predict(X_test)
                results = gen_score(y_test,y_pred,data_name,name,outer_index)
                #plot_name = f"/roc_{data_name}_{name}.pdf"
                #plot_roc_curve(y_test,y_pred,str(output_rocplots_path)+plot_name) 
                print(f"Algo: {name}\tDataset:{data_name}\t Kfold: {fold_index}")
                fold_index +=1
                training_results = training_results.append(results)
                outer_index +=1 
                print("-----------------------------------------------------\n")
        training_results.to_csv(str(output_path)+f"/conv_auto_classification_results_set{index+1}.csv")
