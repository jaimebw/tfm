"""
TO-DO:
    1. Get all features data sets [X]
    2. Add labels to the data sets[X]
    3. Obtain results of the training ( of all three algos)[X]
    4. Obtain the plots and confussion matrix[]
"""
from pathlib import Path
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




RANDOM_STATE = 42
kneigh = KNeighborsClassifier()
dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
forest  = RandomForestClassifier(random_state=RANDOM_STATE)
algos = {"kneigh":kneigh,"dtree":dtree,"forest":forest}

scaler = MinMaxScaler()
input_labels = Path("../data/labels")
input_path = Path("../data/feature_data")
output_path = Path("../data/classification_results")
output_rocplots_path = output_path/"roc_plots"

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
labels_paths = [i for i in input_labels.glob("*.pkl")]
labels_paths.sort()
labels = [pd.read_pickle(i) for i in labels_paths]
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
        X = df[["b1_ch1","b2_ch2","b3_ch3","b4_ch4"]].values 
        print(len(X))
        y = df[["labels"]].values
        y = np.ravel(y)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
        # 3. Train algos
        for name,algo in algos.items():
            algo.fit(X_train,y_train)
            y_pred = algo.predict(X_test)
            confusion_matrix, precision_score,f1_score,recall_score,accuracy_score 
            accuracy = accuracy_score(y_test,y_pred)
            precission = precision_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            roc_score = roc_auc_score(y_test,y_pred)
            plot_name = f"/roc_{data_name}_{name}.pdf"
            plot_roc_curve(y_test,y_pred,str(output_rocplots_path)+plot_name) 
            print(f"Algo: {name}\tDataset:{data_name}\tAccuracy:{accuracy}\tPrecission{precission}\tRecall:{recall}\tf1:{f1}\n")
            algo_df = pd.DataFrame({
                "algo":name,
                "dataset":data_name,
                "accuracy":accuracy,
                "precission":precission,
                "recall":recall,
                "f1":f1,
                "auc":roc_score

            },index = [outer_index])
            #algo_df["Algo"] = name
            #algo_df["dataset"] = data_name
            #algo_df["accuracy"] = accuracy
            training_results = training_results.append(algo_df)
            outer_index +=1 
            print("-----------------------------------------------------\n")
        training_results.to_csv(str(output_path)+f"/classification_results_set{index+1}.csv")



"""
Curva AUC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()

plt.show()
"""