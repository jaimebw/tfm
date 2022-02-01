import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


start_time = time.time()
def MD_threshold(dist, extreme=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold


def MahalanobisDist(inv_cov_matrix, mean_distr, data):
    
    vars_mean = mean_distr
    diff = data - vars_mean
    md = np.array([])
    for i in range(len(diff)):
        md = np.append(md,np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))
    return md


def MD_detectOutliers(dist, extreme=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = np.array([])
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers = np.append(outliers,i)  # index of the outlier
    return outliers


def spree_plot(pca_object, fig_title=None, fig_name=None, **kwargs):
    # returns a spree plot from a PCA object
    per_var = np.round(pca_object.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]
    fig, ax = plt.subplots()
    ax.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_xlabel("Principal Component")
    if fig_title:
        ax.set_title("Scree Plot")
    else:
        ax.set_title(fig_title)
    if fig_name:
        fig.savefig(fig_name,dpi = 300)



RANDOM_STATE = 42
N_COMP = 2
input_data_path = Path("../data/feature_data")
output_data_path = Path("../data/pca_results/")
output_plots_path = output_data_path/"plots"
if not output_data_path.exists():
    output_data_path.mkdir()
if not output_plots_path.exists():
    output_plots_path.mkdir()
scaler = preprocessing.MinMaxScaler()

Results_df = pd.DataFrame()

features_paths = []
for i in input_data_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*.pkl"):
            features_paths.append(j)
#features_paths = features_paths[0:2]
for index,feature_data in enumerate(features_paths):
    # TODO : spre_plot plotting and change functions
    # TODO: build base lib (maybe) 
    print("------------------------")
    print(f"Data set:{feature_data}\t{index+1}/{len(features_paths)}\n")
    df = pd.read_pickle(feature_data)
    data_name = feature_data.stem
    if data_name[-1] == "1":
        X_train = df["2003-11-20":"2003-11-21"]
        X_test = df["2003-11-21":]
       
    elif data_name[-1] == "2":
        X_train = df["2004-02-14":"2004-02-16"]
        X_test = df["2004-02-16":]
    else:
        # muestra reduciada para encontrar los fallos
        X_train = df["2004-04-14":"2004-04-15"]
        X_test = df["2004-04-15":]

    # eliminar aquellos datos que sobran
    print(f"Len_feature :{df.shape[0]} ")
    #X_train, X_test = train_test_split(df,test_size = 0.3,random_state=RANDOM_STATE)
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                       columns = X_train.columns,
                       index = X_train.index)
    X_train.sample(frac=1)
    X_test = pd.DataFrame(scaler.transform(X_test),
                      columns = X_test.columns,
                      index = X_test.index)
    pca = PCA(random_state = RANDOM_STATE,svd_solver ='full',n_components = N_COMP) 
    labels = [f"PCA{i}" for i in range(1,N_COMP+1)]
    X_train_PCA = pca.fit_transform(X_train)
    X_test_PCA = pca.transform(X_test)
    
    X_test_PCA = pd.DataFrame(X_test_PCA,index = X_test.index,columns = labels)
    X_train_PCA = pd.DataFrame(X_train_PCA,index =X_train.index,columns = labels )
    # NOTA: cojemos la covarianza de solo una parte del modelo para que esta sea la que más 
    cov_matrix = np.cov(X_train_PCA.values,rowvar=False) #cov_matrix = pca.get_covariance()[0:N_COMP,0:N_COMP] # voy a usar dos componentes
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    dist_mean = X_train_PCA.mean(axis = 0).values
    dist_test = MahalanobisDist(inv_cov_matrix, dist_mean, X_test_PCA.values)
    dist_train = MahalanobisDist(inv_cov_matrix, dist_mean, X_train_PCA.values)

    threshold = MD_threshold(dist_train, extreme = False)

    print(f"Cov_matrix: \n {cov_matrix}\n")
    print(f"Threshold: {threshold} \n ")

    anomaly_train = pd.DataFrame()
    anomaly_train['dist_mob']= dist_train
    anomaly_train['thresh'] = threshold
    anomaly_train['anomaly'] = anomaly_train['dist_mob'] > anomaly_train['thresh']
    anomaly_train.index = X_train_PCA.index

    anomaly = pd.DataFrame()
    anomaly['dist_mob']= dist_test
    anomaly['thresh'] = threshold
    anomaly['anomaly'] = anomaly['dist_mob'] > anomaly['thresh']
    anomaly.index = X_test_PCA.index
    
    anomaly_alldata = pd.concat([anomaly_train, anomaly])
    print(f" Len_anomaly: {anomaly_alldata.shape[0]}")
    anomaly_alldata.sort_index(inplace=True)
    anomaly_alldata.to_pickle(str(output_data_path/data_name)+"_pca_labels.pkl")
    results_df = pd.DataFrame({"name":data_name,"data_set_n":data_name[-1],"thresh":threshold},index = [index])
    results_df["name"] = data_name
    results_df["thresh"] = threshold
    try:
        results_df["first_detection"] = anomaly_alldata.loc[anomaly_alldata.anomaly == True].index[0]
    except:
        results_df["first_detection"] = np.nan 
    Results_df  = Results_df.append(results_df)
Results_df.to_csv(output_data_path/f"results_pca.csv")
print(f"Tiempo total: {time.time()-start_time}")
