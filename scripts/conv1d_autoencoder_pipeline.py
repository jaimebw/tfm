import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn import preprocessing
import time
from torch import nn
from tqdm import tqdm

feature_names = {"rms":"Root Mean Square",
        "mean_abs":"Absolute Mean",
        "rolling_avg_10": "Moving Average 10 periods",
        "rolling_avg_20": "Moving Average 20 periods",
        "rolling_avg_30": "Moving Average 30 periods",
        "rolling_avg_50": "Moving Average 50 periods",
        "kurt": "Kurtosis",
        "skew": "Skewness"
        }
fail_dates = ("25/11/2003","19/02/2004","18/04/2004")
def get_tensor(X:np.ndarray):
    Xt = torch.empty(len(X),4,1)
    for index,data_row in enumerate(X):
        Xt[index] = torch.from_numpy(data_row).unsqueeze(0).transpose(1,0)
    return Xt


def get_numpy(X):
    x = X.detach().numpy()
    x = x.reshape(x.shape[0],4)
    return x

class ConvAutoEncoder(nn.Module):
    def __init__(self,n_components):
        self.n_components = n_components
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4,12,stride =3,kernel_size = 2,padding = 1),
            nn.ReLU(),
            nn.Conv1d(12,24,stride = 3, kernel_size = 2,padding = 1),
            nn.ReLU(),
            nn.Conv1d(24,self.n_components,stride = 3, kernel_size = 2,padding = 1))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.n_components,24,stride = 2, kernel_size = 1),
            nn.ReLU(),
            nn.ConvTranspose1d(24,12,stride = 2, kernel_size = 1),
            nn.ReLU(),
            nn.ConvTranspose1d(12,4,stride =2,kernel_size = 1))
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

EPOCHS = 5000
start_time = time.time()
df_columns = ["b1","b2","b3","b4"]
input_data_path = Path("../data/feature_data")
output_data_path = Path("../data/conv_autoencoder_results/")
output_plots_path = output_data_path/"plots"
if not output_data_path.exists():
    output_data_path.mkdir()
if not output_plots_path.exists():
    output_plots_path.mkdir()
scaler = preprocessing.MinMaxScaler()

features_paths = []
for i in input_data_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*.pkl"):
            features_paths.append(j)


combined_results = pd.DataFrame()
for index,feature_data in enumerate(features_paths):
    print("------------------------")
    print(f"Data set:{feature_data}\t{index+1}/{len(features_paths)}\n")
    df = pd.read_pickle(feature_data)
    data_name = feature_data.stem
    for key, value in feature_names.items():
        if key in data_name:
            data_label = value
    if data_name[-1] == "1":
        dataset_number = 1
        fail = fail_dates[0]
        X_train,train_index = df["2003-11-20":"2003-11-21"],df["2003-11-20":"2003-11-21"].index
        X_test,test_index = df["2003-11-21":],df["2003-11-21":].index
    elif data_name[-1] == "2":
        X_train,train_index = df["2004-02-14":"2004-02-16"],df["2004-02-14":"2004-02-16"].index
        dataset_number = 2
        X_test,test_index = df["2004-02-16":],df["2004-02-16":].index
    
        fail = fail_dates[1]
    else:
        # muestra reduciada para encontrar los fallos
        X_train,train_index = df["2004-04-14":"2004-04-15"],df["2004-04-14":"2004-04-15"].index
        dataset_number = 3
        X_test,test_index = df["2004-04-15":], df["2004-04-15":].index
        fail = fail_dates[2]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train + np.random.normal(X_train.mean(),X_train.std(),size = X_train.shape)
    X_train = get_tensor(X_train)
    X_test = get_tensor(X_test)
    model = ConvAutoEncoder(2)
    loss_crit = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr= 1e-3,weight_decay=1e-5)
    epochs = EPOCHS
    for epoch in tqdm(range(epochs)):
        recon = model(X_train)
        loss = loss_crit(recon,X_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_recon = model(X_test)
        loss_test = loss_crit(test_recon,X_test)
    print(f"Train loss:{loss.item()}\t Test_loss:{loss_test.item()}")
    test_recon = get_numpy(test_recon)
    recon = get_numpy(recon)
    df_train = pd.DataFrame(recon,train_index,columns = df_columns)
    df_test = pd.DataFrame(test_recon,test_index,columns = df_columns)
    test_mae = np.mean(np.abs(test_recon-get_numpy(X_test)),axis = 1)
    df_test["mae"] = test_mae
    anomaly_day = df_test.loc[df_test.mae>=0.3].index[0]
    #results = pd.DataFrame({"Feature":data_name,"feature_number":dataset_number,"detection_day":anomaly_day,
    #    "train_loss":loss.item(),"test_loss":loss_test.item()},index = [index])
    results = pd.DataFrame({"Label":data_label,"Feature":data_name,"feature_number":dataset_number,"detection_day":anomaly_day,
        "train_loss":loss.item(),"test_loss":loss_test.item(),"Failure date":fail},index = [index])
    combined_results = combined_results.append(results)
    df_test.to_pickle(output_data_path/f"conv1d_autoencoder_{data_name}.pkl")
combined_results.to_csv(output_data_path/"conv1d_autoencoder_results.csv")
end_time = time.time()-start_time
print(end_time)
