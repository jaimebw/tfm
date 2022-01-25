from os import wait
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn import preprocessing
import time
from torch import nn
from tqdm import tqdm
class LinealAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,2))
        self.decoder = nn.Sequential(
            nn.Linear(2,5),
            nn.ReLU(),
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,4))
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

EPOCHS = 5000
start_time = time.time()
df_columns = ["b1","b2","b3","b4"]
input_data_path = Path("../data/feature_data")
output_data_path = Path("../data/lineal_autoencoder_results/")
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
    if data_name[-1] == "1":
        dataset_number = 1
        X_train,train_index = df["2003-11-20":"2003-11-21"],df["2003-11-20":"2003-11-21"].index
        X_test,test_index = df["2003-11-21":],df["2003-11-21":].index
    elif data_name[-1] == "2":
        X_train,train_index = df["2004-02-14":"2004-02-16"],df["2004-02-14":"2004-02-16"].index
        dataset_number = 2
        X_test,test_index = df["2004-02-16":],df["2004-02-16":].index

    else:
        # muestra reduciada para encontrar los fallos
        X_train,train_index = df["2004-04-14":"2004-04-15"],df["2004-04-14":"2004-04-15"].index
        dataset_number = 3
        X_test,test_index = df["2004-04-15":], df["2004-04-15":].index
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train + np.random.normal(X_train.mean(),X_train.std(),size = X_train.shape)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    model = LinealAutoencoder()
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
    df_train = pd.DataFrame(recon.detach().numpy(),train_index,columns = df_columns)
    df_test = pd.DataFrame(test_recon.detach().numpy(),test_index,columns = df_columns)
    test_mae = np.mean(np.abs(test_recon.numpy()-X_test.numpy()),axis = 1)
    df_test["mae"] = test_mae
    anomaly_day = df_test.loc[df_test.mae>=0.3].index[0]
    results = pd.DataFrame({"Feature":data_name,"feature_number":dataset_number,
        "train_loss":loss.item(),"test_loss":loss_test.item()},index = [index])
    combined_results = combined_results.append(results)
    df_test.to_pickle(output_data_path/f"lineal_autoencoder_{data_name}.pkl")
combined_results.to_csv(output_data_path/"lineal_autoencoder_results.csv")
end_time = time.time()-start_time
print(end_time)

