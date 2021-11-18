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

def save_fig(fig,fig_name,**kwargs):
    # saves the figure to a common folder
    import os
    if not os.path.exists("figures"):
        os.mkdir("figures")
    fig.savefig(fname= "figures/{}".format(fig_name) , **kwargs)
    
def rms(values):
    # return the Root Mean Square 
    # More info at https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(np.sum(values**2)/len(values))

def spree_plot(pca_object,fig_title=None,fig_name = None,**kwargs):
    # returns a spree plot from a PCA object
    per_var = np.round(pca_object.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    fig, ax = plt.subplots()
    ax.bar(x = range(1,len(per_var)+1),height=per_var, tick_label=labels)
    ax.set_ylabel('Percentage of Explained Variance')
    ax.set_xlabel('Principal Component')
    if fig_title:
        ax.set_title("Scree Plot")
    else:
        ax.set_title(fig_title)
    if fig_name:
        save_fig(fig,fig_name,**kwargs)
def 