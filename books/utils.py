import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path


class ExperimentData:
    def __init__(self,path_to_data):
        if path_to_data:
            self.path_to_data = Path(path_to_data)
            self.data = pd.read_pickle(self.path_to_data)
        self.df1 = Path("/Users/jaime/repos/tfm/data/1st_test_full.pkl")
        self.df2 = Path("/Users/jaime/repos/tfm/data/2nd_test_full.pkl")
        self.df3 = Path("/Users/jaime/repos/tfm/data/3rd_test_full.pkl")
        
    def full_df1(self):
        return pd.read_pickle(self.df1)
    
    def full_df2(self):
        return pd.read_pickle(self.df2)
    
    def full_df3(self):
        return pd.read_pickle(self.df3)
    
    def features(self,dataset ,feature_type = "rms"):
        """
        Feature types:
        - rms
        - kurt
        - skew
        - sfactor: Shape Factor
        """
        if self.path_to_data:
            df = self.data
        else:
            if dataset:
                if dataset == "1":
                    df = self.full_df1()
                elif dataset == "2":
                    df = self.full_df2()
                else:
                    df = self.full_df3()
        

        df = df.abs()
        if feature_type == "rms":
            df1 = df.groupby(["timestamp"]).apply(get_rms)
        elif feature_type == "kurt":
            df1 = df.groupby(["timestamp"]).apply(pd.DataFrame.kurt)
        elif feature_type == "skew":
            df1 = df.groupby(["timestamp"]).apply(pd.DataFrame.skew)
        elif feature_type == "sfactor":
            df1 = df.groupby(["timestamp"]).apply(get_shapefactor)
        df1.drop(columns=["timestamp"], inplace=True)
        df1.index = pd.to_datetime(df1.index, unit="s")
        return df1


def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    """
    Return the MahalanobiDist of a DataFrame for all its columns
    More info at : https://en.wikipedia.org/wiki/Mahalanobis_distance
    """
    inv_covariance_matrix = inv_cov_matrix 
    diff = data - mean_distr # datos centrados
    md = np.array([]) # lista final de datos
    for i in range(len(diff)):
        maha_dist = np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i]))
        md = np.append(md)
    return md


def str_datetime(file_name):
    # Returns the unix timestamp of the date of the file
    date = file_name.split(".")
    date = list(map(int, date))
    return int(date.timestamp())


def load_data(path, n_channels=8):
    # Loads the data from the dataset in a Dataframe
    cols = ["b1_ch1", "b2_ch2", "b3_ch3", "b4_ch4"]
    if n_channels == 8:
        cols = [
            "b1_ch1",
            "b1_ch2",
            "b2_ch3",
            "b2_ch4",
            "b3_ch5",
            "b3_ch6",
            "b4_ch7",
            "b4_ch8",
        ]
    df = pd.read_csv(path, delimiter="\t")
    df.columns = cols
    _, file_name = os.path.split(path)
    init_date = str_datetime(file_name)
    df = df.astype("float32")
    df["timestamp"] = np.arange(init_date, init_date + df.shape[0])
    df = df.set_index("timestamp")
    return df


def save_fig(fig, fig_name, **kwargs):
    # saves the figure to a common folder

    if not os.path.exists("figures"):
        os.mkdir("figures")
    fig.savefig(fname="figures/{}".format(fig_name), **kwargs)


def get_absmean(values):
    # returns the absolute mean
    return np.sum(np.abs(values))/len(values)


def get_rms(values):
    # return the Root Mean Square
    # More info at https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(np.sum(values ** 2) / len(values))


def get_shapefactor(values):
    # returns the shape factor 
    # More info at https://en.wikipedia.org/wiki/Shape_parameter
    shapefactor = get_rms(values)/get_absmean(values)
    return shapefactor


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
        save_fig(fig, fig_name, **kwargs)