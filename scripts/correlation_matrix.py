# Returns the correlation matrix for all features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

input_data_path = Path("../data/feature_data")
output_plot_path = Path("../data/confussion_matrix")
if not output_plot_path.exists():
    output_plot_path.mkdir()

cols = {"b1_ch1":"Bearing 1","b2_ch2":"Bearing 2", "b3_ch3":"Bearing 3","b4_ch4":"Bearing 4"}

features_paths = []
for i in input_data_path.glob("*"):
    if i.is_dir():
        for j in i.glob("*.pkl"):
            features_paths.append(j)
"""

for path in features_paths:
    #print(path)
    fname = f"cmatrix_{path.stem}.pdf"
    ouput_path = output_plot_path/fname
    df = pd.read_pickle(path)
    df.rename(columns = cols,inplace = True)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize = (6,6))
        sns.heatmap(df.corr(), annot=True,ax = ax,cmap="Blues" ,linewidths=.5,square=True)
        fig.savefig(ouput_path,dpi = 300)
"""
for i in input_data_path.glob("*"):
    if i.is_dir():
        if str(i) == "../data/feature_data/rolling_avg":
            pass
        else:
            fname =f"cmatrix_{i.stem}.pdf"
            ouput_path = output_plot_path/fname
            with sns.axes_style("white"):
                fig, axess = plt.subplots(1,3,figsize = (20,8))
                for index, j in enumerate(i.glob("*.pkl")):
                    #fname = f"cmatrix_{j.parents[0]}.pdf
                    print(j)
                    df = pd.read_pickle(j)
                    df.rename(columns = cols,inplace = True)
                    sns.heatmap(df.corr(), annot=True,ax = axess[index],cmap="Blues" ,linewidths=.5,square=True,cbar_kws={"orientation": "horizontal"})
                fig.savefig(ouput_path,dpi = 300) 
                plt.close(fig)