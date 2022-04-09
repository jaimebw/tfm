
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib import font_manager

# Add fonts to the plot
font_dirs = ['../otros/lmr']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
matplotlib.style.context("seaborn-paper")
plt.rcParams['font.family'] = 'Latin Modern Roman'

myFmt = mdates.DateFormatter("%d/%m")
def mahal_plot(df):
    with plt.style.context("seaborn-paper"):
        fig, ax = plt.subplots(figsize = (12,6))
        ax.plot(df.dist_mob)
        ax.set_yscale("log")
        ax.set_ylabel("Mahalanobis distance")
        ax.axhline(df.thresh.values[0],color = "red", linestyle = "--")
        #ax.axvline(df.loc[df.anomaly == True].index[0], color = "green")
        ax.xaxis.set_major_formatter(myFmt)
        ax.set_xlabel("Days")
        ax.annotate('Anomaly', xy=(df.loc[df.anomaly == True].index[0], df.thresh.values[0]),  xycoords='data',
                xytext=(0.4, 0.4), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="fancy",connectionstyle="arc3",color ="black"),
                horizontalalignment='right', verticalalignment='top',
                )

pdf_imgs = Path("pdf_img")
if not pdf_imgs.exists():
    pdf_imgs.mkdir()

input_data = Path("../data/feature_data/rms")
index = 0
for dataset in input_data.glob("*.pkl"):
    if index == 0:
        print(f"Dataset:{dataset.stem} \n")
        df = pd.read_pickle(dataset)
        fname = str(pdf_imgs/dataset.stem)+"_scatter.pdf"
        with plt.style.context("seaborn-paper"):
            fig,ax = plt.subplots()
            x = df.index
            ax.scatter(x, df.b1_ch1, label="Bearing 1 C-X", alpha=0.7, marker="x")
            ax.scatter(x, df.b1_ch2, label="Bearing 1 C-Y ", alpha=0.7, marker="^")
            ax.scatter(x, df.b2_ch3, label="Bearing 2 C-X", alpha=0.7, marker="x")
            ax.scatter(x, df.b2_ch4, label="Bearing 2 C-Y", alpha=0.7, marker="^")
            ax.scatter(x, df.b3_ch5, label="Bearing 3 C-X", alpha=0.7, marker="x")
            ax.scatter(x, df.b3_ch6, label="Bearing 3 C-Y", alpha=0.7, marker="^")
            ax.scatter(x, df.b4_ch7, label="Bearing 4 C-X", alpha=0.7, marker="x")
            ax.scatter(x, df.b4_ch8, label="Bearing 4 C-Y", alpha=0.7, marker="^")
            ax.grid()
            ax.xaxis.set_major_formatter(myFmt)
            ax.legend()
            ax.set_xlabel("Days")
            ax.set_ylabel("RMS of the vibration")
            fig.savefig(fname = fname, dpi = 300)
    else:
        pass



