# This scrips is for obtaining all the data set representation into
# .pgf images
# Reference: https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
# You might need to add \usepackage{pgfplots} to your main.tex file
# To add the images you just need to:
    # \begin{figure}
    #     \begin{center}
    #         \input{your_plot.pgf}
    #     \end{center}
    #     \caption{Your caption}
    # \end{figure}

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
matplotlib.style.context("seaborn-paper")
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def rms(values):
    # return the Root Mean Square 
    # More info at https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(np.sum(values**2)/len(values))


def plot_rms_general_stats(df,dataset_name):
    cols = df.columns.to_list()[:-1]
    df1 = df.groupby(["timestamp"])[cols].apply(rms).reset_index()
    df1.index = pd.to_datetime(df1.timestamp,unit='s')
    with plt.style.context('seaborn-paper'):
        myFmt = mdates.DateFormatter('%d')
        fig,ax = plt.subplots()
        if len(cols) <6:
            ax.plot(df1.b1_ch1,label = "Bearing 1")
            ax.plot(df1.b2_ch2,label = "Bearing 2")
            ax.plot(df1.b3_ch3,label = "Bearing 3")
            ax.plot(df1.b4_ch4,label = "Bearing 4")
        else:
            ax.plot(df1.b1_ch1,label = "Bearing 1 C-X")
            ax.plot(df1.b1_ch2,label = "Bearing 1 C-Y ")
            ax.plot(df1.b2_ch3,label = "Bearing 2 C-X")
            ax.plot(df1.b2_ch4,label = "Bearing 2 C-Y")
            
            ax.plot(df1.b3_ch5,label = "Bearing 3 C-X")
            ax.plot(df1.b3_ch6,label = "Bearing 3 C-Y")
            ax.plot(df1.b4_ch7,label = "Bearing 4 C-X")
            ax.plot(df1.b4_ch8,label = "Bearing 4 C-Y")
        ax.set_xlabel ("Days")
        ax.legend()
        ax.xaxis.set_major_formatter(myFmt)
        ax.grid()
        ax.set_title("RMS of the bearing vibration snapshots")
        fig.savefig(dataset_name+".pgf")
        #save_fig(fig,"{}.png".format(dataset_name),figsize = (12,6),dpi = 300)

def plot_mean_abs_general_stats(df,dataset_name):
    cols = df.columns.to_list()[:-1]
    df = df.abs()
    df1 = df.groupby(["timestamp"]).mean().reset_index()
    df1.index = pd.to_datetime(df1.timestamp,unit='s')
    with plt.style.context('seaborn-paper'):
        myFmt = mdates.DateFormatter('%d')
        fig,ax = plt.subplots()
        if len(cols) <6:
            ax.plot(df1.b1_ch1,label = "Bearing 1")
            ax.plot(df1.b2_ch2,label = "Bearing 2")
            ax.plot(df1.b3_ch3,label = "Bearing 3")
            ax.plot(df1.b4_ch4,label = "Bearing 4")
        else:
            ax.plot(df1.b1_ch1,label = "Bearing 1 C-X")
            ax.plot(df1.b1_ch2,label = "Bearing 1 C-Y ")
            ax.plot(df1.b2_ch3,label = "Bearing 2 C-X")
            ax.plot(df1.b2_ch4,label = "Bearing 2 C-Y")
            
            ax.plot(df1.b3_ch5,label = "Bearing 3 C-X")
            ax.plot(df1.b3_ch6,label = "Bearing 3 C-Y")
            ax.plot(df1.b4_ch7,label = "Bearing 4 C-X")
            ax.plot(df1.b4_ch8,label = "Bearing 4 C-Y")
        ax.set_xlabel ("Days")
        ax.legend()
        ax.xaxis.set_major_formatter(myFmt)
        ax.grid()
        ax.set_title("Absolute mean of the bearing vibration snapshots")
        #save_fig(fig,"{}.png".format(dataset_name),figsize = (12,6),dpi = 300)
        fig.savefig(dataset_name+".pgf")

pgf_imgs = Path("pgf_img")
if not pgf_imgs.exists():
    pgf_imgs.mkdir()

df1 = pd.read_pickle("../data/1st_test_full.pkl") # first dataset
df2 = pd.read_pickle("../data/2nd_test_full.pkl") # second one
df3 = pd.read_pickle("../data/3th_test_full.pkl") # third one

plot_rms_general_stats(df1,str(pgf_imgs/"rms_dataset1"))
plot_mean_abs_general_stats(df1,str(pgf_imgs/"mean_abs_dataset1"))
plot_rms_general_stats(df2,str(pgf_imgs/"rms_dataset2"))
plot_mean_abs_general_stats(df2,str(pgf_imgs/"mean_abs_dataset2"))
plot_rms_general_stats(df3,str(pgf_imgs/"rms_dataset3"))
plot_mean_abs_general_stats(df3,str(pgf_imgs/"mean_abs_dataset3"))

