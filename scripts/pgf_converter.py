# This scrips is for obtaining all the data set representation into .pgf images
# Reference: https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
# You might need to add \usepackage{pgfplots} to your main.tex file
# To add the images you just need to:
    # \begin{figure}
    #     \begin{center}
    #         \input{your_plot.pgf}
    #     \end{center}
    #     \caption{Your caption}
    # \end{figure}
import sys
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

pgf_imgs = Path("pgf_img")
if not pgf_imgs.exists():
    pgf_imgs.mkdir()

def get_rms(values):
    # return the Root Mean Square 
    # More info at https://en.wikipedia.org/wiki/Root_mean_square
    return np.sqrt(np.sum(values**2)/len(values))

def plot_rolling_average_stats(df,dataset_name,period):
    cols = df.columns.to_list()[:-1]
    df = df.abs()
    df1 = df.groupby(["timestamp"]).mean().reset_index()
    df1.index = pd.to_datetime(df1.timestamp,unit='s')
    df1 = df1.rolling(window = period).mean()
    with plt.style.context('seaborn-paper'):
        myFmt = mdates.DateFormatter('%d/%m')
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
        ax.set_title(f"Rolling average(period = {period}) of the bearing vibration snapshots")
        fig.savefig(dataset_name+".pgf")


def plot_rms_general_stats(df,dataset_name):
    cols = df.columns.to_list()[:-1]
    df1 = df.groupby(["timestamp"])[cols].apply(get_rms).reset_index()
    df1.index = pd.to_datetime(df1.timestamp,unit='s')
    with plt.style.context('seaborn-paper'):
        myFmt = mdates.DateFormatter('%d/%m')
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
        myFmt = mdates.DateFormatter('%d/%m')
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

def scatter_mean_abs_general_stats(df,dataset_name):
    #cols = df.columns.to_list()[:-1]
    df = df.abs()
    df1 = df.groupby(["timestamp"]).mean().reset_index()
    df1.index = pd.to_datetime(df1.timestamp,unit='s')
    with plt.style.context('seaborn-paper'):
        myFmt = mdates.DateFormatter('%d/%m')
        fig,ax = plt.subplots()
        x = df1.index
        ax.scatter(x,df1.b1_ch1,label = "Bearing 1 C-X",alpha=0.7,marker="x")
        ax.scatter(x,df1.b1_ch2,label = "Bearing 1 C-Y ",alpha=0.7,marker="^")
        ax.scatter(x,df1.b2_ch3,label = "Bearing 2 C-X",alpha=0.7,marker="x")
        ax.scatter(x,df1.b2_ch4,label = "Bearing 2 C-Y",alpha=0.7,marker="^")
        ax.scatter(x,df1.b3_ch5,label = "Bearing 3 C-X",alpha=0.7,marker="x")
        ax.scatter(x,df1.b3_ch6,label = "Bearing 3 C-Y",alpha=0.7,marker="^")
        ax.scatter(x,df1.b4_ch7,label = "Bearing 4 C-X",alpha=0.7,marker="x")
        ax.scatter(x,df1.b4_ch8,label = "Bearing 4 C-Y",alpha=0.7,marker="^")
        ax.set_xlabel ("Days")
        ax.legend()
        ax.xaxis.set_major_formatter(myFmt)
        ax.grid()
        ax.set_title("Scatter plot of absolute mean of the bearing vibration snapshots")
        #save_fig(fig,"{}.png".format(dataset_name),figsize = (12,6),dpi = 300)
        fig.savefig(dataset_name+".pdf")

def load_data():
    global df1,df2,df3
    df1 = pd.read_pickle("../data/1st_test_full.pkl") # first dataset
    df2 = pd.read_pickle("../data/2nd_test_full.pkl") # second one
    df3 = pd.read_pickle("../data/3th_test_full.pkl") # third one

def df1_rep():
    for index,i in enumerate([df1]):
        index = index +1 
        plot_rms_general_stats(i,str(pgf_imgs/f"rms_dataset{index}"))
        plot_mean_abs_general_stats(i,str(pgf_imgs/f"mean_abs_dataset{index}"))
        scatter_mean_abs_general_stats(i,str(pgf_imgs/f"scatter_mean_abs_dataset{index}"))
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_10_dataset{index}"),10)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_20_dataset{index}"),20)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_50_dataset{index}"),50)
        


def df2_rep():
    for index,i in enumerate([df2]):
        index = index +2
        plot_rms_general_stats(i,str(pgf_imgs/f"rms_dataset{index}"))
        plot_mean_abs_general_stats(i,str(pgf_imgs/f"mean_abs_dataset{index}"))
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_10_dataset{index}"),10)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_20_dataset{index}"),20)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_50_dataset{index}"),50)


def df3_rep():
    for index,i in enumerate([df3]):
        index = index +3
        plot_rms_general_stats(i,str(pgf_imgs/f"rms_dataset{index}"))
        plot_mean_abs_general_stats(i,str(pgf_imgs/f"mean_abs_dataset{index}"))
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_10_dataset{index}"),10)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_20_dataset{index}"),20)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_50_dataset{index}"),50)


def full_pipe():
    for index,i in enumerate([df1,df2,df3]):
        index = index +1 
        plot_rms_general_stats(i,str(pgf_imgs/f"rms_dataset{index}"))
        plot_mean_abs_general_stats(i,str(pgf_imgs/f"mean_abs_dataset{index}"))
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_10_dataset{index}"),10)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_20_dataset{index}"),20)
        plot_rolling_average_stats(i,str(pgf_imgs/f"rolling_ave_period_50_dataset{index}"),50)


if len(sys.argv) <= 1:
    print("OPTIONS:\n full_pipe \n df1 \n df2 \n df3 ")
elif sys.argv[1] == "full_pipe":
    load_data()
    full_pipe()
elif sys.argv[1] == "df1":
    load_data()
    df1_rep()
elif sys.argv[1] == "df2":
    load_data()
    df2_rep()
elif sys.argv[1] == "df3":
    load_data()
    df3_rep()
elif sys.argv[1] == "test":
    print("lol")
else:
    print("\n Full pipeline")
    load_data()
    full_pipe()