
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


pdf_imgs = Path("pdf_img")
if not pdf_imgs.exists():
    pdf_imgs.mkdir()

input_data = Path("../data/feature_data/rms")

for dataset in input_data.glob("*.pkl"):
    print(f"Dataset:{dataset.stem} \n")
    df = pd.read_pickle(dataset)
    fname = str(pdf_imgs/dataset.stem)+".pdf"
    with plt.style.context("seaborn-paper"):
        fig,ax = plt.subplots()
        df.plot(ax = ax)
        ax.grid()
        ax.set_xlabel("Days")
        ax.set_ylabel("RMS of the vibration")
        fig.savefig(fname = fname, dpi = 300)




