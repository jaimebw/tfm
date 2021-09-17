# Author: Jaime Bowen 
# Once the IMS dataset is uncompressed, this script creates a .pkl file that contais all the experimental
# data at once.
# Example: 1st_test --> will create 1st_test.hdf5
import os
import json
import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

def str_datetime(file_name) :
    date = file_name.split(".")
    date = list(map(int, date))
    date = datetime.datetime(*date)
    return int(date.timestamp())

def join_files(path_to_dir,data_format = "pkl",n_cols = 8):
    """
    Creates a directory that join all the contained files in groups of 10%
    of the total number of files to speed the process.
    Add to the DataFrame, time index
    """
    global final_dir
    global dir_name

    _,dir_name = os.path.split(path_to_dir)
    final_dir = "{}_{}".format(dir_name,data_format)
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    files = os.listdir(path_to_dir)
    files.sort()
    
    DF = pd.DataFrame()
    num_files = len(files)
    files_count = 0 # counts the number of times that
    num_ent = int(num_files/10)
    for index,file in enumerate(tqdm(files)):
        dir_file = "{}/{}".format(path_to_dir,file) # file name
        cols = ["b1_ch1","b1_ch2","b2_ch3","b2_ch4","b3_ch5","b3_ch6","b4_ch7","b4_ch8"]
        if n_cols == 4:
            cols = ["b1_ch1","b2_ch2","b3_ch3","b4_ch4"]
        df = pd.read_csv(dir_file, delimiter = "\t",names = cols)
        init_date = str_datetime(file)
        df = df.astype("float32")
        
        #df["timestamp"] = np.arange(init_date,init_date + df.shape[0])
        df["timestamp"] = np.full(df.shape[0],init_date)
        DF = DF.append(df,ignore_index=True)

        if index == num_ent:
            files_count += 1
            if files_count == 10:
                pass
            else:
                num_ent = int(num_files/10) * (files_count+1)
                if data_format == "hdf5":
                    DF.to_hdf("{}/{}_{}.{}".format(
                    final_dir,dir_name,files_count),
                    key = dir_name)
                else:
                    DF.to_pickle("{}/{}_{}.{}".format(
                        final_dir,dir_name,files_count,"pkl"
                    ))
                DF = pd.DataFrame() # DF is freed from memory
    if data_format == "hdf5":
        DF.to_hdf("{}/{}_{}.{}".format(
            final_dir,dir_name,files_count),
        key = dir_name)
    else:
        DF.to_pickle("{}/{}_{}.{}".format(
            final_dir,dir_name,files_count,"pkl"
        ))
    

def join_end(final_dir,dir_name,data_format = "pkl",delete_secondary_files = True):
    """
    Join all the files inside the directory as a single file.
    """
    files_indir = os.listdir(final_dir)
    files_indir.sort()
    DF = pd.DataFrame()
    for index, file in enumerate(tqdm(files_indir)):
        df = pd.read_pickle(final_dir+"/"+file)
        DF = DF.append(df,ignore_index=True)
    if delete_secondary_files == True:
        filtered_files = [file for file in files_indir if file.endswith(".pkl")]
        for file in filtered_files:
            path_to_file = os.path.join(final_dir,file)
            os.remove(path_to_file)
    DF = DF.sort_values(by = ["timestamp"],ignore_index= True)
    DF.to_pickle("{}_full.{}".format(
        dir_name,"pkl"
    ))
    


