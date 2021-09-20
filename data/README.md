# Information about the datasets used in this project
Due to the number of files and its combined weight, it has been chosen not to
upload the whole dataset to this repo. The whole dataset is available in this 
[link](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing)  
The setup is installed as shown on the next image.


![Setup](img/setup.png)
## General info of the dataset  
The dataset is composed of three different data packets that are divided into many files which
contains exactly 20 480 points recorded every 1 second at a sampling rate of 20kHz.
The next table shows the general features of each set.  
|Set number   |Start Date   |End Date   |No. of files   | No. of Bearing   |No. of channels   | Recording Interval  | Format  | Description  |
|---|---|---|---|---|---|---|---|---|
| 1  | 22/10/2003  | 25/11/2003  | 2156  | 4  | 8  | Every 10 min except first 43 (5min)  | ASCII  |  Test to failure experiment, inner race defect ocurred in bearing 3 and roller element defect in bearing 4 |
| 2  | 12/02/2004  | 19/02/2003  | 984  | 4  | 4  | 10 min   | ASCII  | Test to failure, outer race failure occured in bearing 1  |
| 3 | 4/03/2004 |4/04/2004 |4448 |4 |4 |10 min |ASCII | Test to failure, outer race failure ocurred in bearing 3 |

Supposedly, the third dataset only contains recording until the April 4,2004 but after exploring the dataset, it has been found that there are recordings dated until the 18th of April.

## Union of the datasets 
In order to simplify the loading of the data. All the invidual file have been united in one for each dataset. This has been acomplished using the ```data_unifier.py``` script. The next lines of code show how to use it:
```python
from data_unifier import * # this may change in case the scrip is imported in a different way

join_files("data/3th_test",data_format = "pickle",n_cols = 4) # join the files into 10 different datasets to ease memory usage
join_end("3th_test_pickle","3th_test") # joins all the data as a file
```
Further instruccions may be found inside the code.
## Final Datasets
The fully joined datasets will be available for downlading in the next weeks.