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
### Datasheet of the bearings
As explained on the readme document inclued inside the IMS dataset. The used bearing are the [REXNORD ZA2115](https://www.rexnord.com/Products/za2115).  
The information has been transformed to the metric system.

| Features                                                       | Value          |
|----------------------------------------------------------------|----------------|
| A)Base to Centerline Height[cm]                                | 5,715          |
| (B)Distance Between Bolt Holes[cm]                             | 15,875         |
| (C)Mounting Pad Length[cm]                                     | 20,6375        |
| (D)Length Through Bore[cm]                                     | 7,9375         |
| (E)Mounting Pad Width[cm]                                      | 6,35           |
| (F)Mounting Bolt Size[cm]                                      | 1,5875         |
| (H)Mounting Pad Height[cm]                                     | 3,4925         |
| (M)Locking Collar Outside Diameter[cm]                         | 7,3025         |
| (N)Inner Ring Hub Diameter[cm]                                 | 5,87375        |
| (P)Distance from Housing Cast Face to Locking Feature Face[cm] | 1,74625        |
| (S)Mounting Bolt Hole Slot Length[cm]                          | 2,54           |
| C Basic Dynamic Load Rating[N]                                 | 1285084,98     |
| Co Basic Static Load Rating[N]                                 | 1476802,4      |
| Grease Fitting Size                                            | 1/8 in NPT     |
| Lubrication Type                                               | Exxon Ronex MP |
| Maximum Speed[RPM]                                             | 4050           |
| Maximum Temperature [C]                                        | 107            |
| Microlock Kit                                                  | ML2            |
| Replacement Insert                                             | 2115U          |
| Replacement Seal Kit                                           | ZS6            |
| Setscrew Torque [Nm]                                           | 36,72          |
| Shaft Diameter[cm]                                             | 4,92125        |
| Shaft Locking Collar Kit                                       | SC6            |
| Size Code                                                      | 6              |
| Threaded Cover Kit                                             | TC6            |
| Type of Seal                                                   | Clearance Seal |
| Vibration Frequency Fundamental Train                          | 0,0072         |
| Vibration Frequency Inner Ring Defect                          | 0,1617         |
| Vibration Frequency Outer Ring Defect                          | 0,1217         |
| Vibration Frequency Roller Spin                                | 0,0559         |
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