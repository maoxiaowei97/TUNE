# Code Implementation of TUNE

## Chengdu and Xi'an Data
The dataset can be accessed from the following link:
[Chengdu and Xi'an Data](https://pan.baidu.com/share/init?surl=S6ZzlsHdDjjBzpe5-wFPEQ&pwd=rava)

## Data Preprocessing
To preprocess the data, run the following script:
```bash
python data_process/preprocess/mm/process_all.py
```



## Estimating Travel Time Distributions of Segments using LGDE

To estimate the travel time distributions of segments, execute the script:

```
python segment_dist_estimation/run.py
```



## Estimating Travel Time Uncertainty of Routes using MGUQ

To estimate the travel time uncertainty of routes, run the following command:

```
python run.py
```



## Online Inference

For online inference, use the script:

```
python run_online.py
```
