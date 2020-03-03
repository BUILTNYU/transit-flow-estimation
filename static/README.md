# Transit Flow Estimation Tool (static flow version) V1.0

## Descriptions
This program is designed to estimate/predict transit flow based on hsitory measurement (count) data at stops.
The assignment model is based on the classic work: 
Spiess, H. and Florian, M., 1989. Optimal strategies: a new assignment model for transit networks. Transportation Research Part B: Methodological, 23(2), pp.83-102.
Two modes are supported: 1) transit flow assignment mode; 2) transit flow estimation mode.

## How to Run
0. The computer must support python3; install dependencies;
1. Prepare and put data in transit_flow_estimation;
2. Modify the config/config.json file to configure the program;
3. Run transit_flow_estimation.py; results would be stored in results folder.

## Dependencies
numpy, cvxopt

## Data Requirements
 - stop.cvs
 - routes.cvs
 - links.cvs
 - frequencies.cvs
 - vehicle_capacity_of_routes.csv
 - entry_count_by_hour.cvs
 - exit_count_hour.cvs
 - wifi_count_by_hour.cvs

## Contributors
Qi Liu (ql375@nyu.edu)
Joseph Chow (joseph.chow@nyu.edu)
C2SMART, Tandon School of Engineering, NYU
