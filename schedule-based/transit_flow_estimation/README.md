# Transit Flow Estimation Tool (schedule-based flow version) V1.0

## Descriptions
This program is designed to estimate/predict transit flow based on count data at stops; see paper Liu and Chow - schedule-based transit flow estimation for more details.
Two modes are supported: 1) transit flow assignment mode; 2) transit flow estimation mode.

## How to Run
0. The computer must support python3; install dependencies;
1. Prepare and put the data in transit_flow_estimation/data folder (this folder already contains data based on Sioux Falls network);
2. Configure the model by modifying the config/config.json file;
3. Run transit_flow_estimation.py; results would be stored in results folder.

## Data Requirements
 - stop.cvs
 - routes.cvs
 - links.cvs
 - stoptimes.csv
 - vehicle_capacity_of_routes.csv
 - entry_count_by_designated_period.csv
 - exit_count_by_designated_period.csv
 - wifi_count_by_designated_period.csv

## Dependencies
numpy, groubi (preferred) or cvxopt

## Naming Conventions for Variables:
 Transit Stops ID: N1, N2 ...;
 Transit Routes ID: R1, R2 ...;
 Directional Transit Routes ID: (Route_Direction) R1_0, R1_1,R2_0 ...;
 Transit Links ID: L1, L2 ...;
 Additonal Links ID in static model: LA1, LA2 ...;
 Additional Nodes ID in static model: (Node_Route_Direation) N1_R1_0 ...;
 Services ID: S1, S2, ...;
 Transit Trips ID: (Service_Route_Direction_trip) S1_R1_0_1, S1_R1_1_3...;
 Nodes in time-expanded network: (Node_time) N1_23 ...;

 "exp" means "expanded";
 "_set" means that this represents a set;
 "_flow" means this variable represents flow;
 "_prob" means that it represents probability;
 "_flag" means that it's a flag variable;
 "num_" means "number of ...";
 "_last" means from last iteration; 
 "sn" for static network obj;
 "tn" for time-expanded network obj;

 These terms are frequenctly used:
  "starting_stops", "ending_stops" for links;
  "incoming_links", "outgoing_links" for nodes;
  "upstream_stops", "down_stream_stops" for nodes.

## Contributors
Qi Liu (ql375@nyu.edu
Joseph Chow (joseph.chow@nyu.edu)
C2SMART, Tandon School of Engineering, NYU
