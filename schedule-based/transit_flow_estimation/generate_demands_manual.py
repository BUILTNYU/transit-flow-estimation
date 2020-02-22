import numpy as np

from TimeExpandedNetwork import TimeExpandedNetwork
from open_config_file import open_config_file
from random_choice import random_choice

tn = TimeExpandedNetwork()

T = tn.T
N = len(tn.stops)
od_flow = np.zeros((N,N,T))

q = 1
r = 10
for h in range(20):
	od_flow[q,r,h] = 4

for h in range(20,40):
	od_flow[q,r,h] =8

for h in range(40,60):
	od_flow[q,r,h] = 8


q = 2
r = 12
for h in range(20):
	od_flow[q,r,h] = 4

for h in range(20,40):
	od_flow[q,r,h] =8

for h in range(40,60):
	od_flow[q,r,h] = 8

q = 5
r = 12
for h in range(20):
	od_flow[q,r,h] = 4

for h in range(20,40):
	od_flow[q,r,h] =8

for h in range(40,60):
	od_flow[q,r,h] = 8

print("Total demands: " + str(od_flow.sum()))

np.save('data/od_flow', od_flow)