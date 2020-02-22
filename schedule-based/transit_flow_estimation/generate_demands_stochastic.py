import numpy as np

from TimeExpandedNetwork import TimeExpandedNetwork
from open_config_file import open_config_file
from random_choice import random_choice

tn = TimeExpandedNetwork()

DEMAND_LEVEL = open_config_file('DEMAND_LEVEL')
ANALYSIS_START = open_config_file('ANALYSIS_START')
ANALYSIS_PERIOD = open_config_file('ANALYSIS_PERIOD')

T = tn.T
N = len(tn.stops)
od_flow = np.zeros((N,N,T))

if DEMAND_LEVEL == 'high':
	for q in range(N):
		for r in range(N):

			if q != r:
				for h in range(ANALYSIS_START, ANALYSIS_START + ANALYSIS_PERIOD):
					od_flow[q,r,h] = random_choice("high")

if DEMAND_LEVEL == 'mid':
	for q in range(N):
		for r in range(N):

			if q != r:
				for h in range(ANALYSIS_START, ANALYSIS_START + ANALYSIS_PERIOD):
					od_flow[q,r,h] = random_choice("mid")

if DEMAND_LEVEL == 'low':
	for q in range(N):
		for r in range(N):

			if q != r:
				for h in range(ANALYSIS_START, ANALYSIS_START + ANALYSIS_PERIOD):
					od_flow[q,r,h] = random_choice("low")

if DEMAND_LEVEL == 'verylow':
	for q in range(N):
		for r in range(N):

			if q != r:
				for h in range(ANALYSIS_START, ANALYSIS_START + ANALYSIS_PERIOD):
					od_flow[q,r,h] = random_choice("verylow")

print("Total demands: " + str(od_flow.sum()))

np.save('data/od_flow', od_flow)