# This scrpt can convert the OD flows and proportion matrix obtained from the 
# schedule-based assignment model into measurements.

import numpy as np
import sys
from TimeExpandedNetwork import TimeExpandedNetwork
from open_config_file import open_config_file
from open_data_file_with_header import open_data_file_with_header

tn = TimeExpandedNetwork()

T = int(open_config_file('HORIZON'))

# Parameters
DISPLAY_PROGRESS = int(open_config_file('DISPLAY_PROGRESS'))

# Number of stops
N = len(tn.stops)

# Periods of measurements (Unit: min)
C = int(open_config_file('MEASUREMENT_PERIOD'))

# Times of measurements
K = int(T / C)

# Input: OD flows
od_flow = np.load('data/od_flow.npy')
# Input: proportions
od_flow_to_stop_prob = np.load('data/od_flow_to_stop_prob.npy')
link_combined_flow = np.load('data/link_combined_flow.npy')

print()
print("    Horizon: " + str(T))
print("    Measurements period C: " + str(C))
print("    Times of measurements K: " + str(K))
print("    Total demands: " + str(od_flow.sum()))
print()

# Check flow 
#print("----------------------")
#print("non-zero od_flow_to_stop_prob[q,r,h,i,t]:")
#for q in range(N):
#	for r in range(N):
#		for h in range(T):
#			for i in range(N):
#				for t in range(T):
#					if od_flow_to_stop_prob[q,r,h,i,t] > 0.0001:
#						if q == i:
#							print("enter: " + str([q,r,h,i,t]))
#						elif r == i:
#							print("exit: " + str([q,r,h,i,t]))
#						else:
#							print("transfer: " + str([q,r,h,i,t]))
#
#						print(od_flow_to_stop_prob[q,r,h,i,t])

#print("---------------------")
#print("non-zero link_combined_flow:")
#for a in range(len(link_combined_flow)):
#	if link_combined_flow[a] > 0.001:
#		print(str(tn.links_exp_char[a]) + ": " + str(link_combined_flow[a]))

# Conversion
# 0) import ratio file.
wifi_sample_ratio = open_data_file_with_header('data/wifi_sample_ratio.csv')
# Table header: 
# 0
# sample_ratio
wifi_sample_ratio = [float(item[0]) for item in wifi_sample_ratio]

# 1) Initialization
entry_count = []
exit_count = []
passby_count = []
wifi_count = []

for temp_index in range(N * K):
	passby_count.append(0.0)
	entry_count.append(0.0)
	exit_count.append(0.0)
	passby_count.append(0.0)
	wifi_count.append(0.0)

print('    ----------------------')
# 2) Conversion
# k-th measurement
for q in range(N):
	for r in range(N):
		for h in range(T):
			for i in range(N):
				for t in range(T):

					k = int((t - (t % C)) / C)

					# Enter measurements
					if i == q and h == t:
						entry_count[k * N + i] += od_flow[q,r,h]

					# Exit measurements
					if i == r: 
						if od_flow_to_stop_prob[q,r,h,i,t] > 0.0001:
							if DISPLAY_PROGRESS == 1:
								print("    non-zero exit[q,r,h,r,t]: " + str([q,r,h,i,t]) + str(od_flow_to_stop_prob[q,r,h,i,t]))
							exit_count[k * N + i] += od_flow[q,r,h] * od_flow_to_stop_prob[q,r,h,i,t]

							if od_flow_to_stop_prob[q,r,h,i,t] > 1.0001:
								print()
								print("Error: od_flow_to_stop_prob" + str([q,r,h,i,t]) + ": " + str(od_flow_to_stop_prob[q,r,h,i,t]) + " > 1!")
								print("od_flow" + str([q,r,h]) + ": " + str(od_flow[q,r,h]))
								print("Type: exit")

					# Stopby measurements
					if i!= q and i!= r:
						if od_flow_to_stop_prob[q,r,h,i,t] > 0.0001:
							if DISPLAY_PROGRESS == 1:
								print("    non-zero passby[q,r,h,i,t]: " + str([q,r,h,i,t]) + str(od_flow_to_stop_prob[q,r,h,i,t]))
							passby_count[k * N + i] += od_flow[q,r,h] * od_flow_to_stop_prob[q,r,h,i,t]

							if od_flow_to_stop_prob[q,r,h,i,t] > 1.00001:
								print()
								print("Error: od_flow_to_stop_prob" + str([q,r,h,i,t]) + ":" + str(od_flow_to_stop_prob[q,r,h,i,t]) + " > 1!")
								print("od_flow" + str([q,r,h]) + ": " + str(od_flow[q,r,h]))
								print("Type: stopby")

# conver to wifi count
for i in range(N):
	for k in range(0, K):
		wifi_count[k * N + i] = passby_count[k * N + i] * wifi_sample_ratio[i]

# Write files
f_entry = open('data/entry_count_by_designated_period.csv','w',encoding='utf-8')
f_exit = open('data/exit_count_by_designated_period.csv','w',encoding='utf-8')
f_wifi = open('data/wifi_count_by_designated_period.csv','w',encoding='utf-8')

header = "measurement_seq_number,stop_id,count" + '\n'
f_entry.write(header)
f_exit.write(header)
f_wifi.write(header)

for k in range(K):
	for i in range(N):
		stop_name = tn.stops[i]
		line_entry = str(k) + ',' + stop_name + ',' + str(entry_count[k * N + i]) + '\n'
		line_exit = str(k) + ',' + stop_name + ',' + str(exit_count[k * N + i]) + '\n'
		line_wifi = str(k) + ',' + stop_name + ',' + str(wifi_count[k * N + i]) + '\n'
		
		f_entry.write(line_entry)
		f_exit.write(line_exit)
		f_wifi.write(line_wifi)

f_entry.close()
f_exit.close()
f_wifi.close()