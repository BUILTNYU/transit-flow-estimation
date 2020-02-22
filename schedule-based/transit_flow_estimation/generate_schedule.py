import json
import csv
import math

from open_config_file import open_config_file
from open_data_file_with_header import open_data_file_with_header

num_routes = 4
T = int(open_config_file('HORIZON'))

temp = open_data_file_with_header("data/routes.csv")
routes = [item[0] for item in temp]

schedules = {}

# The starting time of each route.
starting_time = [0,0,0,0,0,0,0,0]
# Unit: min.
frequencies = [15,15,15,15,15,15,15,15]


for l in range(num_routes):
	filename_stops = "data/R" + str(l + 1) + "_0.csv"
	filename_tt = "data/R" + str(l + 1) + "_tt.csv"
	stops = []
	tt = []

	with open(filename_stops) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=",")
		for row in readCSV:
			stops.append(row[0])

	with open(filename_tt) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=",")
		for row in readCSV:
			tt.append(int(row[0]))

	print()
	print("current route: R" + str(l + 1))
	print()
	print("stops:")
	print(stops)
	print("tt")
	print(tt)

	key = "R" + str(l + 1) + "_0"
	schedules[key] = []
	starting_time_of_run = starting_time[2 * l]
	run_count = 1

	while True:	
		accumulate_time = starting_time_of_run
		accumulate_stop = 1
		for i in range(len(stops)):
			schedules[key].append([run_count,accumulate_time,"",stops[i],accumulate_stop])
			if i != len(stops) - 1:
				accumulate_time += tt[i]
				accumulate_stop += 1

		starting_time_of_run += frequencies[2 * l]
		run_count += 1

		if starting_time_of_run >= 120:
			break

	print()
	print("schedule of " + key)
	print(schedules[key])

	key = "R" + str(l + 1) + "_1"
	schedules[key] = []
	starting_time_of_run = starting_time[ 2 * l + 1]
	run_count = 1

	while True:
		accumulate_time = starting_time_of_run
		accumulate_stop = 1
		for i in range(len(stops) - 1, -1, -1):
			schedules[key].append([run_count,accumulate_time,"",stops[i],accumulate_stop])
			if i != 0:
				accumulate_time += tt[i - 1]
				accumulate_stop += 1

		starting_time_of_run += frequencies[2 * l + 1]
		run_count += 1

		if starting_time_of_run >= 120:
			break

	print()
	print("schedule of " + key)
	print(schedules[key])

with open('data/schedules.json', 'w') as fp:
	json.dump(schedules, fp, ensure_ascii=False)