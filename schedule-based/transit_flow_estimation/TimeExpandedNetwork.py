# Time-expanded network as described in Hamdouch, Lawphongpanich (2008) will be created.
# Note: 1) sets {departure nodes}, {arrival nodes}, and {transit stops} coincide;
# 2) number num_routes will be used to label waiting or dummy links!
# 3) "_exp" means "expanded".
# 4) "time-expanded" will be abbreviated as "TE" in comments;


# Variables and formats:
# self.stops: (table, list, elmts being string)
# Fields:
# 0
# stop_name
# --------------
# self.routes: (table, list, elmts being string)
# Fields:
# 0
# route_name
# Note: not directional.
# --------------
# self.stops_exp: (table, list, elmts being string)
# Fields:
# 0
# stop_name
# stop_name format: nodeid_time, e.g. "N1_3"
# Stops will be arranged in T&C order!
# --------------
# self.stops_exp_2: (table, list, elmts being list)
# Fields:
# 0              1                 2
# stop_exp_index,actual_stop_index,time
# "stop_index" means actual stop's index. Say, index of "N" in tn.stops
# ---------------
# self.links_exp: (table, list, elmts being list)
# Fields:
# 0          1*          2*                  3*                4         5             
# link_index,route_index,starting_stop_index,ending_stop_index,link_type,travel_time,
# 6        7*       8
# frequency,capacity,travel_cost
#
# links_exp_char
# 2nd and 3rd fields are char names.
# ----------------
# self.schedule: (dictionary, key: route_direction, item: list, whose elmts being lists)
# fields:
# 0       1           2               3       4                
# trip_id,arrival_time,departure_time,stop_id,stop_sequence
# Note: arrival_time is integral time.

from math import floor
from copy import deepcopy
import json
import pickle

from open_data_file_with_header import open_data_file_with_header
from open_config_file import open_config_file

class TimeExpandedNetwork:
	"""Input physical links data to build a time-expanded network object."""

	def __init__(self):

		print('    ---------------------------------------------------')
		print("    Creating a time-expanded network object ...")
		# Config

		TEST = int(open_config_file('test_TimeExpandedNetwork'))

		TRANSIT_TYPE = open_config_file('TRANSIT_TYPE')

		# T is the time horizon for analysis horizon, with minute as unit;
		# Horizon start form 0 to (T-1) in time-dependent model.
		self.T = open_config_file('HORIZON')

		# Read files from '/data', remove header, decompose elmts.
		temp = open_data_file_with_header("data/stops.csv")
			# stop table header: 
			# 1
			# stop_id
		self.stops = [item[0] for item in temp]
		num_stops = len(self.stops)

		temp = open_data_file_with_header("data/routes.csv")
			# routes table header: 
			# 0
			# route_id eg. "R1_0"
		self.routes = [item[0] for item in temp]
		num_routes = len(self.routes)
		
		self.schedules = {}
		with open("data/schedules.json", encoding="utf8") as f:
			# stop times dictionary:
			# key: eg. "R1_0"
			# item:
			# 0*     1*            2             3*      4                
			#trip_id,arrival_time,departure_time,stop_id,stop_sequence
			self.schedules = json.load(f)

		if TEST == 1:
			print("    the horizon (T) is: " + str(self.T))
			print("    Stops:")
			print(self.stops)
			print("    --------------------------")
			print("    Transit routes:")
			print(self.routes)
			print("    --------------------------")
			print("    Transit schedules:")
			print(self.schedules)

		# Creat links in TE networks according to self.schedule.
		self.links_exp = []

		# Add traveling links.
		# Import capacity file.
		veh_cap_of_routes = open_data_file_with_header('data/vehicle_capacity_of_routes.csv')
			# Header:
			# 0
			# veh_capacity
		# Notice: sequence of routes in cap file must be the same with the seq of routes in routes file.
		# Create a route list (1st colume in veh_capacity_of_routes)
		# to pinpoint the place of the corresponding capacity,
		# since the sort of this table may be different from the self.stop.
		for key in self.schedules:
			for i in range(len(self.schedules[key])):
				# If this is not the end of this schedule ...
				if i != len(self.schedules[key]) - 1:
					# If this is not the end of a run's end ...
					#if int(self.schedules[key][i+1][4]) == int(self.schedules[key][i][4]) + 1:
					if int(self.schedules[key][i+1][0]) == int(self.schedules[key][i][0]):
						starting_time = self.schedules[key][i][1]
						# starting stop in TE network
						starting_stop_in_te = self.schedules[key][i][3] + '_' + str(starting_time)
						ending_time = self.schedules[key][i+1][1]
						ending_stop_in_te = self.schedules[key][i+1][3] + '_' + str(ending_time)

						if ending_time <= (self.T - 1):				
							# Find the bus route for current link
							current_route_index = self.routes.index(key)

							# Find capacity for current link
							# Note: set to float, since fractional flow is allowed.
							capacity = float(veh_cap_of_routes[current_route_index][0])

							# Creat traveling links in TE network
							# Notes: 
							# 1)Travel cost is initialized to be the TT;
							# 2) now the link starting stop and ending stop are stop name;
							# they will be converted to stop_index later;
							# 3) "frequency" will be left empty.
							self.links_exp.append(['', current_route_index, \
								starting_stop_in_te, ending_stop_in_te, TRANSIT_TYPE,ending_time - starting_time,\
								'', capacity, ending_time - starting_time])

		# Add waiting links.
		for i in range(num_stops):
			# At the last time T, users no longer need to wait; they "disappear".
			for t in range(self.T - 1):
				starting_stop_in_te = self.stops[i] + '_' + str(t)
				ending_stop_in_te = self.stops[i] + '_' + str(t + 1)
				# num_routes will be used to label waiting or dummy links!
				self.links_exp.append(['', num_routes,\
					starting_stop_in_te, ending_stop_in_te, 'WT', 1.0, '', float('inf'), 1.0])

		# Add index for links.
		for temp_index in range(len(self.links_exp)):
			self.links_exp[temp_index][0] = temp_index

		self.links_exp_char = deepcopy(self.links_exp)

		if TEST == 1:
			print("    --------------------------")
			print("    Links in TE network (links_exp_char):")
			print(self.links_exp_char)

		# Create stops in TE networks.
		temp_stops_exp = []
		for i in range(num_stops):
			for t in range(self.T):
				temp_stops_exp.append(self.stops[i] + '_' + str(t))

		# Find the T&C order for nodes
		
		# Initialize
		temp_links_exp = deepcopy(self.links_exp)
		self.stops_exp = []
		self.stops_exp_2 = []

		current_node = temp_stops_exp[0]
		current_node_index = 0

		while len(temp_stops_exp) != 0:
			# "flag" used to indicate whether this node has been an upstream node
			# for some link.
			flag = 0
			for i in range(len(temp_links_exp)):
				if temp_links_exp[i][2] == current_node:
					current_node = temp_links_exp[i][3]
					current_node_index = temp_stops_exp.index(current_node)
					flag = 1
					break

			# If this node has never been an upstream node ...
			if flag == 0:
				self.stops_exp.append(current_node)
				# Delete node whose reverse T&C order has been found.
				del temp_stops_exp[current_node_index]
				# Delete all links that end in current node.
				temp_links_exp = [item for item in temp_links_exp if item[3] != current_node]

				if len(temp_stops_exp) != 0:
					current_node = temp_stops_exp[0]
					current_node_index = 0

		# Note that above order is reverse T&C order.
		# Now change it to T&C order.
		# Note: don't use self.stops_exp = self.stops_exp.reverse().
		self.stops_exp.reverse()
		if TEST == 1:
			print("    --------------------------")
			print("    Stops in TE network (sorted in T&C order):")
			print(self.stops_exp)
		
		# Create stops_exp_2.
		# Find i for node i_t.
		# Find t for node i_t.
		for temp_index in range(len(self.stops_exp)):
			self.stops_exp_2.append([temp_index, \
				self.stops.index(self.stops_exp[temp_index].split('_')[0]), \
				int(self.stops_exp[temp_index].split('_')[1])])

		if TEST == 1:
			print("    --------------------------")
			print("    stops_exp_2 is:")
			print(self.stops_exp_2)

		# Convert stop name in self.links_exp to stop_index; and add link_index.
		for temp_index in range(len(self.links_exp)):
			self.links_exp[temp_index][2] = self.stops_exp.index(self.links_exp[temp_index][2])
			self.links_exp[temp_index][3] = self.stops_exp.index(self.links_exp[temp_index][3])

		if TEST == 1:
			print("    --------------------------")
			print("    List of links (with attributes changed to indices):")
			print(self.links_exp)

		print("    A time-expanded network has been successfully created!")

	# Used for finding k-skortest path algorithm.
	def links_exp_dict(self):

		links_dict = {}

		for start_stop in self.stops_exp:
			end_stops = {}
			outgoing_links = [item for item in self.links_exp_char if item[2] == start_stop]

			if outgoing_links:
				for link in outgoing_links:
					end_stops[link[3]] = float(link[8])

			links_dict[start_stop] = end_stops

		json.dump(links_dict, open("YenKSP/data/json/net.json","w"))

		for key in links_dict:
			print("    key: " + key)
			print("    item: " + str(links_dict[key]))

	# Used for finding k-skortest path algorithm.
	def stops_pickle(self):

		print("    --------------------------")
		print(self.stops)
		pickle.dump(self.stops, open('YenKSP/data/stops.p', 'wb'), protocol=2)

	# Used for finding k-skortest path algorithm.
	def stops_exp_pickle(self):

		print("    --------------------------")
		print(self.stops_exp)
		pickle.dump(self.stops_exp, open('YenKSP/data/stops_exp.p', 'wb'), protocol=2)
