# Transit network illustrated in Spiess(1989) will be created. 
# Note: self.routes, self.stops_exp and and self.links_exp are all expanded version, in that
# - self.routes are decomposed by directions;
# - self.stops_exp include additional nodes like N1_R2;
# - self.links_exp include additional links like waiting, walking links.

# Formats:
# --------------
# self.stops: (list)
# Fields:
# 0
# stop_id
# --------------
# self.stops_exp: (list)
# Fields:
# 0
# stop_id
# Example: N3_R1_0
# note: self.stop is a section[0:num_stops] of self.stops_exp
# --------------
# self.routes: (list)
# Fields:
# 0
# route_id
#
# self.routes_directional: (list)
# Example: 
# --------------
# self.links: (table, list, elmt being list)
# Fields:
# 0          1        2             3           4         5
# link_id,route_id,starting_stop,ending_stop,link_type,travel_time,
# 6        7        8          
# frequency,capacity,travel_cost
# Example: ['L1_0', 'R1_0', 'N1', 'N3', 'Bus', 5, 1000000, 200, 5]
#
# self.links_directional: (table, list, elmt being list)
# Example: ['L1_0', 'R1_0', 'N1', 'N3', 'Bus', 5, 1000000, 200, 5]
#
# self.links_exp
# Example: ['L1_0', 'R1_0', 'N1_R1_0', 'N3_R1_0', 'Bus', 5, 1000000, 200, 5]
# --------------
# self.frequencies: (table, list, elmt being list)
# Fields:
# 0
# headway_secs

from copy import deepcopy

from open_data_file_with_header import open_data_file_with_header
from open_config_file import open_config_file

class StaticNetwork():
	"""Input physical stops, routes and links data to build a static network object."""

	def __init__(self):

		print('    ------------------------------------------------------------------------------')
		print("    Creating a static network object ...")

		# Config
		# Small constant
		SMALL_CONST = float(open_config_file('SMALL_CONST'))
		LARGE_CONST = open_config_file('LARGE_CONST')
		TRANSIT_TYPE = open_config_file('TRANSIT_TYPE')
		TEST_SN = open_config_file('test_StaticNetwork')
		
		# Read files from '/data', remove header, decompose elmts
		self.stops = open_data_file_with_header('data/stops.csv')
		# 0
		# stop_id
		self.stops = [item[0] for item in self.stops]

		self.routes = open_data_file_with_header('data/routes.csv') 
		# 0       
		# route_id
		# Will be made directional later.
		self.routes = [item[0] for item in self.routes]

		self.links = open_data_file_with_header('data/links.csv')
		# 0      1        2             3           4         5             
		#link_id,route_id,starting_stop,ending_stop,link_type,travel_time,
		# 6        7        8 
		#frequency,capacity,travel_cost
		# Note: the links are bi-directional, they have to be decomposed into two
		# uni-directional links later.
		# link type in raw file left empty.

		self.frequencies = open_data_file_with_header('data/frequencies.csv')
		# 1
		# headway_secs
		# Note: frequrncies' corrusponding routes must be the same with self.routes file.
		self.frequencies = [item[0] for item in self.frequencies]

		self.routes_directional = []
		# Decompose routes into directional routes.
		for item in self.routes:
			# Add directions
			# Elmts being string
			self.routes_directional.append(item + '_0')
			self.routes_directional.append(item + '_1')
		
		# Decompose the bi-directional links into two uni-directional links.
		# Add direction to route_id.
		veh_cap_of_routes = open_data_file_with_header('data/vehicle_capacity_of_routes.csv')
		veh_cap_of_routes = [item[0] for item in veh_cap_of_routes]
		# 0 
		# veh_capacity
		self.links_directional = []
		for item in self.links:

			capacity = int(veh_cap_of_routes[self.routes.index(item[1])])

			# Add links for one direction.
			self.links_directional.append([item[0] + '_0'] + [item[1] + '_0'] + item[2:4] + \
				[TRANSIT_TYPE, int(item[5]), LARGE_CONST,capacity,int(item[5])])
			
			# Swap the starting stop and ending stop; add links for another direction
			temp = item[2]
			item[2] = item[3]
			item[3] = temp
			# Add direction to route_id
			item[1] += '_1'
			item[0] += '_1'
			# Note that transit links have infinite frequrncies; it's the WT links that have finite frequencies.
			self.links_directional.append(item[0:4] + [TRANSIT_TYPE, int(item[5]), LARGE_CONST,capacity,int(item[5])])

		self.links_exp = deepcopy(self.links_directional)

		self.stops_exp = deepcopy(self.stops)

		# Add new nodes and links at possible transfer points.
		# Steps:
		# 1) For a given directional route, find the set of links that are on 
		# this directional route;
		# 2) Find the starting stop of this directional route; then sort the set in 1);
		# 3) Add additional links at trasfer points, depending on:
		# 3.1) 

		# Used to formulate the link_id for new links.
		additional_link_count = 1
		
		for route in self.routes_directional:

			# 1) Find those links that are on current route.
			links_on_route = []
			for i in range(len(self.links_exp)):
				if self.links_exp[i][1] == route:
					# Modify the node name of links on routes.
					self.links_exp[i][2] = self.links_exp[i][2] + '_' + route
					self.links_exp[i][3] = self.links_exp[i][3] + '_' + route
					# Add to the links_on_route list.
					links_on_route.append(self.links_exp[i])

			# 2) Find starting stop of this route and sort link set.
			# Find starting stop
			# Initialize
			start_stop_of_route = links_on_route[0][2]
			flag = 0
			while flag == 0:
				for count in range(len(links_on_route)):
					# If current start_stop_of_route appears to be ending stop of 
					# some other links ...
					if links_on_route[count][3] == start_stop_of_route:
						start_stop_of_route = links_on_route[count][2]
						break
					# If this links never happens to be an ending stop ...
					if count == len(links_on_route) - 1:
						flag = 1

			# Sort links on current route.
			links_on_route_sorted = []
			current_node = start_stop_of_route
			iteration_count = 1
			while links_on_route:
				for i in range(len(links_on_route)):
					if links_on_route[i][2] == current_node:
						links_on_route_sorted.append(links_on_route[i])
						current_node = links_on_route[i][3]
						del links_on_route[i]
						break
				iteration_count += 1
				if iteration_count == 100000:
					print('    Links Data Error! Some links are missing!')
					print('    Traceback: StaticNetwork')
					links_on_route = []

			# 3) Add links and nodes for current route

			# Note: the additiinal links' cost is set at SMALL_CONST
			# in order to avoid tie on distance! 
			# Tie happens in step 2 of Spiess' algorithm, when 
			# sort links in desending order according to {cij + uj}.
			temp_len = len(links_on_route_sorted)
			for i in range(temp_len):

				temp_freq = 3600 / float(self.frequencies[self.routes.index(route[0:-2])])

				# For each link i, extract the information of the starting nodes:
				#  - the original name (without direction) of the starting stop (original_starting_node);
				#  - the new name of starting stop of this link (directional_starting_node).
				route_name_len = len(route) + 1
				original_starting_node = links_on_route_sorted[i][2][0:-route_name_len]
				directional_starting_node = links_on_route_sorted[i][2]
				# Add directional node.
				self.stops_exp.append(directional_starting_node)

				# If this link is the last link, information concerning the ending nodes
				# are also extracted for later use:
				#  - the original name (without direction) of the ending stop (original_ending_node);
				#  - the new name of ending stop of this link (directional_ending_node).
				if i == (temp_len - 1):
					original_ending_node = links_on_route_sorted[i][3][0:-route_name_len]
					directional_ending_node = links_on_route_sorted[i][3]
					# Add directional node.
					self.stops_exp.append(directional_ending_node)

				# Remember: WT ans WK links should have > 0 small cost!
				# For the first node on route ..
				if i == 0:
					# Add WT link
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						original_starting_node,directional_starting_node,'WT',SMALL_CONST,temp_freq,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1

				# For intermediate node on route
				elif i >= 1 and i <= (temp_len - 2):
					# Add WT and WK links
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						original_starting_node,directional_starting_node,'WT',SMALL_CONST,temp_freq,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						directional_starting_node,original_starting_node,'WK',SMALL_CONST,LARGE_CONST,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1

				# For the second last and the last node on route
				elif i == temp_len - 1:
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						original_starting_node,directional_starting_node,'WT',SMALL_CONST,temp_freq,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						directional_starting_node,original_starting_node,'WK',SMALL_CONST,LARGE_CONST,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1
					# handle last node on the route
					self.links_exp.append(['LA' + str(additional_link_count),route,\
						directional_ending_node,original_ending_node,'WK',SMALL_CONST,LARGE_CONST,LARGE_CONST,SMALL_CONST])
					additional_link_count += 1

		# print network information
		if TEST_SN == 1:
			print('    routes_directional:(length: ' + str(len(self.routes_directional)) + ')')
			print(self.routes_directional)
			print()
			print('    stops: (length: ' + str(len(self.stops)) + ')')
			print(self.stops)
			print()
			print('    stops_exp:(length: ' + str(len(self.stops_exp)) + ')')
			print(self.stops_exp)
			print()
			print('    frequencies:')
			print(self.frequencies)
			print()
			print('    links_directional:(length: ' + str(len(self.links_directional)) + ')')
			print(self.links_directional)
			print()
			print('    links_exp:(length: ' + str(len(self.links_exp)) + ')')
			print(self.links_exp)
			
		print("    A static network has been successfully created!")