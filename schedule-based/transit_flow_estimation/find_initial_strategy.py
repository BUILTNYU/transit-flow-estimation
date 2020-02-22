# Generating a initial strategy

# 1) For each destination r, find the shortst path on TE network based on 
# no-congestion cost;
# 2) For each (q_h,r) pair, the initial strategy is defined to be:
# At each i_t, the preference set inlude all downstream nodes whose
# disntance-to-sink < Infinity.
# Arrange them according to label to destinations.

import numpy as np
from copy import deepcopy

from Graph import Graph
from open_config_file import open_config_file

def find_initial_strategy(tn):
	""" Input the time-dependent network; output an initil strategy."""

	print("    --------------------------")
	print("    Finding initial strategy begins ...")

	DISPLAY_PROGRESS = int(open_config_file('DISPLAY_PROGRESS'))
	TEST = int(open_config_file('test_find_initial_strategy'))

	num_stops = len(tn.stops)
	num_stops_exp = len(tn.stops_exp)
	num_choices = len(tn.routes) + 1
	T = tn.T

	prefer_links_optimal = {}
	prefer_probs_optimal = {}

	temp_key_set = set()
	for r in range(num_stops):
		for l in range(num_choices):
			for i_t in range(num_stops_exp):
			
				t = tn.stops_exp_2[i_t][2]
				temp = 0
				if l != (num_choices - 1):
					temp = t

				for tau in range(temp, t + 1):
					temp_key_set.add((r,l,tau,i_t))
	
	prefer_links_optimal = {key: [] for key in temp_key_set}
	prefer_probs_optimal = {key: [] for key in temp_key_set}

	# Range over destination.
	for r in range(num_stops):

		dest_name = tn.stops[r]

		if DISPLAY_PROGRESS == 1:
			print("    *   *   *")
			print("    Current destination: " + dest_name + " - " + str(r))

		# Make a cpoy of tn.stops_exp and tn.links_exp respectively,
		# since they will be modified later.
		links_exp_temp = deepcopy(tn.links_exp)

		# Add dummy node for this destination.
		# Sink node's index in new node list is num_stops_exp.
		sink_index = num_stops_exp

		# Add dummy links connecting r_t to r
		# link_count used to generate link_index. 
		link_count = len(links_exp_temp)
		for t in range(T):
			starting_stop_index = tn.stops_exp.index(dest_name + '_' + str(t))
			links_exp_temp.append([link_count,'',starting_stop_index,\
				sink_index,'Dummy',0,'',float('inf'),0])
			link_count += 1

		# Create Graph obj
		a_graph = Graph(num_stops_exp + 1, links_exp_temp, tn.stops_exp_2)

		# Use Shortest path tree method
		# Find the label labels of all nodes
		label, sequence = a_graph.dijkstra_travel_cost(sink_index)
		# label could be 1) arrival_time; 2) travel_cost.
		# The travel_cost is recommended, since travel cost may be different from
		# travel time.

		if TEST == 1:
			print("    Distance of nodes in TE network to destination:")
			print(label)
			print("    sequence is:")
			print(sequence)

		# Given the destination r, find the preference sets for all node i_t
		for i_t in range(num_stops_exp):

			# Index of phisical node in tn.stop
			i = tn.stops_exp_2[i_t][1]
			# Time of this node
			t = tn.stops_exp_2[i_t][2]

			if TEST == 1:
				print("    Destination: " + dest_name + " - " + str(r) + " Node: " + tn.stops_exp[i_t] + " - " + str(tn.stops_exp_2[i_t]))
				print("    Distance:" + str(label[i_t]))
			
			# If origin = destination, left empty
			if tn.stops_exp_2[i_t][1] == r:
				continue

			# Here tn.links_exp is used to aviod dummy links added before.
			outgoing_links = [item for item in tn.links_exp if item[2] == i_t]
			outgoing_links = deepcopy(outgoing_links)

			# Add the label to sink node to the end
			for item in outgoing_links:
				item.append(label[item[3]])

			# Sort according to the label to r (ascending order)
			outgoing_links = sorted(outgoing_links, key = lambda x: x[9])
			# Delete these links that have infinite label from sink
			outgoing_links = [item for item in outgoing_links if item[9] != float('inf')]

			# If this set is not empty ... ; otherwise, this node doesn't connect to r;
			# Then it's preference set left empty.
			if outgoing_links:
				outgoing_links_index = [item[0] for item in outgoing_links]
				outgoing_routes_index = [item[1] for item in outgoing_links]
				outgoing_links_cost = [item[8] for item in outgoing_links]
				downstream_nodes = [item[3] for item in outgoing_links]

				# Prioritize WT route in case that its label equals some bus routes.
				if (num_choices - 1) in outgoing_routes_index:
					wt_index = outgoing_routes_index.index(num_choices - 1)

					# Find the most preferred route that arrival == WT route;
					# Note that it may equlas itself.
					temp_index_2 = 99999
					for temp_index in range(len(outgoing_routes_index)):
						if (outgoing_links[temp_index][9] + outgoing_links_cost[temp_index])\
						 == (outgoing_links[wt_index][9] + outgoing_links_cost[wt_index]):
							temp_index_2 = temp_index
							break
					# If it's not itself, then switch order.
					if temp_index_2 != wt_index:
						temp = deepcopy(outgoing_links[temp_index_2])
						outgoing_links[temp_index_2] = deepcopy(outgoing_links[wt_index])
						outgoing_links[wt_index] = deepcopy(temp)

						# Update new indices.
						outgoing_links_index = [item[0] for item in outgoing_links]
						outgoing_routes_index = [item[1] for item in outgoing_links]
						downstream_nodes = [item[3] for item in outgoing_links]

				if TEST == 1:
					print("    outgoing_links:")
					print(outgoing_links)

				# Incoming routes l need to be find in order to construct prefer sets.
				# Here tn.links_exp is used to aviod dummy links added before.
				incoming_links = [item for item in tn.links_exp if item[2] == i_t]
				incoming_links = deepcopy(incoming_links)

				# Whether or not there is a WT link leading to node i_t,
				# set the WT arrival has preferece set ...
				# since there are departuring demand and they will come from
				# "WT route" in this algorithm.
				if incoming_links:
					incoming_routes_index = [item[1] for item in incoming_links]

					if TEST == 1:
						print("    incoming_routes_index:")
						print(incoming_routes_index)

					if (num_choices - 1) not in  incoming_routes_index:
						# Note: "WT route" has index (num_choices - 1).
						incoming_routes_index.append(num_choices - 1)

					for l in incoming_routes_index:
						for temp_index in range(len(downstream_nodes)):

							if l != (num_choices - 1):
									prefer_links_optimal[r,l,t,i_t].append(outgoing_links_index[temp_index])	
									prefer_probs_optimal[r,l,t,i_t].append(0.0)					
							else:
								for tau in range(t + 1):
									prefer_links_optimal[r,l,tau,i_t].append(outgoing_links_index[temp_index])
									prefer_probs_optimal[r,l,tau,i_t].append(0.0)
				else:
					l = num_choices - 1

					# In this case, you only have one incoming route - WT.
					for temp_index in range(len(downstream_nodes)):
						for tau in range(t + 1):
							prefer_links_optimal[r,l,tau,i_t].append(outgoing_links_index[temp_index])
							prefer_probs_optimal[r,l,tau,i_t].append(0.0)

			if TEST == 1:
				print("    The initial preference set:")
				for l in range(num_choices):
					for tau in range(t + 1):
						if l == (num_choices - 1) or (l != (num_choices - 1) and tau == t):
							print("    route(l): " + str(l) + " tau: " + str(tau)  + " prefer_links_optimal:")
							print(prefer_links_optimal[r,l,tau,i_t])
							print("    route(l): " + str(l) + " tau: " + str(tau)  + " prefer_probs_optimal:")
							print(prefer_probs_optimal[r,l,tau,i_t])

	#print(prefer_routes_optimal[1,2,540])

	print("    A initial strategy has been found!")

	return prefer_links_optimal, prefer_probs_optimal