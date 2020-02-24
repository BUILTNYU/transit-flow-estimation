# Models from Nguyen, Pallottino (1988), Spiess and FLorian (1989)

# Notations
# (inherated from Spiess (1989) paper)
# r: index for origins
# s: index for destinations
# a: index for links
# S_set: set of links not yet examined
# A_set: optimal strategy - set of selected links
# f_freq: combined frequency at nodes
# u_label: distance labels
# v_stop: od prob/flow that appears at a particular stop
# v_link: od prob/flow that appears at a particular link
# --------------
# od_flow_to_stop_prob[r,s,i]: return of this function
# (Numpy, size(num_stops * num_stops * num_stops_exp)
#
# Note: three types of flow activities' probabilities are recorded:
# 1) departuring;
# 2) transferring;
# 3) exiting.
# That's because, say node N9's passing by flows, will not corss node N9;
# They will only appear at nodes like N9_R1_1; they will not take WK link
# to node N9.
#
# od_flow_to_stop_exp_prob
# --------------
# od_flow_to_link_directional[r,s,a]
# (numpy, size: num_stops * num_stops * num_links_directional)
#
# od_flow_to_link_exp[r,s,a]
# 
# link_flow[a]
# (numpy array, len: num_links_directional)
# 
# strategy_flow[q,r,s]
# strategy_flow_to_link_exp[q,r,s,a]
# strategy_flow_to_stop_prob[q,r,s,i]

import numpy as np
from math import sqrt
from copy import deepcopy

from open_data_file_with_header import open_data_file_with_header
from open_config_file import open_config_file
from congestion_penalty_coeff import congestion_penalty_coeff

def static_assignment_algorithm(sn,od_flow):
	
	# Inputs
	# sn: statict metwork obj;
	# od_flow[r,s].

	print('    ------------------------------------------------------------------------------')
	print("    Static assignment algorithm begins ...")

	# Parameters
	TEST_STATIC_ASSIGN = int(open_config_file('test_static_assignment_algorithm'))
	MAX_NUMBER_OF_ITERATIONS = int(open_config_file('MAX_NUMBER_OF_ITERATIONS_FOR_ASSIGNMENT_MODEL'))
	CONVERGENCE_CRITERION = float(open_config_file('CONVERGENCE_CRITERION_FOR_ASSIGNMENT_MODEL'))
	TRANSIT_TYPE = open_config_file('TRANSIT_TYPE')
	LARGE_CONST = open_config_file('LARGE_CONST')

	# WAITING_TIME_COEFF is used to determine the waiting time given the frequency.
	WAITING_TIME_COEFF = float(open_config_file('WAITING_TIME_COEFF'))

	num_stops = len(sn.stops)
	num_stops_exp = len(sn.stops_exp)
	num_ods = num_stops ** 2
	num_links_directional = len(sn.links_directional)
	num_links_exp = len(sn.links_exp)
	num_strategies = int(open_config_file('MAX_NUMBER_OF_STRATEGIES_STATIC_MODEL'))

	print()
	print("    num_stops: " + str(num_stops))
	print("    num_links_exp: " + str(num_links_exp))
	print("    num_strategies: " + str(num_strategies))
	print()

	# Used for finding link index later.
	links_id_list = [item[0] for item in sn.links_exp]

	# Initialize strategy flow.
	strategy_flow = np.zeros((num_stops, num_stops, num_strategies))
	for q in range(num_stops):
		for r in range(num_stops):
			strategy_flow[q,r,0] = od_flow[q,r]

	# [q,r,s,a]
	strategy_flow_to_link_exp = np.zeros((num_stops, num_stops, num_strategies, num_links_exp))
	# [q,r,s,i]
	strategy_flow_to_stop_prob = np.zeros((num_stops, num_stops, num_strategies, num_stops))

	# Frank-Wolfe method iterations
	# Goal: for fixed od_flow, find the UE flow assignment.
	iteration_count = 0
	converg_flag = 0
	while converg_flag == 0:
		print()
		print('    Now is the '+ str(iteration_count) + '-th iterations...')
		print("    Total flow: " + str(strategy_flow.sum()))

		# Initialized for each iteration.
		od_flow_to_stop_exp_prob = np.zeros((num_stops, num_stops, num_stops_exp))
		od_flow_to_stop_prob = np.zeros((num_stops, num_stops, num_stops))
		od_flow_to_link_exp = np.zeros((num_stops, num_stops, num_links_exp))
		od_flow_to_link_directional = np.zeros((num_stops, num_stops, num_links_directional))
		link_flow = np.zeros(num_links_directional)

		# Update TC depeding on flow from last iteration.
		if iteration_count >= 1:

			# Note: it's sn.links_exp cost being modfied, not sn.links_directional.
			for a in range(num_links_exp):
				if sn.links_exp[a][4] == TRANSIT_TYPE:
					sn.links_exp[a][8] = congestion_penalty_coeff(link_flow[a],sn.links_exp[a][7], TRANSIT_TYPE)\
					 * sn.links_exp[a][5]
					#cost      =      congestion penalty coefficient   *   TT

		# Spiess' algorithm
		# r: origin index (numeric)
		# s: destination index (numeric)
		# a: link index (numeric)
		# u_label: node labels
		# S_set: set of links not examined
		# A_set: set of links in optimal strategy

		# Assign flow from all origins (excluding current destination) to
		# a destination - s - at a time.
		# Note: only stops in sn.stops could be destinations and origins.
		for r_dest in range(num_stops):

			dest_name = sn.stops[r_dest]
			print('    (Step 1 - backward traverse) Current destination: ' + dest_name + " - " + str(r_dest))

			# Step 1 - reverse traverse to obtain optimal strategy
			print()
			print("    Iteration " + str(iteration_count) +  " Step 1 - backward traverse to obtain optimal strategy begins ...")
			
			# Initialization
			# "u"" stores the distance labels
			u_label = LARGE_CONST * np.ones(num_stops_exp)
			u_label[r_dest] = 0

			# "f" stores the combined frequency of nodes
			f_freq = np.zeros(num_stops_exp)

			# "S_set" stores links not examined;
			# Make a copy from sn.link_exp.
			S_set = deepcopy(sn.links_exp)

			# "A_set" stores the optimal strategy.
			A_set = []

			# While S not empty, do
			while S_set:

				# Initialize
				min_cost_link_index = 0
				min_cost = LARGE_CONST

				# Find the least (cij + uj) link
				for a in range(len(S_set)):
					# Get the distance label of the ending stop.
					end_stop_index = sn.stops_exp.index(S_set[a][3])
					end_stop_label = u_label[end_stop_index]

					if S_set[a][8] + end_stop_label < min_cost:
						# Use cost of link_exp.
						min_cost = S_set[a][8] + end_stop_label
						min_cost_link_index = a

				if TEST_STATIC_ASSIGN == 1:
					print()
					print('    min cost link:')
					print(S_set[min_cost_link_index])

				end_stop_index = sn.stops_exp.index(S_set[min_cost_link_index][3])
				end_stop_label = u_label[end_stop_index]

				if TEST_STATIC_ASSIGN == 1:
					print('    link_travel_cost: ' + str(S_set[min_cost_link_index][8]) + ' end_stop_distance_label is:' + str(end_stop_label))

				# Update label u and combined frequency if needed.
				# "start_stop" means the upstream stop of the link.
				start_stop = S_set[min_cost_link_index][2]
				start_stop_index = sn.stops_exp.index(start_stop)

				# Flag for updating type:
				# 0: no update;
				# 1: first time update;
				# 2: update, not for the first time
				update_flag = []
				if u_label[start_stop_index] <= min_cost:
					update_flag = 0

					if TEST_STATIC_ASSIGN == 1:
						print('    no update.')

				elif u_label[start_stop_index] > min_cost:
					if u_label[start_stop_index] >= LARGE_CONST:
						update_flag = 1
						# If this link has inf freq.
						if S_set[min_cost_link_index][6] == LARGE_CONST:
							u_label[start_stop_index] = min_cost
						else:
							# Note that the time unit is min;
							# Note the WAITING_TIME_COEFF.
							u_label[start_stop_index] = 60/S_set[min_cost_link_index][6] + min_cost
						f_freq[start_stop_index] = S_set[min_cost_link_index][6]

					else:
						update_flag = 2
						u_label[start_stop_index] = (( f_freq[start_stop_index] * u_label[start_stop_index]
							+ S_set[min_cost_link_index][6] * min_cost) / 
							(f_freq[start_stop_index] + S_set[min_cost_link_index][6]))
						f_freq[start_stop_index] += S_set[min_cost_link_index][6]

					if TEST_STATIC_ASSIGN == 1:
						print('    Updated stop is: ' + sn.stops_exp[start_stop_index] + ' - ' + str(start_stop_index))
						print('    Distance label:' + str(u_label[start_stop_index]) + ' combined frequency: ' + str(f_freq[start_stop_index]))

					# Add this link to attractive set
					A_set.append(S_set[min_cost_link_index])

					if TEST_STATIC_ASSIGN == 1:
						print('    This link is added to optimal strategy set.')

				# Remove examined link from S
				del S_set[min_cost_link_index]
			
			A_set_links_id = [item[0] for item in A_set]

			if TEST_STATIC_ASSIGN == 1:
				print()
				print('    All links being examined.')
				print('    The node distance labels (u) and frequencies (f) are:')
				for temp_index in range(num_stops_exp):
					print("    Node name: " + sn.stops_exp[temp_index])
					print("    Distance label: " + str(u_label[temp_index]))
					print("    Combined frequencies (f): " + str(f_freq[temp_index]))
				print()
				print('    The optimal strategy (A) for destination ' + sn.stops[r_dest] + ' is:')
				print(A_set)

			# Step 2: forward traverse to assigne flow to optimal strategy
			# Note: dest s is still fixed.
			print()
			print("    Iteration " + str(iteration_count) +  " Step 2 - forward traverse to assigne flow to optimal strategy begins.")
			print('    (Step 2: forward traverse) Current destination: ' + sn.stops[r_dest])

			# First obtain {cij + uj} for all links
			# Used for sorting;
			# Now 'links_sorted' is unsorted yet.
			links_sorted = deepcopy(sn.links_exp)
			for a in range(num_links_exp):
				end_stop_index = sn.stops_exp.index(links_sorted[a][3])
				end_stop_label = u_label[end_stop_index]
				# Add a col 9;
				# Note: don't modify col 8, cost;
				# Col 8 is for cost with congestion effect
				links_sorted[a].append(float(links_sorted[a][8] + end_stop_label))

			# Then sort links according to {cij + uj} in decreasing order
			links_sorted = sorted(links_sorted, key=lambda x: x[9], reverse = True)
			print('    links sorted according to {cij + uj}:')
			print(links_sorted)
			
			# Iterate over all origins so that prob for each od pair could be obtained.
			# Note: dest r_dest is still fixed.
			for q_origin in range(num_stops):
				# v_stop is used to store the flow that appears at each node;
				# v_link  is used to store the flow that appears at each link;
				v_stop = np.zeros((num_stops_exp))
				v_link = np.zeros((num_links_exp))

				# Assign od flow from q_origin to r_dest
				# Unit flow is first assigned in order to obtain assign probability.
				v_stop[q_origin] = 1.0

				for link in links_sorted:
					start_stop = link[2]
					start_stop_index = sn.stops_exp.index(start_stop)
					end_stop = link[3]
					end_stop_index = sn.stops_exp.index(end_stop)
					link_index = links_id_list.index(link[0])

					if link[0] in A_set_links_id:
					# Note: 
					# 1) there is an additional colume in links_sorted, hence "-1"
					# 2) link index in sn.links_exp is needed since od_flow_to_link_directional_prob_exp is arranged 
					#    according to that sequence;
					# 3) this sorted seq is temporary -- for each q_origin-r_dest, the seq
					#    may be different;
						v_link[link_index] += (v_stop[start_stop_index] * link[6]
							/ f_freq[start_stop_index])
						v_stop[end_stop_index] += v_link[link_index]
					else:
						v_link[link_index] = 0

				od_flow_to_stop_exp_prob[q_origin,r_dest,:] = v_stop
				
				# Extract the first num_stops elmts
				# Multiply the r-s flow
				v_link = od_flow[q_origin,r_dest] * v_link
				od_flow_to_link_exp[q_origin,r_dest,:] = v_link

				print()
				print('    Current destination: ' + dest_name + ' - ' + str(r_dest) + ' current origin: ' + sn.stops[q_origin] + ' - ' + str(q_origin))
				if TEST_STATIC_ASSIGN == 1:
					for temp_index in range(num_stops):
						print("    Assignment probability of flow "  + " to stops: " + sn.stops[temp_index] + " v_stop:" + str(v_stop[temp_index]))

				print("    Assignment flow of flow from " + dest_name + " to links (v_link):")
				print(v_link)

		print()
		print('    All origins and destinations visited once.')

		# Extract non "_exp" info.
		for q_origin in range(num_stops):
			for r_dest in range(num_stops):

				od_flow_to_link_directional[q_origin,r_dest,:] = od_flow_to_link_exp[r,r_dest,0 : num_links_directional]

				# Note: if od_flow_to_stop_prob[r,s,:] = od_flow_to_stop_exp_prob[r,s,0 : num_stops]
				# is used, then only transfer flow will be recorded;
				# Only stops from num_stops to num_stops_exp will be counted; namely nodes like N2_R2_1;
				for index in range(num_stops, num_stops_exp):

					temp = sn.stops_exp[index].split('_')[0]
					i = sn.stops.index(temp)

					od_flow_to_stop_prob[q_origin,r_dest,i] += od_flow_to_stop_exp_prob[q_origin,r_dest,index]

					if TEST_STATIC_ASSIGN == 1 and od_flow_to_stop_exp_prob[q_origin,r_dest,index] > 0.000001:
						print()
						print("    Step 1) od_flow_to_stop_prob" + str([q_origin,r_dest,i]) + " updated to: " + str(od_flow_to_stop_prob[q_origin,r_dest,i]))

				# Delete repetition counts.
				# Note that flow that transfer at node N1 from R1 to R2 will be counted 
				# at N1_R1_0, N1, and N1_R2_0;
				for index in range(num_stops):
					if index != q_origin and index != r_dest:
						od_flow_to_stop_prob[q_origin,r_dest,index] -= od_flow_to_stop_exp_prob[q_origin,r_dest,index]

						if TEST_STATIC_ASSIGN == 1 and od_flow_to_stop_exp_prob[q_origin,r_dest,index] > 0.000001:
							print()
							print("    Step 2) od_flow_to_stop_prob" + str([q_origin,r_dest,index]) + " updated to: " + str(od_flow_to_stop_prob[q_origin,r_dest,index]))

		# Save results.
		np.save('results/static/od_flow_to_stop_prob_iteration_' + str(iteration_count), od_flow_to_stop_prob)
		np.save('results/static/od_flow_to_link_directional_iteration_' + str(iteration_count), od_flow_to_link_directional)

		if iteration_count >= MAX_NUMBER_OF_ITERATIONS:
			print('    Warning! Diagnolization method not converging at ' + str(MAX_NUMBER_OF_ITERATIONS) + 'th iteration!')
			converg_flag = 1

		# Use MSA to update the strategy flow.
		# Note that probabilities don't need MSA; flow need.
		for q_origin in range(num_stops):
			for r_dest in range(num_stops):
				for i in range(num_stops):
					strategy_flow_to_stop_prob[q_origin,r_dest,iteration_count,i] = od_flow_to_stop_prob[q_origin,r_dest,i]
		# Direction direction strategy flow.
		Y_strategy = np.zeros((num_stops, num_stops, num_strategies))
		for q in range(num_stops):
			for r in range(num_stops):
				Y_strategy[q,r,iteration_count] = od_flow[q,r]
		# Direction direction link flow.
		Y_link_exp = np.zeros((num_stops, num_stops, num_strategies, num_links_exp))
		for q_origin in range(num_stops):
			for r_dest in range(num_stops):
				for a in range(num_links_exp):
					Y_link_exp[q_origin,r_dest,iteration_count,a] = od_flow_to_link_exp[q_origin,r_dest,a]

		if iteration_count == 0:
			strategy_flow_to_link_exp = deepcopy(Y_link_exp)
		else:
			strategy_flow = 1/(iteration_count + 1) * (iteration_count * strategy_flow + Y_strategy)
			strategy_flow_to_link_exp = 1/(iteration_count + 1) * (iteration_count * strategy_flow_to_link_exp + Y_link_exp)

		# Calculate link_flow.
		for a in range(num_links_directional):
			link_flow[a] = strategy_flow_to_link_exp[:,:,:,a].sum()

		# Print results.
		if TEST_STATIC_ASSIGN == 1:
			print()
			print("    link_flow:")
			for temp_index in range(num_links_directional):
				if link_flow[temp_index] > 0.0001:
					print("    Link: " + str(sn.links_directional[temp_index]))
					print("    Link flow: " + str(link_flow[temp_index]))

		# Compute return
		for q_origin in range(num_stops):
			for r_dest in range(num_stops):

				if od_flow[r,r_dest] > 0.0001:
					for i_stop in range(num_stops):
						temp = 0
						for s_strategy in range(num_strategies):
							temp += strategy_flow[q_origin,r_dest,s_strategy] * strategy_flow_to_stop_prob[q_origin,r_dest,s_strategy,i_stop]

						od_flow_to_stop_prob[q_origin,r_dest,i_stop] = temp / od_flow[q_origin,r_dest]

		# Convergence test
		if iteration_count >= 2:

			avg_abs_relative_change = 0
			
			for a in range(num_links_directional):
				if (link_flow[a] + link_flow_last[a]) != 0:
					avg_abs_relative_change += abs(link_flow[a] - link_flow_last[a]) / ((link_flow[a] + link_flow_last[a]) / 2)

			avg_abs_relative_change /= num_links_directional
			print("    (Lower levle) iteration: " + str(iteration_count) + " avg_abs_relative_change: " + str(avg_abs_relative_change))

			if avg_abs_relative_change < CONVERGENCE_CRITERION:
				converg_flag = 1

				print("    Lower level CONVERGENCE_CRITERION met!")
			else:
				print("    CONVERGENCE_CRITERION not met; next iteration begins...")
		
		link_flow_last = deepcopy(link_flow)

		iteration_count += 1

	return od_flow_to_stop_prob
