# Algorithm in Hamdouch, Lawphongpanich (2008) will be adopted;
# Modified by Qi.

# Notes: 
# 1) There is no initial flow on the network at time = 0; 
# 2) how to handle a node that has no preference set, which 
#    means that it cannot reach the destination in analysis horizon?
#    That means we have no good idea where these flow will go；
#    If they are ignored, the flow counts won't match the measurements.
#    But here they will simply ignored here for simplicity. There are definitely 
#    better methods. There is space for improvemrnt.
#    Because of 1) and 2), this model is suitable for estinating flows 
#    during the middle part of the horizon.

# 3) departure time choice dimension will be discarded;
# 4) State for dynamicn proramming will be modified to (i_t,l,τ)
#    where, i_t: node index in TE network nodes
#        l: upstream bus line index
#		 τ: arrival time at node i_t
# 5) "Waiting route" or "departing route" will take route index num_routes, which 
#    equals (num_choices - 1); 

# Notations (inherited from hamdouch (2008) paper)
# q: index for origin
# r: index for destination
# h: index for departure time
# i_t: index for nodes in TE network
# a: index for arcs
# t: index for current time
# T: horizon
# s: index for strategy
# num_strategies: maximum number of strategies
# tau: index for arrival time
# num_choices: the total number of routes would be len(tn.routes) + 1 := num_choices.
# num_stages: total number of states
# num_stops: number of actual stops
# l: index for bus line
# num_routes: number of bus lines
# num_links_exp: numer of arcs
#
# --------------
# prefer_links[r,s,l,τ,i_t]: prefrence set (dict, keys are 5-dim tuple, item are lists)
# The elmts are link idnex of arc connecting i_t to downstream nodes.
#
# prefer_links_optimal (E^[r,l,τ,i_t]): no "s", strategy index.
#
# prefer_stops_temp[r,s,l,τ]
# prefer_links_temp[r,s,l,τ]
# prefer_routes_temp[r,s,l,τ]
# Suffix "_temp" is used to remind of the fact that it's changed in each stage.
#
# prefer_probs[r,s,l,τ,i_t]: the probability of bording outgoing links.
# --------------
# strategy_flow[q,r,h,s] (X^[q,r,h,s] in paper): strategy flow (4-dim numpy array)
# Note: default value float('inf') means no records; zero is not used instead since it
# would be misleading.
# --------------
# node_flow[q,r,h,s,l,τ,i_t] (Z^[q,r,h,s,l,τ,i_t]in paper): node arrival flow
# (7-dim numpy array)
#
# node_prob (β^[q,r,h,s,l,τ,i_t] in paper): node arrival probability
# (7-dim numpy array)
#
# node_flow[r,s,l,τ,i_t]: no q,h info required;
# (5-dim numpy array)
# [r,s,l,τ,i_t] is still needed for identifying the preference sets;
#
# node_flow has been used twice:
# first in loading flow for i_t: node_flow[r,s,l,tau] - i_t is neglected;
# second in computing return: node_flow[s,l,tau,i_t]
#
# Note: Arrival time τ at node i_t is needed for virtual flow loading.
# --------------
# *link_flow[q,r,h,s,l,τ,link_index]: startegy-to-arc flow (this is not from the paper)
# (6-dim numpy array) Dafault value is zero.
# τ is used to record since when users has been waiting for this link at up stream node.

# Note: Arrival time τ at link in fact is just needed for waiting links, used to record
# since when users has been waiting for this link.
#
# link_flow[r,s,l,τ,link_index]: no q,h info stored;
#
# link_combined_flow, flow[link_index]: link flow from all strategies 
# (numpy array, shape: num_stops_exp * 1)
# Used to calculate link congestion cost.
# Return of thid function.
# --------------
# phi_cost: expected cost, used in finding optimal preference set for i_t
# len_outgoing_links * 2 numpy matrix;
# (cols) 0 index   1 cost
#
# Note: phi will be sorted according to cost;
# This strange form can help to retrain previous information.
# --------------
# w_cost (w^[l,τ,i_t]): expexted cost; optimal value function, used in finding optimal preference set for i_t
# (4-dim numpy array)
#
# w_cost_1 and w_cost_2: scaler
# --------------
# virtual_flow_prob: vector, arc acess probability for virtual flow 
# --------------
# od_flow_to_stop_prob[q,r,h,i,t]: return of this function
# Numpy matrix; elemts being od_flow_to_stop_prob[q,r,h,i,t].
# Note: three types of flow activities' probabilities are recorded:
# 1) departuring;
# 2) transferring;
# 3) pass-by on train;
# 3) exiting.
# In fact, I'd like to call od_flow_to_stop_prob the "detedcted" prob.
# --------------
# Y[q,r,h,s]: the dscent direction, format being the same with strategy_flow.
# C_X: scaler, the cost of current strategy flows.

# Memory requirments
# First, transportation network is generally sparse;
# num_links_exp ~ num_stops_exp = num_stops * T
# Let "N" ne num_stops for simplicity.
# 
# prefer_links[q,r,h,s,l,τ,link_index] and prefer_routes/stops are 5 dimsinal; 
# node_flow[q,r,h,s,l,τ,i_t] and link_flow[q,r,h,s,τ,link_index] are 7-dim.
# The latter is most critical.
# "*" indicate that this is the most memory-critical var.
# For each strategy flow, node flow file with 
# memory requirement: N * N * T * num_choices * T * num_links_exp * 8 (byte)
# Case 1)
# If N = 20, T = 120, num_routes = 3, num_links_exp = N * T,
# = 20*20*120*4*120*(20*120)*8  = 412 GiB.
# Case 2)
# if N = 14, T = 60, num_routes = 2, num_links_exp = 3 * N,
# = 14*14*60*3*60*(14*60)*8  = 13 GiB.
# Case 3)
# if N = 14, T = 60, num_routes = 2, num_links_exp = 3 * N,
# = 14*14*90*3*60*14*90*8 = 30 Gib
#
# Note: if l == (num_choices - 1) or (l != (num_choices - 1) and tau == t): 
# is often used befor tying to acess preference sets.

import sys
import numpy as np
from copy import deepcopy
import datetime

from open_config_file import open_config_file
from congestion_penalty_coeff import congestion_penalty_coeff
from find_initial_strategy import find_initial_strategy

def dynamic_schedule_based_ue_assignment_algorithm(tn,od_flow, od_flow_to_stop_prob):
	""" Inptut time-dependent network and demand; output node probabilities."""

	# tn: time-dependent network obj;
	# od_flow: origin-destination flow matrix.

	print('    ---------------------------------------------------')
	print("    Schedule-based UE assignment algorithm begins ...")

	# Parameters
	TEST_LOAD_STRATEGY_FLOW = int(open_config_file('test_load_strategy_flow'))
	TEST_FIND_OPTIMAL_STRATEGY = int(open_config_file('test_find_optimal_strategy'))
	DISPLAY_PROGRESS = int(open_config_file('DISPLAY_PROGRESS'))
	MAX_NUMBER_OF_ITERATIONS = int(open_config_file('MAX_NUMBER_OF_ITERATIONS_FOR_ASSIGNMENT_MODEL'))
	DOUBLE_STREAMLINED = int(open_config_file('DOUBLE_STREAMLINED'))
	TRANSIT_TYPE = open_config_file('TRANSIT_TYPE')
	# If TEST_LOAD_STRATEGY_FLOW  == 1, algorithm stop after first assignment;
	# If TEST_FIND_OPTIMAL_STRATEGY == 1, algorithm will stop after first iteration;

	LARGE_CONST = int(open_config_file('LARGE_CONST'))

	# In this program, maximum number of strategies equals the maximun number of iterations,
	# since one new strategy is generated each iteration.
	# This could be different if other algorithms is adopted.
	# Set num_strategies
	num_strategies = MAX_NUMBER_OF_ITERATIONS

	# Number of stops.
	num_stops = len(tn.stops)
	num_stops_exp = len(tn.stops_exp)

	# Number of stages in dynamic pogramming
	# "+ 1" because an additional stage is added when an additional node is added.
	num_stages = len(tn.stops_exp) + 1

	# Time horizon.
	T = tn.T

	# Number of bus lines.
	num_routes = len(tn.routes)
	# The most number of choices user could have at a node, which equals
	# the number of routes, adding a waiting link.
	num_choices = num_routes + 1
	# Note: waiting link will has index = num_routes.

	# Number of arcs.
	num_links_exp = len(tn.links_exp)

	CONVERGENCE_CRITERION = float(open_config_file('CONVERGENCE_CRITERION_FOR_ASSIGNMENT_MODEL'))

	print()
	print("    T: " + str(T))
	print("    num_stops: " + str(num_stops))
	print("    num_stops_exp: " + str(num_stops_exp))
	print("    num_links_exp: " + str(num_links_exp))
	print("    num_routes: " + str(num_routes))
	print("    num_strategies: " + str(num_strategies))

	# Initialization
	# Preference sets
	print()
	print("    Initializing...")
	prefer_links = {}
	prefer_probs = {}

	# Used to create dict more efficiently.
	# The keys will be tuples.
	temp_key_set = set()
	for r in range(num_stops):
		for s in range(num_strategies):
			for l in range(num_choices):
				for i_t in range(num_stops_exp):

					t = tn.stops_exp_2[i_t][2]
					temp = 0
					if l != (num_choices - 1):
						temp = t

					for tau in range(temp, t + 1):
						temp_key_set.add((r,s,l,tau,i_t))

	prefer_links = {key: [] for key in temp_key_set}
	prefer_probs = {key: [] for key in temp_key_set}

	# Strategy flow[q,r,h,s]
	strategy_flow = np.zeros((num_stops, num_stops, T, num_strategies))

	# MSA iteration begins ...

	iteration_count = 1
	# At 1st iteration, available strategy (initial strategy) is indexed 0.
	# iteration_count_effective is used since sometimes there maybe strategy repetetions.
	iteration_count_effective = 1

	converg_flag = 0

	# Find initial strategy flow.
	# Note this _optimal vars will be reused later.
	prefer_links_optimal, prefer_probs_optimal = find_initial_strategy(tn)

	print()
	print("    Creating preference sets ...")
	# s = 0 correspond to the initial strategy.
	s = 0
	for r in range(num_stops):
		for l in range(num_choices):
			for i_t in range(num_stops_exp):

				# This technique will used often to speed up.
				t = tn.stops_exp_2[i_t][2]
				temp = 0
				if l != (num_choices - 1):
					temp = t

				for tau in range(temp, t + 1):
					prefer_links[r,s,l,tau,i_t] = deepcopy(prefer_links_optimal[r,l,tau,i_t])
					prefer_probs[r,s,l,tau,i_t] = deepcopy(prefer_probs_optimal[r,l,tau,i_t])

	# Assign all od_demand to the initial strategy.
	# s still 0.
	for q in range(num_stops):
		for r in range(num_stops):
			for h in range(T):
				strategy_flow[q,r,h,s] = od_flow[q,r,h]

	# Time-dependent UE iterations begin.
	while converg_flag == 0:
		print()
		print("    --------------------------")
		print("    " + str(iteration_count) + "-th iteration begins...")
		currentDT = datetime.datetime.now()
		print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
		print("    " + str(iteration_count_effective) + " effective strategies exist...")

		# Print the total flow on network.
		total_flow = strategy_flow.sum()
		print()
		print("    The total_flow is: " + str(total_flow))
		
		# Initialization. These vars will change in each iteration.
		# Note that they will not be used in optimal strategy finding except prob_of_boarding.

		# Link flow[r,s,l,τ,link_id]: no q,h info required;
		# Initialized for each iteration.
		link_flow = {}

		# link_combined_flow[link_index]
		# Calculated each iteration in order to update link costs;
		# Also act as function return.
		link_combined_flow = np.zeros(num_links_exp)

		# Store the probabilities of boarding each outgoing link.
		# No preference sets involved; just the probabillity for one-time boarding; not successive waiting & boarding.
		# prob_of_boarding[i_t,tau,l]
		# where tau is the arrival time at i_t, l is the route to board.
		# Initialized for each iteration.
		# The probability of boarding a WT link is always 1.
		#
		# Note: prob_of_boarding does not record the incoming route; 
		# hence continuence with priority prob (1.0) is not considered;
		# Remmber this fact in finding optimal strategy step.
		prob_of_boarding = np.zeros((num_stops_exp,T,num_choices))
		for temp in range(num_stops_exp):
			t = tn.stops_exp_2[i_t][2]

			for temp2 in range(t + 1):
				# The probability of boarding a WT link is always 1.
				prob_of_boarding[temp,temp2,num_choices - 1] = 1.0
		
		# Descending direction strategy flow
		# Initialized for each iteration.
		Y = np.zeros((num_stops, num_stops, T, num_strategies))
		# Cost of current strategy flows.
		C_X = 0
		# Cost of optimal strtegy flow's cost under current strategy flow loading.
		C_Y = 0

		# For each iteration, the prefer_prob need to be initialized to be zero,
		# since flows will be reloaded;
		# But the preference links, routes, downstream nodes are kept across iterations.
		for r in range(num_stops):
			for s in range(iteration_count_effective):
				for l in range(num_choices):
					for i_t in range(num_stops_exp):

						t = tn.stops_exp_2[i_t][2]
						temp = 0
						if l != (num_choices - 1):
							temp = t

						for tau in range(temp, t + 1):

							for temp_index in range(len(prefer_probs[r,s,l,tau,i_t])):
								prefer_probs[r,s,l,tau,i_t][temp_index] = 0.0

		# -------- Load current strategy flows onto the network.
		print()
		print("    --------------------------")
		print("    Loading current startegy flows onto TE network begins ...")
		# Given strategy_flow, prefer_stops, prefer_links, prefer_routes
		# Obtain link_prob, link_flow, node_prob, node_flow, link_combined_flow
		# and update link cost in tn.links_exp

		# Make a copy of the .links_exp once each loading, since the capacity will be modified
		# during assignment.
		links_exp_temp = deepcopy(tn.links_exp)

		# Loading strategy flow onto nodes; stage labeled by i_t.
		# Iterate over nodes index (i_t) in TE network in T&C order.
		for i_t in range(num_stops_exp):

			if DISPLAY_PROGRESS == 1:
				print()
				print("    *   *   *")
				print("    (Loading strategy flow) handling stop i_t: " + tn.stops_exp[i_t] + " - " + str(tn.stops_exp_2[i_t]))
				print("    len(link_flow): " + str(len(link_flow)))

				if TEST_LOAD_STRATEGY_FLOW == 1:
					print()
					print("    link_flow:")
					print(link_flow)
				# End of test print.

			# Node arrival flow, [r,s,l,τ]: no q,h, i_t;
			# No i_t info required, since it's initialized for each stage i_t.
			node_flow = {}

			# Index of phisical node in tn.stop
			i = tn.stops_exp_2[i_t][1]
			# Time of this node
			t = tn.stops_exp_2[i_t][2]

			# ---- Collecting incoming flows at i_t.
			# Update node_prob, node_flow.
			# Once link flows are loaded onto the link flows, they are deleted from
			# dict link_flow to save memory.

			# There are three cases:
			# Case 1) Users depart from i_t (arrive by "departing route"); these nodes
			#         will come from (l, tau) = (num_choices - 1, t);
			# Case 2) Users arrive at i_t by transit;
			# Case 3) Users arrive at i_t by "waiting route" - waiiting at that node;
			#         The difference between 2) and 3) is that the arrivel time at i_t doesn't
			#         change in case 3)!

			# Find the incoming links and extract arrtibutes.
			incoming_links = [item for item in links_exp_temp if item[3] == i_t]
			incoming_links_index = [item[0] for item in incoming_links]
			incoming_routes_index = [item[1] for item in incoming_links]
			incoming_links_tt = [item[5] for item in incoming_links]

			if TEST_LOAD_STRATEGY_FLOW == 1:
				print()
				print("    incoming_links[r,s,l,tau,link_id]:")
				print(incoming_links)
			# End of test print.

			# Case 1) Assign departure flow to nodes.
			q = i
			h = t
			l = num_choices - 1
			tau = t

			for s in range(iteration_count_effective):
				for r in range(num_stops):

					if strategy_flow[q,r,h,s] > 0.0001:

						node_flow_key = (r,s,l,tau)
						if node_flow_key in node_flow:
							node_flow[node_flow_key] += strategy_flow[q,r,h,s]
						else:
							node_flow[node_flow_key] = strategy_flow[q,r,h,s]

						# Print
						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Collecting flow - Case 1) assign departure flow to nodes - " + "node_flow" +\
							 str(node_flow_key) + ": " + str(strategy_flow[q,r,h,s]))
						# End of test print.

			# Case 2) & 3)
			# Iterate over link_flow.
			# If there is no incoming links for this node, these two cases will be skipped.

			# Used to record the dict elmt to be deleted.
			to_be_deleted = set()

			# link_flow_key: (r,s,l,t,outgoing_link)
			for link_flow_key in link_flow:

				incoming_link = link_flow_key[4]
				ending_node = tn.links_exp[incoming_link][3]
				# If this is link flow that flows into node i_t, ...
				if ending_node == i_t:

					r = link_flow_key[0]
					s = link_flow_key[1]
					# Note l is the incoming route for link flow, not the route of this link.
					l_prime = link_flow_key[2]
					tau_prime = link_flow_key[3]
					

					l = tn.links_exp[incoming_link][1]

					# Case 2) this (l) is a bus route, ...
					# tau is the time that user arrive at upstream node, range in [0, t - TT + 1],
					# namely, user can at most arrive at upstream node at time(t - incoming_links_tt[temp_index]).
					if l != (num_choices - 1):

						tau = t
						node_flow_key = (r,s,l,tau)

						if node_flow_key in node_flow:
							node_flow[node_flow_key] += link_flow[link_flow_key]
						else:
							node_flow[node_flow_key] = link_flow[link_flow_key]

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Collecting flow - Case 2): node_flow" + str(node_flow_key) + " get "\
							 + "incoming flow from transit link_flow" + str(link_flow_key) + ": "\
							 + str(link_flow[link_flow_key]))
						# End of test print.

					else:
					# Case 3) This is a waiting route.
					# Notice: the difference is that for a WT route,
					# the arrival time doesn't change!

						tau = tau_prime
						# Note that arrival time at i_t is tau.
						node_flow_key = (r,s,l,tau)
						
						if node_flow_key in node_flow:
							node_flow[node_flow_key] += link_flow[link_flow_key]
						else:
							node_flow[node_flow_key] = link_flow[link_flow_key]

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Collecting flow - Case 3): node_flow" + str(node_flow_key) + " get "\
							 +	"incoming flow from waiting link_flow" + str(link_flow_key) + ": "\
							 + str(link_flow[link_flow_key]))
						# End of test print.

					# Remember to delete assigned link_flow.
					to_be_deleted.add(link_flow_key)

			for link_flow_key in to_be_deleted:
				del link_flow[link_flow_key]

			# ---- Assign incoming flows at i_t to outgoing links according to their preference sets.
			# Update link_flow

			# There are three cases:
			# Case 1) flows that has empty preference set at i_t, which means this flow 
			#         is NOT able to reach their destination within time T; the amount of flow assigned
			#         to outgoing links and probability would be zero. Hence there is no need
			#         to treat this case on purpose, since zero is default value.
			#         But this case will be reported;
			# Case 2) flows that can, and choose to, enjoy the continuance priority;
			# Case 3) flows that cannot, or choose not to, enjoy continuance priority, 
			#         assigned based FIFO rule.

			# Find the outgoing links from i_t and extract arrtibutes.
			# Note that when the item's attribute of links_exp_temp, say capacity, is modified,
			# the item's attribute of outgoing_links will also be modified.
			# outgoing_links_copy is hence created to keep a record of un-modified outgoing links,
			# it will be used in boarding_probs calculation.
			outgoing_links = [item for item in links_exp_temp if item[2] == i_t]
			outgoing_links_index = [item[0] for item in outgoing_links]
			outgoing_routes_index = [item[1] for item in outgoing_links]
			outgoing_links_tt = [item[5] for item in outgoing_links]
			downstream_stops_index = [item[3] for item in outgoing_links]

			outgoing_links_copy = deepcopy(outgoing_links)
			outgoing_links_index_copy = deepcopy(outgoing_links_index)
			outgoing_routes_index_copy = deepcopy(outgoing_routes_index)

			if TEST_LOAD_STRATEGY_FLOW == 1:
				print()
				print("    node_flow[r,s,l,tau]:")
				print(node_flow)
				print()
				print("    outgoing_links: ")
				print(outgoing_links)
			# End of test print.

			# Copy files
			# Make a copy of preference sets and node flows, since they will be modified when satisfied.
			# prefer_probs are all zero now and to be modified, hence no need to make a copy.
			# These temp sets are initialized for each i_t, hence no i_t index is needed.
			prefer_stops_temp = {}
			prefer_links_temp = {}
			prefer_routes_temp = {}
			
			for r in range(num_stops):
				for s in range(iteration_count_effective):
					for l in range(num_choices):

						temp = 0
						if l != (num_choices - 1):
							temp = t

						for tau in range(temp, t + 1):
							# Note that the elmts are lists; deepcopy is used.
							prefer_links_temp[r,s,l,tau] = deepcopy(prefer_links[r,s,l,tau,i_t])
							prefer_routes_temp[r,s,l,tau] = [tn.links_exp[item][1] for item in prefer_links_temp[r,s,l,tau]]
							prefer_stops_temp[r,s,l,tau] = [tn.links_exp[item][3] for item in prefer_links_temp[r,s,l,tau]]

			if TEST_LOAD_STRATEGY_FLOW == 1:
				print()
				print("    Preference set of node i_t (prefer_routes_temp[r,s,l,tau]):")
				print(prefer_routes_temp)
			# End of test print.

			# Case 2) Assign flows that can, and choose to, enjoy the continuance priority first.
			# If i_t has no incoming links, this case is skipped.

			to_be_deleted = set()

			# Iterate over node flow elmts.
			for node_flow_key in node_flow:

				# It should correspond to strategy flows that use this link whose 
				# preference set has this route as 1st choice;
				# then they have continuence priority.
				r = node_flow_key[0]
				s = node_flow_key[1]
				# Incoming route of this node flow.
				l = node_flow_key[2]
				tau = node_flow_key[3]

				if l != (num_choices - 1) and tau == t:

					# Preference set has this route as 1st choice.
					# tau = t.
					if prefer_routes_temp[r,s,l,t] and prefer_routes_temp[r,s,l,t][0] == l:

						temp_flow = node_flow[node_flow_key]
						outgoing_link = outgoing_links_index[outgoing_routes_index.index(l)]
						
						# Modify the capacity of outgoing link.
						links_exp_temp[outgoing_link][7] -=  temp_flow
						# To avoid numerical issues.
						if links_exp_temp[outgoing_link][7] <= 0.0001:
							links_exp_temp[outgoing_link][7] = 0

						# Update probability and flows.
						# It must be the first time to update;
						# hence no need to use "+=".
						link_flow_key = (r,s,l,t,outgoing_link)
						link_flow[link_flow_key] = temp_flow
						# Update link_combined_flow.
						link_combined_flow[outgoing_link] += temp_flow

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Assign node flow - Case 2) - flow with continuance priority - from node_flow" + str(node_flow_key) +\
							 " to link_flow" + str(link_flow_key)  + ": " + str(link_flow[link_flow_key]))
							print("    Capacity of link - " + str(outgoing_link) + " updated to " + str(links_exp_temp[outgoing_link][7]))
						# End of test print.

						# Delete the demand when it has been satisfied.
						to_be_deleted.add(node_flow_key)

						prefer_probs[r,s,l,t,i_t][0] = 1.0
						if TEST_LOAD_STRATEGY_FLOW == 1:
							print("    prefer_probs" + str([r,s,l,t,i_t]) + "[0]" + " updated to 1.0;")
						# End of test print.

			for node_flow_key in to_be_deleted:
				del node_flow[node_flow_key]

			# Case 3) Assign flow that don't have or choose not to enjoy continuance priority in FIFO order.
			#		
			# Steps:
			# 3.0) Start from tau = 0;
			# 3.1) Find the 1st-choice demands from current preference set to each outgoing links;
			# 3.2) Find the (remaining) capacity for all outgoing links;
			# 3.3) identify the outgoing link with smallest capacity/demand ratio;
			#	   Notice that singularity case may happen: after continuance flow assigned in Case 2), some link's
			# 	   capacity and demand both goes to zero; this link has to be deleted from all preference sets;
			# 	   If not, you will find that {capacity/demand ratio} all non-meaningful.
			# 	   If singularity happens, delete from all preference sets; then go to step 3.3);
			# 3.4) If this ratio >= 1, then all flow assignmed to their 1st choice; tau += 1; goto step 3.1);
			# 	   If not, all flow assigned according to this ratio, and this link's capacity run out;
			#	   then delete this link from all demand's preference set, namely update choice set;
			# 	   then update the capacity;
			# 3.5) if all demands from tau have been assigned, tau += 1; go to step 3.1); 
			#	   If tau = t, then finish for this node i_t.
			#
			# Note: if t = 0, there could only be departing flows.
			
			# Iterate over time tau that strategy flow arrives at i_t.
			# Start from tau = 0, the eariest possible arrival time.
			for tau in range(t + 1):

				if TEST_LOAD_STRATEGY_FLOW == 1:
					print()
					print("    Assign node flow - Case 3) - current tau: " + str(tau))
				# End of test print.

				# tau_loading_flag is used to indicate whether the assignemnt for a specific tau has been finished.
				tau_loading_flag = 0

				while tau_loading_flag == 0:

					# Step 3.1) Find the 1st-choice demands for all outgoing links.
					first_choice_demand = np.zeros(len(outgoing_links_index))
					# Note: first_choice_demand is initialized to have the same dim with outgoing_links_index.
					# updated every time a link is saturated or preference set changes.

					for node_flow_key in node_flow:

						# Arrival time is tau.
						if node_flow_key[3] == tau:

							r = node_flow_key[0]
							s = node_flow_key[1]
							l = node_flow_key[2]

							if prefer_links_temp[r,s,l,tau]:										

								temp_index = outgoing_links_index.index(prefer_links_temp[r,s,l,tau][0])
								first_choice_demand[temp_index] += node_flow[node_flow_key]			

					if TEST_LOAD_STRATEGY_FLOW == 1:
						print()
						print("    Step 3.1) - first_choice_demand arriving at " + str(tau) + " is: ")
						print(first_choice_demand)
					# End of test print.

					# Step 3.2) Find the remaining capacity.
					remaining_capacity = np.zeros(len(outgoing_links_index))
					for temp_index in range(len(outgoing_links_index)):
						link_index = outgoing_links_index[temp_index]
						# Notice to use links_exp_temp file to find capacity, since it's updated.
						remaining_capacity[temp_index] = links_exp_temp[link_index][7]

					if TEST_LOAD_STRATEGY_FLOW == 1:
						print()
						print("    Step 3.2) - remaining_capacity is:")
						print(remaining_capacity)
					# End of test print.

					if first_choice_demand.sum() < 0.0001:
						tau_loading_flag = 1
						continue

					# Step 3.3) Identify the outgoing link with smallest capacity/demand ratio.
					# Note: singularity case may happen.
					min_ratio = float('inf')
					min_ratio_index = LARGE_CONST
					min_ratio_link_index = LARGE_CONST
					min_ratio_route_index = LARGE_CONST

					# This flag is used to "continue" under while and for loop.
					singularity_flag = 0
					for temp_index in range(len(outgoing_links_index) - 1, -1, -1):

						# Singularity handling.
						if remaining_capacity[temp_index] == 0 and first_choice_demand[temp_index] == 0:
							singularity_flag = 1
							print()
							print("    Singularity happens once - the index of link: " + str(outgoing_links_index[temp_index]) +\
							 " the index of route: " + str(outgoing_routes_index[temp_index]))

							saturated_link = outgoing_links_index[temp_index]
							# Delete this link from all related preference sets.
							for node_flow_key in node_flow:

								r = node_flow_key[0]
								s = node_flow_key[1]
								l = node_flow_key[2]

								for tau2 in range(tau,t + 1):
									if l == (num_choices - 1) or (l != (num_choices - 1) and tau2 == t):
										if saturated_link in prefer_links_temp[r,s,l,tau2]:
											temp = prefer_links_temp[r,s,l,tau2].index(saturated_link)
											del prefer_links_temp[r,s,l,tau2][temp]
											del prefer_routes_temp[r,s,l,tau2][temp]
											del prefer_stops_temp[r,s,l,tau2][temp]			

							# Delete this link from outgoing links.
							del outgoing_links[temp_index]
							del outgoing_links_index[temp_index]
							del outgoing_routes_index[temp_index]
							del outgoing_links_tt[temp_index]
							del downstream_stops_index[temp_index]

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print("    Current outgoing_links:")
								print(outgoing_links)
							# End of test print.

							# If singularity happens, then go back to step 3.1).
							if singularity_flag == 1:
								continue

						elif first_choice_demand[temp_index] > 0.0001:
							temp_ratio = remaining_capacity[temp_index] / first_choice_demand[temp_index]

							if temp_ratio < min_ratio:
							# Note: inf < inf is fase.
								min_ratio = temp_ratio
								# index in set outgoing_links
								min_ratio_index = temp_index
								min_ratio_link_index = outgoing_links_index[min_ratio_index]
								min_ratio_route_index = outgoing_routes_index[min_ratio_index]

					if singularity_flag == 1:
						continue

					if TEST_LOAD_STRATEGY_FLOW == 1 and singularity_flag == 0:
						print()
						print("    Step 3.3) - min_ratio: " + str(min_ratio))
						print("    min_ratio_index: " + str(min_ratio_index))
						print("    min_ratio_link_index: " + str(min_ratio_link_index))
						print("    min_ratio_route_index: " + str(min_ratio_route_index))
					# End of test print.

					# Step 3.4) Assign node flow.
					# If min_ratio >= 1, all demnand at time tau could be assigned 
					# to their 1st choices in the current preference set.
					# Update flow, probabilities; then tau ++; go to next iteration.
					if min_ratio >= 1:

						to_be_deleted = set()

						for node_flow_key in node_flow:
							if node_flow_key[3] == tau:
								r = node_flow_key[0]
								s = node_flow_key[1]
								l = node_flow_key[2]

								if prefer_links_temp[r,s,l,tau]:
									# Find 1st choice link index.
									outgoing_link = prefer_links_temp[r,s,l,tau][0]

									temp_flow = node_flow[node_flow_key]

									link_flow_key =  (r,s,l,tau,outgoing_link)

									if link_flow_key in link_flow:
										link_flow[link_flow_key] += temp_flow
									else:
										link_flow[link_flow_key] = temp_flow

									link_combined_flow[outgoing_link] += temp_flow

									# Modify capacties.
									links_exp_temp[outgoing_link][7] -= temp_flow
									# “0.0001” is to avoid numerical precision issues.
									if links_exp_temp[link_index][7] <= 0.0001:
										links_exp_temp[link_index][7] = 0

									if TEST_LOAD_STRATEGY_FLOW == 1:
										print()
										print("    Assign node flow - Case 3) - FIFO, min_ratio >=1 - from node_flow" + str(node_flow_key) +\
										 " to link_flow" + str(link_flow_key)  + ": " + str(temp_flow))
										print("    Capacity of link - " + str(outgoing_link) + " updated to " + str(links_exp_temp[outgoing_link][7]))
									# End of test print.

									# Delete assigned flows.
									to_be_deleted.add(node_flow_key)

									# Update prob.
									temp = prefer_links[r,s,l,tau,i_t].index(outgoing_link)

									prefer_probs[r,s,l,tau,i_t][temp] = 1.0 - sum(prefer_probs[r,s,l,tau,i_t])
									if TEST_LOAD_STRATEGY_FLOW == 1:
										print()
										print("    prefer_probs" + str([r,s,l,tau,i_t]) + '[' + str(temp) +']' + " updated to " + str(prefer_probs[r,s,l,tau,i_t][temp]))
									# End of test print.

						for node_flow_key in to_be_deleted:
							del node_flow[node_flow_key]

						tau_loading_flag = 1

					# If min_ratio < 1:
					# Assignem flow according to the ratio;
					# Delete the saturated links from all preference sets at i_t;
					# Go over to step 3.1) again.
					# Note that WT route has inf capacity, the saturated route must be 
					# transit routes.
					elif min_ratio < 1:

						for node_flow_key in node_flow:
							r = node_flow_key[0]
							s = node_flow_key[1]
							l = node_flow_key[2]

							if node_flow_key[3] == tau:
								if prefer_links_temp[r,s,l,tau]:
									outgoing_link = prefer_links_temp[r,s,l,tau][0]

									# Assign flow to outgoing links.
									temp_flow = min_ratio * node_flow[r,s,l,tau]

									link_flow_key = (r,s,l,tau,outgoing_link)
									if link_flow_key in link_flow:
										link_flow[link_flow_key] += temp_flow
									else:
										link_flow[link_flow_key] = temp_flow

									link_combined_flow[outgoing_link] += temp_flow

									# Modify capacties.
									links_exp_temp[outgoing_link][7] -= temp_flow

									if links_exp_temp[outgoing_link][7] <= 0.0001:
										links_exp_temp[outgoing_link][7] = 0
									
									if TEST_LOAD_STRATEGY_FLOW == 1:
										print()
										print("    Assign node flow - Case 3) - FIFO, min_ratio < 1 - from node_flow" + str(node_flow_key) +\
											 " to link_flow" + str(link_flow_key)  + ": " + str(temp_flow))
										print("    Capacity of link - " + str(outgoing_link) + " updated " + str(links_exp_temp[outgoing_link][7]))
									# End of test print.

									# Update flows.
									node_flow[node_flow_key] *= (1 - min_ratio)

									temp = prefer_links[r,s,l,tau,i_t].index(outgoing_link)
									prefer_probs[r,s,l,tau,i_t][temp] = min_ratio * (1.0 - sum(prefer_probs[r,s,l,tau,i_t]))

									if TEST_LOAD_STRATEGY_FLOW == 1:
										print()
										print("    prefer_probs" + str([r,s,l,tau,i_t]) + '[' + str(temp) +']' + " updated to " + str(prefer_probs[r,s,l,tau,i_t][temp]))
									# End of test print.

							# Modify the preference set.
							# Even the flow is zero, still need to purge preference set!
							for tau2 in range(tau,t + 1):
								if l == (num_choices - 1) or (l != (num_choices - 1) and tau2 == t):
									if min_ratio_link_index in prefer_links_temp[r,s,l,tau2]:
										temp = prefer_links_temp[r,s,l,tau2].index(min_ratio_link_index)
										del prefer_routes_temp[r,s,l,tau2][temp]
										del prefer_links_temp[r,s,l,tau2][temp]
										del prefer_stops_temp[r,s,l,tau2][temp]			

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print('    Link - ' + str(outgoing_links_index[min_ratio_index]) + " is deleted from outgoing_links.")
						# End of test print.

						# Delete this link from outgoing links.
						del outgoing_links[min_ratio_index] # Note that tn.links_exp will not be modified.
						del outgoing_links_index[min_ratio_index]
						del outgoing_routes_index[min_ratio_index]
						del outgoing_links_tt[min_ratio_index]
						del downstream_stops_index[min_ratio_index]

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Current outgoing_links:")
							print(outgoing_links)
						# End of test print.

			# ---- Compute prob_of_boarding.
			# Links are visited according to the seq in outgoing links.
			# Remenmber to use outgoing_links_copy, since outgoing_links has been modified above.
			if TEST_LOAD_STRATEGY_FLOW == 1:
				print()
				print("    link_flow[r,s,l,tau,link_id]:")
				print(link_flow)
			# End of test print.

			# Make a dict to save all routes and starting nodes; since they will be used often.
			outgoing_routes_all = {}
			starting_nodes_all = {}
			for link_flow_key in link_flow:
				outgoing_link_temp = link_flow_key[4]
				outgoing_routes_all[link_flow_key] = tn.links_exp[outgoing_link_temp][1]
				starting_nodes_all[link_flow_key] = tn.links_exp[outgoing_link_temp][2]

			# Iterate over all possible states (l,tau), namely (outgoing_route, tau)
			for outgoing_link_temp_index in range(len(outgoing_links_copy)):

				outgoing_link = outgoing_links_index_copy[outgoing_link_temp_index]
				outgoing_route = outgoing_routes_index_copy[outgoing_link_temp_index]

				# If outgoing_route is (num_choices - 1), no update, 
				# since its prob is 1.0 and has already been initialized.
				if outgoing_route != (num_choices - 1):

					if TEST_LOAD_STRATEGY_FLOW == 1:
						print()
						print("    (Computing prob_of_boarding) current link: ")
						print(outgoing_links_copy[outgoing_link_temp_index])
						print("namely,")
						print(tn.links_exp_char[outgoing_link])
					# End of test print.

					# If it's detected that the capacity has been used up,
					# then go to next outgoing_link.
					capacity_used_up_by_priority_flow_flag = 0

					# tau is the arrival time at node i_t.
					for tau in range(t + 1):

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    (Computing prob_of_boarding) current tau: " + str(tau))
						# End of test print.

						# Find the link probabilities for virtual flow user. This include finding:
						#
						# i) Finding the remaining_capacity for outgoing_link:
						# remaining_capacity =  capacity - used_capacity,
						# where used_capacity is the sum of flows that have priorities over virtual flow:
						# i.a) flow that arrives before tau (successfully board outgoing_link or not);
						# i.b) flow that arrives at t but have continuance priority.
						# Note: if it's deteted that some priority flow have to board some routes
						# that rank lower than outgoing_route, then there is no chance for virtual flow 
						# to board this outgoing_link; continue to next outgoing_link.
						#
						# ii) Finding the competing_flow for this outgoing_link
						# Virtual flow user will compete with strategy flow passengers that arrives 
						# at i at time = tau without continuance & FIFO priority over virtual flow;
						# Note that these competitors may or may not have being able to make it to board
						# this outgoing_link in previous loading process. 
						# Namely, the amount of flow that:
						# - outgoing_link is in its preference set at i_t;
						# - it arrives exacltly at tau;
						#   Note: users that arrives < tau has proiority over virtual user, and users arrives > tau
						#   should queue after virtual user.
						# - if tau == t, his/her prefer_set[0] is NOT the route where he come;
						#   Note: if tau = t, the flow that depart at i_t is also counted in.
						#   (NO continuence priority over virtual flow)
						# - user that chosed the route that rank_in_prefer_set is equal to or lower than outgoing_link.

						used_capacity = 0
						competing_flow = 0
						remaining_capacity = outgoing_links_copy[outgoing_link_temp_index][7]

						# Following iterations will be repeated |outgoing_links|*(t+1) many times - maybe inefficient!
						for link_flow_key in link_flow:

							starting_node2 = starting_nodes_all[link_flow_key]

							# If this link emernating from i_t.
							if starting_node2 == i_t:

								r2 = link_flow_key[0]
								s2 = link_flow_key[1]
								# Note l2 is where this flow from.
								l2 = link_flow_key[2]
								tau2 = link_flow_key[3]
								outgoing_link2 = link_flow_key[4]
								outgoing_route2 = outgoing_routes_all[link_flow_key]

								if prefer_links[r2,s2,l2,tau2,i_t] and outgoing_link in prefer_links[r2,s2,l2,tau2,i_t]:

									rank_in_prefer_set = prefer_links[r2,s2,l2,tau2,i_t].index(outgoing_link)
									rank_in_prefer_set2 = prefer_links[r2,s2,l2,tau2,i_t].index(outgoing_link2)

									# Used capacity
									# i.a) If flow that arrives before tau and board j_cij (tau2 < tau);
									# outgoing_link is in his pregerence set;
									# and he didn't board the outgoing_link (outgoing_link2 != outgoing_link);
									# and the outgoing_link of this flow has rank rank_in_prefer_set2 > rank_in_prefer_set;
									# it should NOT be used if outgoing_link had enough capacity.
									# then that means the capacity has been used up by proproty flows;
									# there is no chance of boarding; go to next outgoing_link.
									
									# First check whether there is a strategy flow signing "used-up" situation;
									# If it does, continue to next outgoing link - there is no need to iterate over
									# larger tau, there will be no chance for these people to board.

									if tau2 < tau and outgoing_link2 != outgoing_link and rank_in_prefer_set2 > rank_in_prefer_set:
										capacity_used_up_by_priority_flow_flag = 1

										if TEST_LOAD_STRATEGY_FLOW == 1:
											print()
											print("    Capacity used up by flows with priorities.")
										# End of test print.

										break

									# If not, and if this flow arrives earilier than tau, 
									# and the outgoing_link2 od this flow is current outgoing_link;
									# then add this outgoing_link flow to used_capacity. 
									if tau2 < tau and outgoing_link2 == outgoing_link:
										used_capacity += link_flow[link_flow_key]

										if TEST_LOAD_STRATEGY_FLOW == 1:
											print()
											print("    used_capacity increased from early arrival link_flow" + str(link_flow_key)\
											 + ": " + str(link_flow[link_flow_key]))
										# End of test print.

									# i.b) this link_flow arrives at t but have continuance priority.
									# If outgoing_route is the 1st preferred and incoming route equals outgoing_route,
									# and this toute is not WK route...
									# Note: here shows why the incoming route of link flow must be recorded;
									# otherwise, you cannot tell whether these flow have continuence priority or just
									# transfer and board outgoing_route.
									if (tau2 == t)\
									 and (outgoing_route != (num_choices - 1))\
									 and (outgoing_link2 == outgoing_link)\
									 and (outgoing_route == l2)\
									 and (outgoing_link2 == prefer_links[r2,s2,l2,tau2,i_t][0]):
										# Note: this user has to arrives at i_t at t.
										used_capacity += link_flow[link_flow_key]

										if TEST_LOAD_STRATEGY_FLOW == 1:
											print()
											print("    used_capacity increased from continuance link_flow" + str(link_flow_key)\
											 + ": " + str(link_flow[link_flow_key]))
										# End of test print.

									# Note: from here we can see why we need to add 'l' dimension for link_flow. 
									# if link_flow's upstream link l is not recorded,then 
									# 1) previous used_capacity flow may be repeatedly added!
									# for example, a user come from l' arrive at tau  < t and take outgoing_link;
									# But when you range over l, his contribution to the link flow will be added 
									# multiple times!
									# 
									# 2) here we cannot tell the flow is really competig flow!
									# for example, a user from another bus line l' arrive at time t,
									# he transfer to l. But tau2 == t, outgoing_route != (num_choices - 1))
									# also l == outgoing_route. You would think this is a continuence 
									# flow; because the information l' is lost.

									# ii) Competing flow
									# If this flows' prefer set is not empty and index of outgoing_link is in 
									# this preference set and has no continuance priority, and arrives at tau...
									# Note: l is where this flow from;  (outgoing_link2 == outgoing_link) is not required.
									# tn.links_exp[prefer_links[r2,s2,l2,tau2,i_t][0]][1] is the_first_choice_route 
									if (tau2 == tau)\
									 and (outgoing_link in prefer_links[r2,s2,l2,tau2,i_t])\
									 and ((tau != t) or (l2 != tn.links_exp[prefer_links[r2,s2,l2,tau2,i_t][0]][1]))\
									 and rank_in_prefer_set2 >= rank_in_prefer_set:

										competing_flow += link_flow[link_flow_key]

										# Add flows whose preferenece set of this strategy at i_t which is 
										# preferred lower than or equal to outgoing_link;
										# Note that even some competing flow is not able to board outgoing link,
										# which means that the capacity is used up, but virtual flow still may
										# board it.

										# Competing strategy flow that arrive at tau but used links with priorities
										# equal or lower than outgoing_links.

										if TEST_LOAD_STRATEGY_FLOW == 1:
											if rank_in_prefer_set2 == rank_in_prefer_set:
												print()
												print("    competing_flow increased from successful-broading link_flow" + str(link_flow_key)\
												 + ": " + str(link_flow[link_flow_key]))
											else:
												print()
												print("    competing_flow increased from failing-broading link_flow" + str(link_flow_key)\
												 + ": " + str(link_flow[link_flow_key]))
										# End of test print.

						if capacity_used_up_by_priority_flow_flag == 1:

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print()
								print("    The boarding prob for this link for time >= tau is 0, since capacity_used_up_by_priority_flow_flag is 1;")
								print("    move to next link.")
							# End of test print.

							break

						remaining_capacity -= used_capacity
						remaining_capacity = max(remaining_capacity,0)
						
						# “0.0001” is to avoid numerical precision issues.
						if remaining_capacity < 0.0001:
							remaining_capacity = 0

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    remaining_capacity is " + str(remaining_capacity))
							print("    competing_flow is " + str(competing_flow))
						# End of test print.
								
						# Compute virtual_flow_prob
						# prob_of_boarding = min{1, remaining_capacity / competing_flow}
						# virtual_flow_prob = probability_left * prob_of_boarding

						# If there is no competitors and there is capacity left.
						if competing_flow == 0 and remaining_capacity != 0:
							prob_of_boarding[i_t,tau,outgoing_route] = 1.0
						# Degenerate case
						elif remaining_capacity == 0:
							prob_of_boarding[i_t,tau,outgoing_route] = 0
						# General case
						else:
							prob_of_boarding[i_t,tau,outgoing_route] = min(1,remaining_capacity / competing_flow)

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print("    prob_of_boarding" + str([i_t,tau,outgoing_route]) + " updated to: " + str(prob_of_boarding[i_t,tau,outgoing_route]))
						# End of test print.

			# Note: link_combined_flow has to be computed in this process, 
			# since is not saved over the process of whole network loading.

		print()
		print("    All nodes have been visited for assignmen strategy flow!")

		# -------- Update link costs based on current link flow.
		print()
		print("    --------------------------")
		print("    Updating link costs and compute link_combined_flow ...")

		# Update transit link costs.
		for temp_index in range(num_links_exp):

			if tn.links_exp[temp_index][4] == TRANSIT_TYPE:
				# cost = TT * congestion_penalty_coeff(link_flow, capacity, TRANSIT_TYPE)
				temp = congestion_penalty_coeff(link_combined_flow[temp_index], tn.links_exp[temp_index][7], TRANSIT_TYPE)
				tn.links_exp[temp_index][8] = tn.links_exp[temp_index][5] * temp

			# Update C_X, the total cost.
			C_X += link_combined_flow[temp_index] * tn.links_exp[temp_index][8]

			if TEST_LOAD_STRATEGY_FLOW == 1 and link_combined_flow[temp_index] > 0.0001:
				print()
				print("    Link: " + str(tn.links_exp_char[temp_index])  + "   flow: " + str(link_combined_flow[temp_index]))
				print("    congestion_penalty_coeff: " + str(temp) + "   link cost: " + str(tn.links_exp[temp_index][8]))
			# End of test print.

		print("    C_X: " + str(C_X))

		if iteration_count == 1:
			with open("results/dynamic/C_X.csv", 'w') as f:
				f.write(str(C_X))
		else:
			with open("results/dynamic/C_X.csv", 'a') as f:
				f.write('\n')
				f.write(str(C_X))
		# Note: np.save('filename',var)
		# np.load(filename.npy')
		print()
		print("    Writing files ...")
		#np.save('results/node_flow_iteration_' + str(iteration_count),node_flow)
		#np.save('results/link_flow_iteration_' + str(iteration_count),link_flow)
		np.save('results/dynamic/link_combined_flow_lower_level_iteration_' + str(iteration_count),link_combined_flow)
		print("    Strategy flow loading stage has finished!")

		# -------- Determine the optimal strategy s* for each r depending on current loaded strategy flow.
		# Given the above loaded flow by dynamic programming,
		# find prefer_links_optimal, prefer_routes_optimal, prefer_stops_optimal.

		# Notice: don't use "continue" statement before "iteration_count += 1"
		# otherwise, iteration_count will stay at "1" forever.

		print()
		print("    --------------------------")
		print("    Finding the optimal strategy under current loaded strategy flows ...")

		# Optimal preference set initialized for each iteration.
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

		# Iterate over destination r; r is the index of destination in tn.stops.
		for r in range(num_stops):

			# w_cost is the cost to r;
			# w_cost initialized for each r;
			# w_cost[l,tau,i_t]
			# Initialized to be inf.
			w_cost = float('inf') * np.ones((num_choices, T, num_stops_exp))

			dest_name = tn.stops[r]

			if DISPLAY_PROGRESS == 1:
				print()
				print("    *   *   *")
				print("    (Finding optimal strategy) current destination: " + dest_name + " - " + str(r))

			# Find the optimal strategy.
			# Using dynamic programming; iteration over stages; 
			# TE network is acyclic, stages are labeled by nodes in REVERSE T&C order.
			# Here 'i_t' will be used instead of 'n'.
			#
			# State (i_t,l,tau) interpretation:
			# l != (num_choices - 1), tau = t: users arrive at i_t by transit link;
			# l == (num_choices - 1), tau < t: users arrive at i_t by waiting link;
			# l == (num_choices - 1), tau = t: users depart from i_t.
			#
			# Choices to be made are preference set at state (i_t,l,tau).
			#
			# State transision function is determined by the virtual flow assigment.

			# Visit nodes in reverse T&C order.
			for i_t in range(num_stops_exp - 1, -1, -1):

				if DISPLAY_PROGRESS == 1:
					print()
					print("    (Finding optimal strategy) destination: " + dest_name + " - " + str(r) + " stage: " + tn.stops_exp[i_t] + " - " +  str(i_t))

				# Current stop i_t's i part, which is index in tn.stops.
				i = tn.stops_exp_2[i_t][1]
				# Current stop time is denoted by t.
				t = tn.stops_exp_2[i_t][2]

				# Recursion formula to find w_cost, there are three cases:
 				#
				# Case 1) this node is a r_t node
				#   	  preference set remain empty as the default.
				# Case 2) this node not connected to r in TE network
				#   	  From this node user cannot reach the destination;
				#   	  preference set remain empty as the default.
				#    	  remain w to be float('inf') as the defalut.
				# Case 3) the general case.

				# Case 1) this node is a r_t node
				if i == r:
					for l in range(num_choices):
						for tau in range(t + 1):
							w_cost[l,tau,i_t] = 0

					if TEST_FIND_OPTIMAL_STRATEGY == 1:
						print()
						print("    Case 1) this node is a r_t node")
					# End of test print.

				else:
					# Find the outgoing links index of this node.
					# No need to deepcopy.
					outgoing_links = [item for item in tn.links_exp if item[2] == i_t]

					# Case 2) (1st half part) this node not connected to r in TE network
					# This node has no outgoing links.
					# Pass.
					if not outgoing_links:

						if TEST_FIND_OPTIMAL_STRATEGY == 1:
							print()
							print("    Case 2) (1st half part) this node not connected to r in TE network")
						# End of test print.

						pass

					# Case 3)
					else:
						outgoing_links_index = [item[0] for item in outgoing_links]
						outgoing_routes_index = [item[1] for item in outgoing_links]
						downstream_stops_index = [item[3] for item in outgoing_links]

						if TEST_FIND_OPTIMAL_STRATEGY == 1:
							print()
							print("    Case 3) General case")
							print("    outgoing_links:")
							print(outgoing_links)
						# End of test print.

						incoming_routes_index = [item[1] for item in tn.links_exp if item[3] == i_t]

						# Notes on how to calculate the preference set and w_cost:
						# -- The preference set generation
						# 1) for users who arrives >= t, or at t but cannot/choose not to continue,
						#    just sort phi_cost to get the preference set;
						# 2) for users who arrives at t and could enjoy continuence,
						#    this question has to be answered: should this user alight the train?
						#    If do, go to 1);
						#    If not, preference set just need to include the continuing route.
						#
						# -- w_cost update
						# The cost has to be based on the probability of boarding each link 
						# in preference set.

						# Case 3) the general case						
						# Case 3.1) state: (i_t, l = num_choices - 1, t) or 
						# (i_t, l != num_choices - 1, t) but l end at i_t.
						#   Virtual flow user depart from i_t, namely q=i, h=t.
						#   These users have no has no continuence priority nor FIFO priority，
						# and their states are (i_t, num_choices, t).
						#
						# Case 3.2) state: (i_t, l = num_choices - 1, tau)
						#   Virtual flow user arrive at i_t from "waiting route";
						#   User can arrive at time tau < t; discuss on tau.
						# Note that what only matters now is the time s/he arrived at i, namely the value tau;
						# the transit route he had taken doesn't matters any more.
						#
						# Case 3.3) state: (i_t, l != num_choices - 1, t) and l in outgoing route set.
						# Case 3.3.1) 
						#   Virtual flow user arrive from route l at time t, which is not 'waiting route'
						# but he alighted the train, thus he lost the continuance priority,
						# and he has no FIFO priority. This case is basicly the samw with Case 3.1.
						# Its cost w_cost has to be compared with cost in Case 3.3.2,
						# The strategy with smaller cost will be selected by user.
						#   Note: when virtual flow user left the train, s/he no longer 
						# enjoys continuance priority; and s/he also don't have any FIFO priority.
						#
						# Case 3.3.2)
						#   Suppose that virtual flow user can continue and choose to 
						# continue the same bus route (l).
						# Compare the cost of case 3.3.1 and 3.3.2 to answer the
						# question: should the user leave the train to join the queue?
						#
						# Case 3.1 and 3.3.1 are the same.
						# Case 3.1, 3.2 and 3.3.1 can be handled together.
						#
						# Case 3.1, 3.2 and 3.3.1 - No continuance priority
						# Iterate over all arrival time, tau.
						# tau may take value t;
						for tau in range(t + 1):
					
							if TEST_FIND_OPTIMAL_STRATEGY == 1:
								print()
								print("    Current tau: " + str(tau))
							# End of test print.

							# phi, expected cost of reaching destination from node i_t via node j_(t + cij)
							# Initialized for each (r,i_t) iteration.
							# The first col tells which record in outgoing_links it correspounds.
							phi_cost = np.zeros((len(outgoing_links),2))
							for temp_index in range(len(outgoing_links)):
								phi_cost[temp_index,0] = temp_index
								phi_cost[temp_index,1] = float('inf')

							for temp_index in range(len(outgoing_links)):
								downstream_stop = downstream_stops_index[temp_index]
								outgoing_link = outgoing_links_index[temp_index]
								outgoing_route = outgoing_routes_index[temp_index]

								# Ending node time
								t_cij = tn.stops_exp_2[downstream_stop][2]

								# Obtain phi
								# (i_t,downstream_stop) cost  +  w[l,arrival_time,downstream_stop]
								# Note that the arrival time will depend on the route type.
								if outgoing_route != (num_choices -1):
									phi_cost[temp_index][1] = tn.links_exp[outgoing_link][8] + w_cost[outgoing_route,t_cij,downstream_stop]
								else:
									phi_cost[temp_index][1] = tn.links_exp[outgoing_link][8] + w_cost[outgoing_route,tau,downstream_stop]

							# Sort phi in descending order
							# Note that phi may have infinite value elmts.
							phi_cost = phi_cost[phi_cost[:,1].argsort()]
							#phi_cost.sort(key = lambda x:x[1])
							# This is wrong, since phi_cost is a numpy file, not a list.


							if TEST_FIND_OPTIMAL_STRATEGY == 1:
								print()
								print("    Sorted phi_cost is:")
								print(phi_cost)
							# End of test print.

							# Consideration of a very special case:
							# If the cost of WT and some other links have equal cost,
							# make WT the first choice.
							smallest_cost = phi_cost[0,1]
							smallest_cost_route = outgoing_routes_index[int(phi_cost[0,0])]

							if smallest_cost_route != (num_choices - 1)\
							 and  smallest_cost != float('inf'):
								temp_flag = 0
								temp_index_2 = 99999

								if len(outgoing_links) >= 2:
									for temp_index in range(1, len(outgoing_links)):
										if phi_cost[temp_index,1] == smallest_cost\
										 and outgoing_routes_index[int(phi_cost[temp_index,0])] == (num_choices - 1):
											temp_flag = 1
											temp_index_2 = temp_index
											break

									if temp_flag == 1:
										# Switch the 1st ranked route and the equal cost WK route.
										phi_cost[0,0] = temp_index_2
										phi_cost[temp_index_2, 0] = 0										

										if TEST_FIND_OPTIMAL_STRATEGY == 1:
											print()
											print("    Re-ordered sorted phi_cost is:")
											print(phi_cost)
										# End of test print.

							# Case 2) (2nd half part) this node not connected to r in TE network
							# This node is not connected to dest r.
							if np.amin(phi_cost[:,1]) == float('inf'):

								if TEST_FIND_OPTIMAL_STRATEGY == 1:
									print()
									print("    Case 2) (2nd half part) this node not connected to r in TE network")
								# End of test print.

								continue

							# Preferred stops at this node
							prefer_links_1 = []
							prefer_probs_1 = []

							# Note that more links than needed (according to virtual_flow_prob) are 
							# added to preference set;
							# otherwise, users under this "optimal strategy" may not 
							# be able to go to the destination in actual loading process which happens next!
							for temp_index in range(len(outgoing_links)):		
								temp = int(phi_cost[temp_index,0])
								prefer_links_1.append(outgoing_links_index[temp])
								prefer_probs_1.append(0.0)

							# virtual_flow_prob calculation
							# Used to record the assign probability over prefer_links.
							virtual_flow_prob = []

							# Used to tell wheter this is the first elmt in phi_cost.
							# The procedures for updating probability for the first and the others are
							# a little bit different.
							temp_count = 0

							# Note that links are visited according to the seq in phi_cost.
							for outgoing_link_temp_index in range(len(phi_cost)):

								# If this is an effective arc in preference set ...
								if phi_cost[outgoing_link_temp_index,1] < float('inf'):

									# The index of this downstream node.
									# Note that links are visited according to the seq in phi_cost.
									index_in_outgoing_links = int(phi_cost[outgoing_link_temp_index,0])
									outgoing_route = outgoing_routes_index[index_in_outgoing_links]

									# Compute virtual_flow_prob
									# prob_of_boarding = min{1, remaining_capacity / competing_flow}
									# virtual_flow_prob = probability_left * prob_of_boarding
									# This step is moved to strategy flow loading process.

									# If this is the first time updating probabilities.
									if temp_count == 0:
										virtual_flow_prob.append(prob_of_boarding[i_t,tau,outgoing_route])
									elif temp_count > 0:
										virtual_flow_prob.append((1 - prob_accumulated) * prob_of_boarding[i_t,tau,outgoing_route])
									
									temp_count += 1

									if TEST_FIND_OPTIMAL_STRATEGY == 1:
										print()
										print("    virtual_flow_prob updated to:")
										print(virtual_flow_prob)
									# End of test print.

									prob_accumulated = 0
									for temp in range(len(virtual_flow_prob)):
										prob_accumulated += virtual_flow_prob[temp]

									if prob_accumulated >= (1 - 0.0001):
									# Preference set should never run out before prob_accumulated comes to 1,
									# becuase there is at least a waiting link with infinite capacity.
										if TEST_FIND_OPTIMAL_STRATEGY == 1:
											print()
											print("    probability accumulated to 1.")
										# End of test print.

										break

							# Compute the expected cost.
							# Temp vars are used, to facilitate comparison with 3.3.2.
							w_cost_1 = 0
							for temp in range(len(virtual_flow_prob)):
								w_cost_1 += virtual_flow_prob[temp] * phi_cost[temp,1]

							if TEST_FIND_OPTIMAL_STRATEGY == 1:
								print()
								print("    w_cost_1 : " + str(w_cost_1))
							# End of test print.

							# Case 3.1 - state: (i_t, l = num_choices - 1, t), finalize the results
							if tau == t:

								if incoming_routes_index:
									for temp in range(len(incoming_routes_index)):
										l = incoming_routes_index[temp]
										# From WK route, or from bus but cannot continue.
										if (l == (num_choices - 1)) or (l not in outgoing_routes_index):
											w_cost[l,tau,i_t] = w_cost_1
											prefer_links_optimal[r,l,tau,i_t] = deepcopy(prefer_links_1)
											prefer_probs_optimal[r,l,tau,i_t] = deepcopy(prefer_probs_1)

									# Note: once the incoming route is not empty, this node cannot be a first-time node
									# hence it must have incoming WT route at least.

								# Some nodes may don't have incoming links - first-time node, you still need to add preference sets for WT route;
								# In fact for departure flows.
								else:
									l = num_choices - 1
									w_cost[l,tau,i_t] = w_cost_1
									prefer_links_optimal[r,l,tau,i_t] = deepcopy(prefer_links_1)
									prefer_probs_optimal[r,l,tau,i_t] = deepcopy(prefer_probs_1)

								if TEST_FIND_OPTIMAL_STRATEGY == 1:
									print()
									print("    Case 3.1 - prefer_links_optimal" + str([r,l,tau,i_t]) + ":")
									print(prefer_links_optimal[r,l,tau,i_t])
									print("    Case 3.1 - prefer_probs_optimal" + str([r,l,tau,i_t]) + ":")
									print(prefer_probs_optimal[r,l,tau,i_t])
								# End of test print.

								# Update the optmial strategy cost for flow that start from i to r and depart at t.
								# Note: 
								# 1) In calculate C_Y, we assume that all od_flow are loaded onto optimal strategy.
								# 2) Departure q = i, dest = r, departure h = t, arrival route  = num_choices - 1 (WT). 
								C_Y += od_flow[i,r,t] * w_cost_1
								if TEST_LOAD_STRATEGY_FLOW == 1 and od_flow[i,r,t] > 0.0001:
									print()
									print("    C_Y got updated from od_flow" + str([i,r,t]) + ' * w_cost_1' + ' : ' + \
									 str(od_flow[i,r,t]) + ' * ' + str(w_cost_1))
								# End of test print.

							# Case 3.2 - state: (i_t, l = num_choices - 1, tau), finalize the results
							# Arrive earlier than t.
							# The set of incoming links cannot be empty;
							# and it must incude a WT route.
							elif tau < t:
								for temp in range(len(incoming_routes_index)):
									l = incoming_routes_index[temp]
									w_cost[l,tau,i_t] = w_cost_1
									prefer_links_optimal[r,l,tau,i_t] = deepcopy(prefer_links_1)
									prefer_probs_optimal[r,l,tau,i_t] = deepcopy(prefer_probs_1)

									if TEST_FIND_OPTIMAL_STRATEGY == 1:
										print()
										print("    Case 3.2 - prefer_links_optimal" + str([r,l,tau,i_t]) + ":")
										print(prefer_links_optimal[r,l,tau,i_t])
										print("    Case 3.2 - prefer_probs_optimal" + str([r,l,tau,i_t]) + ":")
										print(prefer_probs_optimal[r,l,tau,i_t])
									# End of test print.

							# Note: from here we can see why we need to add 'tau' dimension to prefernce sets.
							# w_cost_1 is for waiting route is related to his arrival time at this node,
							# preference set is generated by comparing costs; then it's natural to suppose that 
							# preference set should depend on tau.

							# Case 3.3.2:
							# Suppose that virtual flow user can continue and indeed choose to 
							# continue the same bus route (l).
							if tau == t:
								for temp_index in range(len(incoming_routes_index)):
									l = incoming_routes_index[temp_index]

									if (l != (num_choices - 1)) and (l in outgoing_routes_index):

										# Find the link of that route
										downstream_stop = downstream_stops_index[outgoing_routes_index.index(l)]
										t_cij = tn.stops_exp_2[downstream_stop][2]
										outgoing_link = outgoing_links_index[outgoing_routes_index.index(l)]
										outgoing_link_cost = tn.links_exp[outgoing_link][8]

										w_cost_2 = outgoing_link_cost + w_cost[l,t_cij,downstream_stop]

										if TEST_FIND_OPTIMAL_STRATEGY == 1:
											print()
											print("    w_cost_2 : " + str(w_cost_2))
										# End of test print.
										
										# Case 3.3: finalize results.
										# Compare w_cost_1 and w_cost_2 to decide whether virtual flow user
										# should choose to continue the bus route.
										# Note that w_cost_2 has already been "finalized",
										# update the final result if w_cost_1 is smaller.
										#
										# Notice here strict < is used.
										# “0.0001” is to avoid numerical precision issues.
										if w_cost_2  + 0.0001 < w_cost_1:

											if TEST_FIND_OPTIMAL_STRATEGY == 1:
												print()
												print("    User choose to continue the journey.")
											# End of test print.

											# For continuance flow, only one elmt is enough, 
											# this user is definetly able to board.
											w_cost[l,tau,i_t] = w_cost_2
											prefer_links_optimal[r,l,tau,i_t] = []
											prefer_probs_optimal[r,l,tau,i_t] = []

											prefer_links_optimal[r,l,tau,i_t].append(outgoing_link)
											prefer_probs_optimal[r,l,tau,i_t].append(0.0)

										else:

											if TEST_FIND_OPTIMAL_STRATEGY == 1:
												print()
												print("    User choose to alight the transit.")
											# End of test print.

											w_cost[l,tau,i_t] = w_cost_1
											prefer_links_optimal[r,l,tau,i_t] = deepcopy(prefer_links_1)
											prefer_probs_optimal[r,l,tau,i_t] = deepcopy(prefer_probs_1)

										if TEST_FIND_OPTIMAL_STRATEGY == 1:
											print("    Case 3.3 - prefer_links_optimal" + str([r,l,tau,i_t]) + ":")
											print(prefer_links_optimal[r,l,tau,i_t])
											print("    Case 3.3 - prefer_probs_optimal" + str([r,l,tau,i_t]) + ":")
											print(prefer_probs_optimal[r,l,tau,i_t])
										# End of test print.

		print()
		print("    C_X: " + str(C_X))
		print("    C_Y: " + str(C_Y))

		if iteration_count == 1:
			with open("results/dynamic/C_Y.csv", 'w') as f:
				f.write(str(C_Y))
		else:
			with open("results/dynamic/C_Y.csv", 'a') as f:
				f.write('\n')
				f.write(str(C_Y))

		print("    Optimal strategy obtained for all destinations!")

		# -------- Convergence Test
		# Relative gap function is used.
		print()
		print("    --------------------------")
		relative_gap = abs((C_X - C_Y)/C_X)
		print()
		print("    iteration_count: " + str(iteration_count) +  " iteration_count_effective: " + str(iteration_count_effective))
		print("    relative_gap value: " + str(relative_gap))

		if iteration_count == 1:
			with open("results/dynamic/relative_gap_lower_level.csv", 'w') as f:
				f.write(str(relative_gap))
		else:
			with open("results/dynamic/relative_gap_lower_level.csv", 'a') as f:
				f.write('\n')
				f.write(str(relative_gap))

		if relative_gap <= CONVERGENCE_CRITERION:
			print("    iteration_count: " + str(iteration_count) +  " CONVERGENCE_CRITERION met!")
			converg_flag = 1

		# If double-streamlined method is chosen, then just execute one iteration, then jump out of the iteration.
		if DOUBLE_STREAMLINED == 1 and iteration_count == 2:
			# The first iteration load theinitial flow, hence after the 2nd one.
			converg_flag = 1

		if iteration_count == MAX_NUMBER_OF_ITERATIONS:
				print()
				print("    MAX_NUMBER_OF_ITERATIONS reached without converging!")
				converg_flag = 1

		# -------- Compute the return.
		if converg_flag == 1:
			print()
			print("    --------------------------")
			print("    Computing the return of assignment algorithm...")

			# If the od_flow > 0, then it's probability will be updated;
			# hence it's prob valus are initialized to be zeros.
			for q in range(num_stops):
				for r in range(num_stops):
					for h in range(T):

						if od_flow[q,r,h] > 0.0001:
							for i in range(num_stops):
								for t in range(h,T):
									od_flow_to_stop_prob[q,r,h,i,t] = 0

			links_exp_temp = deepcopy(tn.links_exp)

			# link_flow[q,r,h,s,l,τ,link_index]\
			# Compared to previous handling, q,h is added;
			link_flow = {}

			for i_t in range(num_stops_exp):					

				if DISPLAY_PROGRESS == 1:
					print()
					print("    *   *   *")
					print("    (Computing return) handling stop: " + tn.stops_exp[i_t] + " - " + str(tn.stops_exp_2[i_t]))
			
				# node_flow[q,r,h,s,l,τ].
				# No i_t info required, since it's initialized for each stage.
				# Compared to previous handling, q,h is added;
				node_flow = {}

				# Index of phisical node in tn.stop
				i = tn.stops_exp_2[i_t][1]
				# Time of this node
				t = tn.stops_exp_2[i_t][2]

				# ---- Collecting incoming flows at i_t.
				# Update node_prob, node_flow.
				# Once link flows are loaded onto the link flows, they are deleted from dict link_flow to save memory.
				#
				# This part is almost the same as before, except q,h index being added.

				# There are three cases:
				# Case 1) Users depart from i_t (arrive by "departing route"); these nodes
				#         will come from (l, tau) = (num_choices - 1, t);
				# Case 2) Users arrive at i_t by transit;
				# Case 3) Users arrive at i_t by "waiting route" - waiiting at that node;
				#         The difference between 2) and 3) is that the arrivel time at i_t doesn't
				#         change in case 3)!
				#
				# Note: these flow and probabilities are zeros by default; just need 
				# to update these non-zero terms.

				# Find the incoming links and extract arrtibutes.
				incoming_links = [item for item in links_exp_temp if item[3] == i_t]
				incoming_links_index = [item[0] for item in incoming_links]
				incoming_routes_index = [item[1] for item in incoming_links]
				incoming_links_tt = [item[5] for item in incoming_links]

				if TEST_LOAD_STRATEGY_FLOW == 1:
					print("    incoming_links:")
					print(incoming_links)
					print()
					print("    link_flow:")
					print(link_flow)
				# End of test print.

				# Case 1) Assign departure flow to nodes.
				q = i
				h = t
				l = num_choices - 1
				tau = t

				for s in range(iteration_count_effective):
					for r in range(num_stops):

						if strategy_flow[q,r,h,s] > 0.0001:

							node_flow_key = (q,r,h,s,l,tau)
							if node_flow_key in node_flow:
								node_flow[node_flow_key] += strategy_flow[q,r,h,s]
							else:
								node_flow[node_flow_key] = strategy_flow[q,r,h,s]

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print()
								print("    Collecting flow - Case 1) assign departure flow to nodes - " + "node_flow" +\
								 str(node_flow_key) + ": " + str(strategy_flow[q,r,h,s]))
							# End of test print.

				# Case 2) & 3)
				# Iterate over link_flow.
				# If there is no incoming links for this node, these two cases will be skipped.

				to_be_deleted = set()

				for link_flow_key in link_flow:

					# If this is link flow that flows into node i_t, ...
					incoming_link = link_flow_key[6]
					ending_node = tn.links_exp[incoming_link][3]
					if ending_node == i_t:

						q = link_flow_key[0]
						r = link_flow_key[1]
						h = link_flow_key[2]
						s = link_flow_key[3]
						# Route of the incoming link of this flow.
						l_prime = link_flow_key[4]
						tau_prime = link_flow_key[5]
						
						# Route of this link.
						l = tn.links_exp[incoming_link][1]

						# Case 2) this (l) is a bus route, ...
						# tau is the time that user arrive at upstream node, range in [0, t - TT + 1],
						# namely, user can at most arrive at upstream node at time(t - incoming_links_tt[temp_index]).
						if l != (num_choices - 1):

							tau = t
							node_flow_key = (q,r,h,s,l,tau)
							if node_flow_key in node_flow:
								node_flow[node_flow_key] += link_flow[link_flow_key]
							else:
								node_flow[node_flow_key] = link_flow[link_flow_key]

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print()
								print("    (Computing return) collecting flow - Case 2): node_flow" + str(node_flow_key) + " get "\
								 + "incoming flow from transit link_flow" + str(link_flow_key) + ": "\
								 + str(link_flow[link_flow_key]))
							# End of test print.

						else:
						# Case 3) This is a waiting route.
						# Notice: the difference is that for a WT route,
						# the arrival time doesn't change!

							tau = tau_prime
							# Note that arrival time at i_t is tau.
							node_flow_key = (q,r,h,s,l,tau)

							if node_flow_key in node_flow:
								node_flow[node_flow_key] += link_flow[link_flow_key]
							else:
								node_flow[node_flow_key] = link_flow[link_flow_key]

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print()
								print("    (Computing return) collecting flow - Case 3): node_flow" + str(node_flow_key) + " get "\
								 +	"incoming flow from waiting link_flow" + str(link_flow_key) + ": "\
								 + str(link_flow[link_flow_key]))
							# End of test print.

						# Remenber to delete assigned link_flow.
						to_be_deleted.add(link_flow_key)

				for link_flow_key in to_be_deleted:
					del link_flow[link_flow_key]

				# ---- Assign flow to outgoing links according to assign prob vec.
				# This part is much simplier than before.

				# meas_prob_modify_flag[q,r,h] is used to flag whether this node prob has been modified at previous stages.
				# No need to record i_t.
				# This is added because od_flow_to_stop_prob was not zeroed for those od_flow > 0.0001;
				# hence there was need to notice whether it's the first time to update;
				# however, it's no lnger needed; anyway there is no harm to keep it.				
				meas_prob_modify_flag = np.zeros((num_stops, num_stops, T))

				if TEST_LOAD_STRATEGY_FLOW == 1:
					print()
					print("    node_flow:")
					print(node_flow)
				# End of test print.

				for node_flow_key in node_flow:

					q = node_flow_key[0]
					r = node_flow_key[1]
					h = node_flow_key[2]
					s = node_flow_key[3]
					l = node_flow_key[4]
					tau = node_flow_key[5]

					if TEST_LOAD_STRATEGY_FLOW == 1:
						print()
						print("    (Computing return) Assign flow - node_flow" + str(node_flow_key) + ": " + str(node_flow[node_flow_key]))
						print("    prefer_links" + str([r,s,l,tau,i_t]) + ":")
						print(prefer_links[r,s,l,tau,i_t])
						print("    prefer_probs" + str([r,s,l,tau,i_t]) + ":")
						print(prefer_probs[r,s,l,tau,i_t])
					# End of test print.
		
					# Assign according to the preference set.
					for temp_index in range(len(prefer_links[r,s,l,tau,i_t])):

						# If this link is used...
						if prefer_probs[r,s,l,tau,i_t][temp_index] >= 0.0001:
							outgoing_link = prefer_links[r,s,l,tau,i_t][temp_index]

							link_flow_key = (q,r,h,s,l,tau,outgoing_link)
							if link_flow_key in link_flow:
								link_flow[link_flow_key] += node_flow[node_flow_key] * prefer_probs[r,s,l,tau,i_t][temp_index]
							else:
								link_flow[link_flow_key] = node_flow[node_flow_key] * prefer_probs[r,s,l,tau,i_t][temp_index]

							if TEST_LOAD_STRATEGY_FLOW == 1:
								print()
								print(tn.links_exp_char[outgoing_link])
								print("    link_flow" + str(link_flow_key) + " updated to " + str(link_flow[link_flow_key]))
							# End of test print.

					# ---- Update measurement detection probabilities.

					# Entry count.
					if i == q and t == h:
						od_flow_to_stop_prob[q,r,h,i,t] = 1.0
						# Print
						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Entry count prob: od_flow_to_stop_prob" + str([q,r,h,i,t]) + " is 1.0")
						# End of test print.

					# Exit count.
					if i == r:

						if meas_prob_modify_flag[q,r,h] < 0.5:
							od_flow_to_stop_prob[q,r,h,i,t] = node_flow[node_flow_key] / od_flow[q,r,h]
							meas_prob_modify_flag[q,r,h] = 1
						else:
							od_flow_to_stop_prob[q,r,h,i,t] += node_flow[node_flow_key] / od_flow[q,r,h]

						# Print
						if TEST_LOAD_STRATEGY_FLOW ==  1:
							print()
							print("    Exit count prob: od_flow_to_stop_prob" + str([q,r,h,i,t]) + " is updated to " + str(od_flow_to_stop_prob[q,r,h,i,t]))
						# End of test print.

					# Passby count.
					# Exclude WT roure.
					if i!= q and i!= r and l != (num_choices - 1):

						if meas_prob_modify_flag[q,r,h] < 0.5:
							od_flow_to_stop_prob[q,r,h,i,t] = node_flow[node_flow_key] / od_flow[q,r,h]
							meas_prob_modify_flag[q,r,h] = 1
						else:
							od_flow_to_stop_prob[q,r,h,i,t] += node_flow[node_flow_key] / od_flow[q,r,h]

						if TEST_LOAD_STRATEGY_FLOW == 1:
							print()
							print("    Passby count prob: od_flow_to_stop_prob" + str([q,r,h,i,t]) + " is updated to " + str(od_flow_to_stop_prob[q,r,h,i,t]) )
						# End of test print.

			#np.save('results/dynamic/od_flow_to_stop_prob_lower_level_iteration_' + str(iteration_count), od_flow_to_stop_prob)
			np.save('data/od_flow_to_stop_prob', od_flow_to_stop_prob)
			np.save('data/link_combined_flow', link_combined_flow)

		# -------- Update flow by MSA steps {1/iteration_count}
		# Otherwise, set X[n] = 1/n [(n-1)X[n-1] + Y[n]]
		print()
		print("    --------------------------")
		if converg_flag == 0:

			print()
			print("    CONVERGENCE_CRITERION not met, next iteration begins ...")

			# Tell whether the optimal strategy generated above is the same with some previoud strategy.
			# Initializ the strategy_index_of_Y.
			strategy_index_of_Y = iteration_count

			# Tell whether the optimal strategy generated above is the same with some previoud strategy.
			for s in range(iteration_count_effective):
				coincidence_flag = 1

				for r in range(num_stops):
					for l in range(num_choices):
						
						for i_t in range(num_stops_exp):

							t = tn.stops_exp_2[i_t][2]

							temp = 0
							if l != (num_choices - 1):
								temp = t

							for tau in range(temp, t + 1):
								# If preference set at one [r,l,tau,i_t] is different.
								if prefer_links_optimal[r,l,tau,i_t] != prefer_links[r,s,l,tau,i_t]:
									coincidence_flag = 0

								if coincidence_flag == 0:
									break
							if coincidence_flag == 0:
								break
						if coincidence_flag == 0:
							break
					if coincidence_flag == 0:
						break

				# Continue to next strategy.
				if coincidence_flag == 0:
					continue
				elif coincidence_flag == 1:
					strategy_index_of_Y = s
					print()
					print("    This optimal strategy is the same with the " + str(s) + "-th stratrgy.")
					break

			if coincidence_flag == 0:
				print()
				print("    This optimal strategy doesn't coincide with any previous strategy.")
			else:
				print()
				print("    This optimal stratey coincide with the " + str(strategy_index_of_Y) + "-th strategy.")

			# Define the new direction Y
			print()
			print("    Defining descending direction Y...")
			for q in range(num_stops):
				for r in range(num_stops):
					for h in range(T):
						Y[q,r,h,strategy_index_of_Y] = od_flow[q,r,h]

			# You need to add new preference set of strategy if this strategy is new
			#  and will be stored.
			# If a new strategy is generated...
			if coincidence_flag == 0:
				for r in range(num_stops):
					for l in range(num_choices):
						for i_t in range(num_stops_exp):

							t = tn.stops_exp_2[i_t][2]

							temp = 0
							if l != (num_choices - 1):
								temp = t

							for tau in range(temp, t + 1):
								prefer_links[r,strategy_index_of_Y,l,tau,i_t] = deepcopy(prefer_links_optimal[r,l,tau,i_t])
								prefer_probs[r,strategy_index_of_Y,l,tau,i_t] = deepcopy(prefer_probs_optimal[r,l,tau,i_t])

			# Use MSA to update the strategy flow.
			strategy_flow = 1/(iteration_count + 1) * (iteration_count * strategy_flow + Y)

			iteration_count += 1

			if coincidence_flag == 0:
				iteration_count_effective += 1

	return od_flow_to_stop_prob, link_combined_flow