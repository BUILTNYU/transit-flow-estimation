import numpy as np
import random

from TimeExpandedNetwork import TimeExpandedNetwork
from open_config_file import open_config_file
from find_initial_strategy import find_initial_strategy
from dynamic_schedule_based_ue_assignment_algorithm import dynamic_schedule_based_ue_assignment_algorithm
from dynamic_schedule_based_ue_assignment_algorithm_with_noises import dynamic_schedule_based_ue_assignment_algorithm_with_noises

def dynamic_schedule_based_ue_assignment_algorithm_run():

	tn = TimeExpandedNetwork()

	# Option 1)Input manually
	# Some parameters
	TEST_LOADED_OD_FLOW = int(open_config_file('test_loaded_od_flow'))
	TEST_PROB_INITIALIZATION = int(open_config_file('test_prob_initialization'))

	MAX_NUMBER_OF_ITERATIONS = int(open_config_file('MAX_NUMBER_OF_ITERATIONS_FOR_ASSIGNMENT_MODEL'))
	DOUBLE_STREAMLINED = int(open_config_file('DOUBLE_STREAMLINED'))
	ADD_NOISE = int(open_config_file('ADD_NOISE'))

	num_strategies = MAX_NUMBER_OF_ITERATIONS
	# Number of stops.
	num_stops = len(tn.stops)
	N = len(tn.stops)
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

	print()
	print("    T: " + str(T))
	print("    num_stops: " + str(num_stops))
	print("    num_stops_exp: " + str(num_stops_exp))
	print("    num_links_exp: " + str(num_links_exp))
	print("    num_routes: " + str(num_routes))
	print("    num_strategies: " + str(num_strategies))
	print("    add noise: " + str(ADD_NOISE))

	print("    Find initial od_flow_to_stop_prob...")

	# Initialize the cost of links
	for index in range(len(tn.links_exp)):
		tn.links_exp[index][8] = tn.links_exp[index][5]

	# Initialize od_flow_to_stop_prob.
	# Users will follow the path determined by the initial preference set.
	# No capacity constraints etc. considered.
	prefer_links_optimal, prefer_probs_optimal = find_initial_strategy(tn)

	od_flow_to_stop_prob = np.zeros((N,N,T,N,T))

	for q_origin in range(N):
		for r_dest in range(N):

			if r_dest != q_origin:
				for h_depart in range(T):

					# Initialize departure node in TE network.
					# Notations inherated from dynamic_schedule_based_ue_assignment_algorithm.
					i = q_origin
					t = h_depart
					i_t = tn.stops_exp.index(tn.stops[q_origin] + '_' + str(h_depart))
					# coming from route
					l = num_choices - 1
					tau = h_depart

					arrive_dest_flag = 0
					while arrive_dest_flag == 0:

						if TEST_PROB_INITIALIZATION == 1:
							print()
							print("    Current [q,r,h]: " + str([q_origin,r_dest,h_depart]) + " current node: " + str(tn.stops_exp[i_t]) + " - " + str(i_t))
							print("    prefer_links_optimal:")
							print(prefer_links_optimal[r_dest, l, tau, i_t])
							print("    prefer_stops_optimal:")
							print(prefer_stops_optimal[r_dest, l, tau, i_t])

						# Update od_flow_to_stop_prob for current node.
						if q_origin != r_dest :

							if i == q_origin and t == h_depart:
								od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

								if TEST_PROB_INITIALIZATION == 1:
									print("    Entry count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updatd to 1.")

							if i == r_dest:
								od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

								if TEST_PROB_INITIALIZATION == 1:
									print("    Exit count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updatd to 1.")

							if i != q_origin and i != r_dest and l != (num_choices -1):
								od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

								if TEST_PROB_INITIALIZATION == 1:
									print("    Passby count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updatd to 1.")

						# Update arrive_dest_flag if needed.
						if i == r_dest:
							arrive_dest_flag == 1

							if TEST_PROB_INITIALIZATION == 1:
								print("    Arrive at destination.")

							break

						# Find next node.
						# If preference set not empty, which means it's able to get to the destination in T;
						if prefer_links_optimal[r_dest, l, tau, i_t]:
							link_next = prefer_links_optimal[r_dest, l, tau, i_t][0]
							l_next = tn.links_exp[link_next][1]
							i_t_next = tn.links_exp[link_next][3]
							i_next = tn.stops_exp_2[i_t_next][1]
							t_next = tn.stops_exp_2[i_t_next][2]
							
							if l_next == (num_choices - 1):
								tau_next = tau
							else:
								# Use TT to update tau.
								tau_next = t_next

						else:
							if TEST_PROB_INITIALIZATION == 1:
								print("    This node cannot reach destination within horizon.")
							break

						# Update node.
						i_t = i_t_next
						i = i_next
						t = t_next
						l = l_next
						tau = tau_next

	del prefer_links_optimal, prefer_probs_optimal

	# Option 1) Manual Input
	od_flow = np.zeros((num_stops, num_stops, T))

	od_flow[0,4,0] = 150
	od_flow[1,4,0] = 150

	# Option 2) Import
	if TEST_LOADED_OD_FLOW == 1:
		od_flow = np.load('data/od_flow.npy')

	print("    Total flow: " + str(od_flow.sum()))

	if ADD_NOISE == 0:
		od_flow_to_stop_prob, link_combined_flow = dynamic_schedule_based_ue_assignment_algorithm(tn, od_flow, od_flow_to_stop_prob)
	else:
		od_flow_to_stop_prob, link_combined_flow = dynamic_schedule_based_ue_assignment_algorithm_with_noises(tn, od_flow, od_flow_to_stop_prob)

