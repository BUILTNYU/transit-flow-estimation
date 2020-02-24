# Bi-level model
# Diagnolization method used to handle interaction effect;
# Spiess' algorithm model used to assign users to links for fixed cost;
# MSA used for step size.

# Upper level is quadratic programming
# Variables:
# [...q_rs..., ...O_r..., ...D_s..., ...x_i...]
# Lower level is a Variational Inequality problem
#
# Prepare coefficients for solve quadratic programming:
#    min 1/2x^TQx + p^Tx
# s.t. Ax = b
#	   Gx <= h

# Variables and formats:
# od_flow_vector_to_stop_prob:
# proportion of flow to nodes (matrix, numpy, shape: num_stops * num_ods matrix)
# Illustration:
#     p^(1,1)_1,p^(1,2)_1,...p^(2,1)_1...
#     p^(1,1)_2,p^(1,2)_2,...p^(2,1)_2...
#     ...
# where p^(r,s)_i means the flow from r to s assigned to node i;
# --------------
# count files, including entry, exit, wifi, passby count 
# (table, numpy, shape: num_stops * 2)
# 0       1
# stop_id,count
# --------------
# od_flow_vector: (1 dim array, numpy, shape: num_od_raw * 1)
#
# od_flow[r,s]: (matrix, numpy, shape: num_stops * num_stops)
#
# Dimension descriptions:
# r       s
# origin, destination
# --------------
# Q: parameter in quadratic programming
# p: parameter in quadratic programming
# G: parameter in quadratic programming
# h: parameter in quadratic programming
# A: parameter in quadratic programming
# b: parameter in quadratic programming
#
# i: the index for node (stop)

import sys
import numpy as np
from cvxopt import matrix, solvers

from open_data_file_with_header import open_data_file_with_header
from open_config_file import open_config_file
from import_covar_matrix import import_covar_matrix
from static_assignment_algorithm import static_assignment_algorithm

def static_model(sn):
	"""Input StaticNetwork obj; output UE assignment results into files."""

	# sn: static network object.

	print('    ------------------------------------------------------------------------------')
	print("    Static model begins ...")

	# Total number of stops
	num_stops = len(sn.stops)
	num_stops_exp = len(sn.stops_exp) # 'exp' means expanded
	num_ods = num_stops ** 2
	num_links = len(sn.links)
	num_links_exp = len(sn.links_exp)

	MAX_NUMBER_OF_ITERATIONS = int(open_config_file('MAX_NUMBER_OF_ITERATIONS_FOR_BILEVEL_MODEL'))
	CONVERGENCE_CRITERION = float(open_config_file('CONVERGENCE_CRITERION_FOR_BILEVEL_MODEL'))
	TEST_STATIC_MODEL =  float(open_config_file('test_static_model'))
	TRANSIT_TYPE =  open_config_file('TRANSIT_TYPE')
	
	od_flow = np.zeros((num_stops,num_stops))
	od_flow_vector = np.zeros(num_ods)

	# Input Data
	# Count data and covariance matrices.
	entry_count = np.zeros((num_stops,1))
	exit_count = np.zeros((num_stops,1))
	wifi_count = np.zeros((num_stops,1))
	passby_count = np.zeros((num_stops,1))

	# Retain only data column.
	entry_count_file = open_data_file_with_header('data/entry_count_by_hour.csv')
	exit_count_file = open_data_file_with_header('data/exit_count_by_hour.csv')
	wifi_count_file = open_data_file_with_header('data/wifi_count_by_hour.csv')
	
	for i_stop in range(num_stops):
		entry_count[i_stop,0] = float(entry_count_file[i_stop][1])
		exit_count[i_stop,0] = float(exit_count_file[i_stop][1])
		wifi_count[i_stop,0] = float(wifi_count_file[i_stop][1])

	# Convert wifi count to stop count
	wifi_sample_ratio = open_data_file_with_header('data/wifi_sample_ratio.csv')
	# File header: 
	# 0
	# sample_ratio
	wifi_sample_ratio = [item[0] for item in wifi_sample_ratio]

	passby_count = np.zeros((num_stops,1))
	for i_stop in range(num_stops):
		passby_count[i_stop,0] = wifi_count[i_stop,0] / float(wifi_sample_ratio[i_stop])
	
	if TEST_STATIC_MODEL == 1:
		print("    entry_count:")
		print(entry_count)
		print("    exit_count:")
		print(exit_count)
		print("    passby_count:")
		print(passby_count)

	# Prepare coefficients for solve quadratic programming:

	# The composition of decision variable vector:
	# [od_flow_vector  enter_flow  exit_flow  stop_flow]
	# which correspond to follwoing symboles in the paper:
	# [q        O           D          x        ]

	# Measurements will be denoted by:
	# [od_prior  entry_count  exit_count  passby_count]
	# dimension = num_ods + 3 * num_stops

	# Coefficients: Q,p,A,b

	# Q
	Q = np.zeros((num_ods + 3 * num_stops, num_ods + 3 * num_stops))
	for temp_index in range(num_ods, num_ods + 3 * num_stops):
		Q[temp_index, temp_index] = 2

	# p
	temp = np.zeros((num_ods,1))
	p = (-2) * np.concatenate((temp, entry_count, exit_count, passby_count), axis=0)
	del temp, entry_count, exit_count, wifi_count, passby_count

	# A
	A11 = np.zeros((num_stops,num_ods))
	for n in range( num_stops):
		A11[n, n * num_stops : (n + 1) * num_stops] = np.ones((1,num_stops))
	A12 = (-1) * np.identity(num_stops)
	A13 = np.zeros((num_stops,num_stops))
	A14 = np.zeros((num_stops,num_stops))
	A1 = np.concatenate((A11, A12, A13, A14), axis=1)
	del A11, A12, A13, A14
	
	# A21
	A21 = np.zeros((num_stops,num_ods))
	for n1 in range(num_stops):
		for n2 in range(num_stops):
			A21[n1, num_stops * n2 + n1] = 1
	A22 = np.zeros((num_stops,num_stops))
	A23 = (-1) * np.identity(num_stops)
	A24 = np.zeros((num_stops,num_stops))
	A2 = np.concatenate((A21, A22, A23, A24), axis=1)
	del A21, A22, A23, A24

	A1_A2 = np.concatenate((A1, A2), axis=0)
	del A1, A2

	# A31 will change during each iteration
	# A31 = od_flow_vector_to_stop_prob
	A32 = np.zeros((num_stops,num_stops))
	A33 = np.zeros((num_stops,num_stops))
	A34 = (-1) * np.identity(num_stops)

	A32_A33_A34 = np.concatenate((A32, A33, A34), axis=1)
	del A32, A33, A34
	
	# b
	b = np.zeros(( 3 * num_stops, 1))

	# G
	G = (-1) * np.identity(num_ods + 3 * num_stops)

	# h
	h = np.zeros((num_ods + 3 * num_stops, 1))

	# Transform numpy matrices to cvxopt matrices
	Q = matrix(Q, tc='d')
	p = matrix(p, tc='d')
	G = matrix(G, tc='d')
	h = matrix(h, tc='d')
	b = matrix(b, tc='d')

	print()
	print('    Bi-level programming iteration begins...')

	# Bi-level programming iterations
	# Initialization
	iteration_count = 1
	converg_flag = 0

	# While convergence or maximum, iteration not reached, continue to loop
	while converg_flag == 0:
		print()
		print('    The ' + str(iteration_count) + '-th iteration of bi-level programming begins.')

		print()
		print('    Upper level begins ...')

		# Initialize the cost of network obj.
		for a_link in range(len(sn.links_exp)):
			if sn.links_exp[a_link][4] == TRANSIT_TYPE:
				sn.links_exp[a_link][8] = sn.links_exp[a_link][5]

		# Upper level
		if iteration_count == 1:


			# Initialization
			# Input very small od_flow to get od_flow_vector_to_stop_prob.
			od_flow = np.ones((num_stops, num_stops))
			od_flow_vector_to_stop_prob = static_assignment_algorithm(sn, od_flow)

		# Adopt od_flow_vector_to_stop_prob matrix from assignment of last iteration.
		# Retain only these probabilities for departure and transfer appearences (exculde
		# arrivals).
		A31 = np.zeros((num_stops,num_ods))
		for r_origin in range(num_stops):
			for s_dest in range(num_stops):
				for i_stop in range(num_stops):
					if i_stop != s_dest:
						A31[i_stop, num_stops * r_origin + s_dest] = od_flow_vector_to_stop_prob[r_origin,s_dest,i_stop]

		A3 = np.concatenate((A31, A32_A33_A34), axis=1)
		# A = np.concatenate((A1, A2, A3), axis=0)
		# A will change when A31 changes
		A = np.concatenate((A1_A2, A3), axis=0)

		# Transform numpy matrix to cvxopt matrix
		A = matrix(A, tc='d')
		
		print()
		print('    Quadratic programming solvers begins.')
		sol = solvers.qp(Q, p, G, h, A, b)
		print('    Solvers finished once.')
		print('    status:')
		print(sol['status'])
		# 'sol' is dictionary
		# Key: 'status', 'x', 'primal objective'

		if sol['status'] == 'optimal':
			od_flow_vector = sol['x'][0 : num_ods]

			print('    The optimal od_flow_vector is: (length: ' + str(len(od_flow_vector)) + ')')
			print(od_flow_vector)

		else:
			print('    Error: no optimal solution found!')
			# Stop iteration
			converg_flag = 1


		# Convert od_flow_vector to  od_flow matrix first, then use latter as input.
		# Note: this act is aimed at uniforming the data exchanging format with other modules,
		# like time-dependent module.
		for r_origin in range(num_stops):
			for s_dest in range(num_stops):
				od_flow[r_origin,s_dest] = od_flow_vector[r_origin * num_stops + s_dest]

		# Save files.
		np.save("results/static/od_flow_iteration_" + str(iteration_count), od_flow)

		# Lower level
		print('    Lower level begins...')

		# Apply static assignment model.
		# Input static network, od flow.
		od_flow_vector_to_stop_prob = static_assignment_algorithm(sn, od_flow)

		# Convergence test
		if iteration_count >= 2:
			avg_abs_relative_change = 0

			for r_origin in range(num_stops):
				for s_dest in range(num_stops):
					if (od_flow_last[r_origin,s_dest] + od_flow[r_origin,s_dest]) != 0:
						avg_abs_relative_change += abs(od_flow_last[r_origin,s_dest] - od_flow[r_origin,s_dest])\
						 / ((od_flow_last[r_origin,s_dest] + od_flow[r_origin,s_dest]) / 2)

			avg_abs_relative_change /= num_ods
			print("    (Upper level) Iteration: " + str(iteration_count) + " avg_abs_relative_change:" + str(avg_abs_relative_change))

			if avg_abs_relative_change < CONVERGENCE_CRITERION:
				print("    Upper level CONVERGENCE_CRITERION met!")
				np.save("results/static/od_flow", od_flow)
				converg_flag = 1
			
		od_flow_last = od_flow

		if iteration_count >= MAX_NUMBER_OF_ITERATIONS:
			print('    Warning! bi-level problem not converging at ' + str(MAX_NUMBER_OF_ITERATIONS) + 'th iteration!')
			sys.exit(1)

		iteration_count += 1
			
	return od_flow