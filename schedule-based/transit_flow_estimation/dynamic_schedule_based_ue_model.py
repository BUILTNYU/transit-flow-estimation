# Bi-level algorithm
#
# Upper level is quadratic programming;
# Lower level is dynamic schedual-based UE assignment (variational inequality problem);
# For lower level, diagnolization method used to handle interaction effect;
# Hamdouch' algorithm model used to assign users to links for fixed cost;
# MSA used for step size;
#
# Notes:
# 1) All types of measurements will be assumed to have the same periods for simplicity;
# 2) Only entering and transfer flows could possibly be detected by wifi;
# 3) No covariance matrix is used in upper level. This model could support 
#    covariance matrix; it's neglected just for simplicity;
# 4) In static model, r and s are used to denote OD pair, h_depart for departure time; 
#    in time-dependent UE model, q and r will be used to denote OD pair, h for departure time,
#    as used by the paper;
# 5) Short variables names are used in this model, which is bad, just to keep code neat.

# Quadratic pogramming 
# Variables:
# (The composition of decision variable vector)
#
# [od_flow  enter_flow  exit_flow  stop_flow]
# [q             O           D          x     ]
# [..q_rsh_depart..,..O_r... , ...Ds... , ...xi... ]

# Formulation
#    min 1/2x^TQx + p^Tx
# s.t. Gx <= h
#      Ax = b

# Notations
# Q: parameter in quadratic programming
# p: parameter in quadratic programming
# G: parameter in quadratic programming
# h: parameter in quadratic programming
# A: parameter in quadratic programming
# b: parameter in quadratic programming
#
# r: index for origins
# s: idnex for destinations
# h_depart : index for departure time
# n: index for stop
# N: number of stops
# t: index for time
# T: horizon (Unit: min)
# i: index for measurements
# I: number of measurements (I = T / C)
# C: period of measurements (Unit: min) 
# 'tn' means time-expanded network obj
#
# od_flow[q,r,h_depart]: od flow
# (3 dim numpy array)
# --------------
# od_flow_to_stop_prob[r,s,h_depart,i,t]: proportion of flow appearing at nodes at specific times
# (5 dim numpy array, shape: (N,N,T,N,T))
# --------------
# Count files, including entry, exit, wifi, passby count: 
# (numpy vector, shape: (N*I,1))

import sys
import platform
import numpy as np
from copy import deepcopy
try:
	from cvxopt import matrix, solvers
except ImportError:
	pass
try:
	from gurobipy import *
except ImportError:
	pass
from open_data_file_with_header import open_data_file_with_header
from open_config_file import open_config_file
from find_initial_strategy import find_initial_strategy
from dynamic_schedule_based_ue_assignment_algorithm import dynamic_schedule_based_ue_assignment_algorithm

def dynamic_schedule_based_ue_model(tn):
	""" Input TimeExpandedNetwork obj; output the time-dependent equilibrium flow."""

	print('    ---------------------------------------------------')
	print("    Time-dependent model begins ...")

	# Parameters
	TEST_QUADRATIC_PROG = int(open_config_file('test_quadratic_programming'))
	QUADRATIC_PROG_SOLVING_PKG = open_config_file('QUADRATIC_PROG_SOLVING_PKG')
	INTEGER_PROG = float(open_config_file('INTEGER_PROG'))
	
	# Number of stops
	N = len(tn.stops)

	# Horizon
	T = tn.T
	MAX_NUMBER_OF_ITERATIONS = int(open_config_file('MAX_NUMBER_OF_ITERATIONS_FOR_BILEVEL_MODEL'))
	CONVERGENCE_CRITERION = float(open_config_file('CONVERGENCE_CRITERION_FOR_BILEVEL_MODEL'))
	
	# Number of bus lines.
	num_routes = len(tn.routes)

	# The most number of choices user could have at a node, which equals
	# the number of routes, adding a waiting link.
	num_choices = num_routes + 1

	# Periods of measurements (Unit: min)
	C = int(open_config_file('MEASUREMENT_PERIOD'))
	
	# Times of measurements
	I = int(T / C)

	sys.setrecursionlimit(5000000)

	global m

	print()
	print("    Horion (T): " + str(T))
	print("    Measurements period C: " + str(C))
	print("    Times of measurements I: " + str(I))
	print("    Optimization package: " + QUADRATIC_PROG_SOLVING_PKG)

	# Initialization
	od_flow = np.zeros((N,N,T))

	# Input Data
	# Count data.
	entry_count = []
	exit_count = []
	wifi_count = []
	passby_count = []
	# Retain only data column.
	entry_count_file = open_data_file_with_header('data/entry_count_by_designated_period.csv')
	exit_count_file = open_data_file_with_header('data/exit_count_by_designated_period.csv')
	wifi_count_file = open_data_file_with_header('data/wifi_count_by_designated_period.csv')
	# Data formats:
	# (col) 1                       2        3
	#       measurement_seq_number, stop_id, count
	
	# Retain only count data.
	entry_count = [float(item[2]) for item in entry_count_file]
	exit_count = [float(item[2]) for item in exit_count_file]
	wifi_count = [float(item[2]) for item in wifi_count_file]

	# Convert wifi count to stop count.

	# i) First, import ratio file.
	wifi_sample_ratio = open_data_file_with_header('data/wifi_sample_ratio.csv')
	# Table header: 
	# 0
	# sample_ratio
	wifi_sample_ratio = [float(item[0]) for item in wifi_sample_ratio]

	# ii) Second, transform.
	passby_count = []
	for k_meas in range(I):
		for i_stop in range(N):
			# Note: wifi_count is list
			passby_count.append(wifi_count[k_meas*N + i_stop] / wifi_sample_ratio[i_stop])
	
	# Print measurements data.
	if TEST_QUADRATIC_PROG == 1:
		print()
		print('    Measurements data:')
		print('    entry_count:')
		print(entry_count)
		print('    exit_count:')
		print(exit_count)
		print('    wifi_count:')
		print(wifi_count)
		print('    wifi_sample_ratio:')
		print(wifi_sample_ratio)
		print('   passby_count:')
		print(passby_count)
	# End of test print.

	# Prepare coefficients for solve quadratic programming

	# Non-integer prog
	if QUADRATIC_PROG_SOLVING_PKG == "cvxopt":
		# Coefficients: Q,p,A,b,G,h
		# Q & p
		Q = np.zeros((N*N*T + 3*N*T, N*N*T + 3*N*T))
		p = np.zeros((N*N*T + 3*N*T, 1))

		# period k_meas
		for k_meas in range(I):
			# stop n
			for i_stop in range(N):
				for t1 in range(C*k_meas, C*(k_meas+1)):
					for t2 in range(C*k_meas, C*(k_meas+1)):
						Q[N*N*T + N*t1 + i_stop, N*N*T + N*t2 + i_stop] = 2
						Q[N*N*T + N*T + N*t1 + i_stop, N*N*T + N*T + N*t2 + i_stop] = 2
						Q[N*N*T + 2*N*T + N*t1 + i_stop, N*N*T + 2*N*T + N*t2 + i_stop] = 2
					# p
					p[N*N*T + N*t1 + i_stop, 0] = (-2) * entry_count[N*k_meas + i_stop]
					p[N*N*T + N*T + N*t1 + i_stop, 0] = (-2) * exit_count[N*k_meas + i_stop]
					p[N*N*T + 2*N*T + N*t1 + i_stop, 0] = (-2) * passby_count[N*k_meas + i_stop]

		# A
		# A will be fully determined in the iterations.
		# Initialized here.
		A = np.zeros((3*N*T, N*N*T + 3*N*T))
		
		# b is zero as default.
		b = np.zeros((3*N*T, 1))

		# G
		# var >= 0, namely - var <= 0.
		G = (-1) * np.identity(N*N*T + 3*N*T)

		# h is zero as default.
		h = np.zeros((N*N*T + 3*N*T, 1))

		# Transform numpy matrices to cvxopt matrices.
		Q = matrix(Q, tc='d')
		p = matrix(p, tc='d')
		G = matrix(G, tc='d')
		h = matrix(h, tc='d')
		b = matrix(b, tc='d')

	# Mixed integer prog.
	# Sparse model.
	elif QUADRATIC_PROG_SOLVING_PKG == "gurobi" and platform.system() != 'Windows':
		
		m = Model("qp")

		# Create variables
		# f_i_j_h
		for q_origin in range(N):
			for r_dest in range(N):
				for h_depart in range(T):
					if INTEGER_PROG == 1:
						exec("f_%d_%d_%d = m.addVar(vtype=GRB.INTEGER, lb=0, name='f_%d_%d_%d')" %(q_origin, r_dest, h_depart, q_origin, r_dest, h_depart), globals())
					elif INTEGER_PROG == 0:
						exec("f_%d_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='f_%d_%d_%d')" %(q_origin, r_dest, h_depart, q_origin, r_dest, h_depart), globals())
		# o_q_h
		for q_origin in range(N):
			for h_depart in range(T):
				exec("o_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='o_%d_%d')" %(q_origin, h_depart, q_origin, h_depart), globals())

		# d_r_t
		for r_dest in range(N):
			for t in range(T):
				exec("d_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='d_%d_%d')" %(r_dest, t, r_dest, t), globals())

		# x_i_t
		for i_stop in range(N):
			for t in range(T):
				exec("x_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='x_%d_%d')" %(i_stop, t, i_stop, t), globals())

		# Set objective
		obj_str = 'obj = '

		for k_meas in range(I):

			# o_q_h
			for q_origin in range(N):

				temp_str = '('
				for h_depart in range(C*k_meas, C*(k_meas+1)):

					temp_str += 'o_' + str(q_origin) + '_' + str(h_depart)

					if h_depart != C*(k_meas+1) - 1:
						temp_str += '+'

				temp_str += ' - ' + str(entry_count[N*k_meas + q_origin])
				temp_str += ')'

				obj_str += temp_str + '*' + temp_str

				obj_str += ' + '

			# d_r_t
			for r_dest in range(N):

				temp_str = '('
				for t in range(C*k_meas, C*(k_meas+1)):

					temp_str += 'd_' + str(r_dest) + '_' + str(t)

					if t != C*(k_meas+1) - 1:
						temp_str += '+'

				temp_str += ' - ' + str(exit_count[N*k_meas + r_dest])
				temp_str += ')'

				obj_str += temp_str + '*' + temp_str

				obj_str += ' + '

			# x_i_t
			for i_stop in range(N):

				temp_str = '('
				for t in range(C*k_meas, C*(k_meas+1)):

					temp_str += 'x_' + str(i_stop) + '_' + str(t)

					if t != C*(k_meas+1) - 1:
						temp_str += '+'

				temp_str += ' - ' + str(passby_count[N*k_meas + i_stop])
				temp_str += ')'

				obj_str += temp_str + '*' + temp_str

				if not(i_stop == (N - 1) and k_meas == (I - 1)):
					obj_str += ' + '

		if TEST_QUADRATIC_PROG == 1:
			print()
			print("    objective is:")
			print(obj_str)
		# End of test print.

		exec(obj_str, globals())

		m.setObjective(obj, GRB.MINIMIZE)

		# Add constraints (part).
		count_constraint = 0

		# o_q_h
		for q_origin in range(N):
			for h_depart in range(T):

				const_str = ''
				temp_count = 0
				
				for r_dest in range(N):
					if r_dest != q_origin:

						if temp_count != 0:
							const_str += ' + '

						const_str += 'f_' + str(q_origin) + '_' + str(r_dest) + '_' + str(h_depart)
						temp_count += 1

				if temp_count > 0:
					const_str += ' == '
					const_str += 'o_' + str(q_origin) + '_' + str(h_depart)

					if TEST_QUADRATIC_PROG == 1:
						print()
						print("    A entry count constraint is added: ")
						print(const_str)
					# End of test print.
				
					m.addConstr(eval(const_str), 'c' + str(count_constraint))
					count_constraint += 1

		# d_r_t
		# TBD

		# x_r_t
		# TBD
	elif QUADRATIC_PROG_SOLVING_PKG == "gurobi" and platform.system() == 'Windows':
		pass

	elif QUADRATIC_PROG_SOLVING_PKG != "cvxopt" and QUADRATIC_PROG_SOLVING_PKG != "gurobi":
		print()
		print("    Error: Unkown method!")
		sys.exit(1)

	print('    --------------------------')
	print('    Bi-level programming iterations begins ... ')

	# Bi-level programming iterations
	# Initialization
	iteration_count = 0
	converg_flag = 0

	# While convergence or maximum, iteration not reached, continue to loop.
	while converg_flag == 0:

		# Upper level
		print()
		print("    Bi-level iteration: " + str(iteration_count))
		print('    Upper level begins ...')

		print("    Find initial od_flow_to_stop_prob...")

		# Initialize the cost of links
		for index in range(len(tn.links_exp)):
			tn.links_exp[index][8] = tn.links_exp[index][5]

		# Initialize od_flow_to_stop_prob.
		# Users will follow the path determined by the initial preference set.
		# No capacity constraints etc. considered.
		if iteration_count == 0:

			prefer_links_optimal, prefer_probs_optimal = find_initial_strategy(tn)

			od_flow_to_stop_prob = np.zeros((N,N,T,N,T))

			for q_origin in range(N):
				for r_dest in range(N):

					if q_origin != r_dest:
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

								if TEST_QUADRATIC_PROG == 1:
									print()
									print("    Current [q,r,h]: " + str([q_origin,r_dest,h_depart]) + " current node: " + str(tn.stops_exp[i_t]) + " - " + str(i_t))
									print("    prefer_links_optimal:")
									print(prefer_links_optimal[r_dest, l, tau, i_t])
								# End of test print.

								# Update od_flow_to_stop_prob for current node.
								if i == q_origin and t == h_depart:
									od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

									if TEST_QUADRATIC_PROG == 1:
										print()
										print("    Entry count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updated to 1.")
									# End of test print.

								if i == r_dest:
									od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

									if TEST_QUADRATIC_PROG == 1:
										print()
										print("    Exit count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updated to 1.")
									# End of test print.

								if i != q_origin and i != r_dest and l != (num_choices -1):
									od_flow_to_stop_prob[q_origin,r_dest,h_depart,i,t] = 1.0

									if TEST_QUADRATIC_PROG == 1:
										print()
										print("    Passby count od_flow_to_stop_prob" + str([q_origin,r_dest,h_depart,i,t]) + " updated to 1.")
									# End of test print.

								# Update arrive_dest_flag if needed.
								if i == r_dest:
									arrive_dest_flag == 1

									if TEST_QUADRATIC_PROG == 1:
										print()
										print("    Arrive at destination.")
									# End of test print.

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
									if TEST_QUADRATIC_PROG == 1:
										print()
										print("    This node cannot reach destination within horizon.")
									# End of test print.

									break

								# Update node.
								i_t = i_t_next
								i = i_next
								t = t_next
								l = l_next
								tau = tau_next

		if QUADRATIC_PROG_SOLVING_PKG == "cvxopt":
			# A
			# Adopt od_flow_to_stop_prob matrix from assignment of last iteration to obtain A.
			# For enter measurement at node r_h_depart
			for h_depart in range(T):
				for q_origin in range(N):

					# flow var coeff
					for r_dest in range(N):
						A[N*h_depart + q_origin, N*N*h_depart + N*q_origin + r_dest] = 1

					# Measurement var (O) coeff
					A[N*h_depart + q_origin, N*N*T + N*h_depart + q_origin] = -1

			# For exit measurement at node s_t
			for t_exit in range(T):
				for r_dest in range(N):

					#  flow var coeff
					for q_origin in range(N):
						for h_depart in range(t_exit):
							if q_origin != r_dest:
								A[N*T + N*t_exit + r_dest, N*N*h_depart + N*q_origin + r_dest] = od_flow_to_stop_prob[q_origin,r_dest,h_depart,r_dest,t_exit]

					# Measurement var (D) coeff
					# Remember to use t!
					A[N*T + N*t_exit + r_dest, N*N*T + N*T + N*t_exit + r_dest] = -1

			# For passby measurement at node n_t
			for i_stop in range(N):
				for t_passby in range(T):

					# Q
					# Note that enter flows could also be detedted by wifi, hence here 
					# h_depart range in 0 ~ t.
					for h_depart  in range(t_passby):
						for q_origin in range(N):
							for r_dest in range(N):

								if i_stop != q_origin and i_stop != r_dest:
									A[2*N*T + N*t_passby + i_stop, N*N*h_depart + N*q_origin + r_dest] = od_flow_to_stop_prob[q_origin,r_dest,h_depart,i_stop,t_passby]

					# Measuremnt var (X) coeff
					# Remember to use t!
					A[2*N*T + N*t_passby  + i_stop, N*N*T + 2*N*T + N*t_passby + i_stop] = -1

			print()
			print('    Quadratic programming solver begins ...')

			# Transform numpy matrix to cvxopt matrix
			A = matrix(A, tc='d')

			sol = solvers.qp(Q, p, G, h, A, b)

			print()
			print('    Solver finished once!')
			print('    Slover status:')
			print(sol['status'])
			# 'sol' is dictionary
			# Key: 'status', 'x', 'primal objective'

			if sol['status'] == 'optimal':
				print()
				optimal_objective = sol['primal objective']

				# Add constants neglected in optimization.
				for k_meas in range(I):
					for i_stop in range(N):
						optimal_objective += entry_count[N*k_meas + i_stop] ** 2
						optimal_objective += exit_count[N*k_meas + i_stop] ** 2
						optimal_objective += passby_count[N*k_meas + i_stop] ** 2

				print("Optimal objective " + str(optimal_objective))

				for q_origin in range(N):
					for r_dest in range(N):
						for h_depart  in range(T):
							od_flow[q_origin,r_dest,h_depart] = sol['x'][N*N*h_depart  + N*q_origin + r_dest]
			else:
				print()
				print('    Error: no optimal solution found!')

				# Stop iteration
				sys.eixt(1)

		# Sparse matrix
		elif QUADRATIC_PROG_SOLVING_PKG == "gurobi":

			# If the platform is Win, objective and constraints should be added.
			if platform.system() == 'Windows':
				
				m = Model("qp")

				# Create variables
				# f_i_j_h
				for q_origin in range(N):
					for r_dest in range(N):
						for h_depart in range(T):
							if INTEGER_PROG == 1:
								exec("f_%d_%d_%d = m.addVar(vtype=GRB.INTEGER, lb=0, name='f_%d_%d_%d')" %(q_origin, r_dest, h_depart, q_origin, r_dest, h_depart), globals())
							elif INTEGER_PROG == 0:
								exec("f_%d_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='f_%d_%d_%d')" %(q_origin, r_dest, h_depart, q_origin, r_dest, h_depart), globals())
				# o_q_h
				for q_origin in range(N):
					for h_depart in range(T):
						exec("o_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='o_%d_%d')" %(q_origin, h_depart, q_origin, h_depart), globals())

				# d_r_t
				for r_dest in range(N):
					for t in range(T):
						exec("d_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='d_%d_%d')" %(r_dest, t, r_dest, t), globals())

				# x_i_t
				for i_stop in range(N):
					for t in range(T):
						exec("x_%d_%d = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='x_%d_%d')" %(i_stop, t, i_stop, t), globals())

				# Set objective
				obj_str = 'obj = '

				for k_meas in range(I):

					# o_q_h
					for q_origin in range(N):

						temp_str = '('
						for h_depart in range(C*k_meas, C*(k_meas+1)):

							temp_str += 'o_' + str(q_origin) + '_' + str(h_depart)

							if h_depart != C*(k_meas+1) - 1:
								temp_str += '+'

						temp_str += ' - ' + str(entry_count[N*k_meas + q_origin])
						temp_str += ')'

						obj_str += temp_str + '*' + temp_str

						obj_str += ' + '

					# d_r_t
					for r_dest in range(N):

						temp_str = '('
						for t in range(C*k_meas, C*(k_meas+1)):

							temp_str += 'd_' + str(r_dest) + '_' + str(t)

							if t != C*(k_meas+1) - 1:
								temp_str += '+'

						temp_str += ' - ' + str(exit_count[N*k_meas + r_dest])
						temp_str += ')'

						obj_str += temp_str + '*' + temp_str

						obj_str += ' + '

					# x_i_t
					for i_stop in range(N):

						temp_str = '('
						for t in range(C*k_meas, C*(k_meas+1)):

							temp_str += 'x_' + str(i_stop) + '_' + str(t)

							if t != C*(k_meas+1) - 1:
								temp_str += '+'

						temp_str += ' - ' + str(passby_count[N*k_meas + i_stop])
						temp_str += ')'

						obj_str += temp_str + '*' + temp_str

						if not(i_stop == (N - 1) and k_meas == (I - 1)):
							obj_str += ' + '

				if TEST_QUADRATIC_PROG == 1:
					print()
					print("    objective is:")
					print(obj_str)
				# End of test print.

				exec(obj_str, globals())

				m.setObjective(obj, GRB.MINIMIZE)

				# Add constraints (part).
				count_constraint = 0

				# o_q_h
				for q_origin in range(N):
					for h_depart in range(T):

						const_str = ''
						temp_count = 0
						
						for r_dest in range(N):
							if r_dest != q_origin:

								if temp_count != 0:
									const_str += ' + '

								const_str += 'f_' + str(q_origin) + '_' + str(r_dest) + '_' + str(h_depart)
								temp_count += 1

						if temp_count > 0:
							const_str += ' == '
							const_str += 'o_' + str(q_origin) + '_' + str(h_depart)

							if TEST_QUADRATIC_PROG == 1:
								print()
								print("    A entry count constraint is added: ")
								print(const_str)
							# End of test print.
						
							m.addConstr(eval(const_str), 'c' + str(count_constraint))
							count_constraint += 1

			# If the sys is not Win, delete dated constraints.
			if platform.system() != 'Windows' and iteration_count != 0:
				m.remove(m.getConstrs()[N*T : 3 * N*T])

			# Add other constraints in the following.
			count_constraint = N*T

			# d_r_t
			for r_dest in range(N):
				for t in range(1,T):

					const_str = ''
					temp_count = 0

					for q_origin in range(N):
						for h_depart in range(T):

							if q_origin != r_dest and h_depart < t:

								if od_flow_to_stop_prob[q_origin,r_dest,h_depart,r_dest,t] > 0.0001:

									if temp_count != 0:
										const_str += ' + '

									const_str += str(od_flow_to_stop_prob[q_origin,r_dest,h_depart,r_dest,t]) + ' * f_' + str(q_origin) + '_' + str(r_dest) + '_' + str(h_depart)
									temp_count += 1
										
					if temp_count > 0:
						const_str += ' == '
						const_str += 'd_' + str(r_dest) + '_' +str(t)

						if TEST_QUADRATIC_PROG == 1:
							print()
							print("    A exit count constraint is added: ")
							print(const_str)
						# End of test print.

						m.addConstr(eval(const_str), 'c' + str(count_constraint))
						count_constraint += 1

			# x_i_t
			for i_stop in range(N):
				for t in range(T):

					const_str = ''
					temp_count = 0

					for q_origin in range(N):
						for r_dest in range(N):
							for h_depart in range(T):
								
								if i_stop != q_origin and i_stop != r_dest and t > h_depart and od_flow_to_stop_prob[q_origin,r_dest,h_depart,i_stop,t] > 0.0001:

									if temp_count != 0:
										const_str += ' + '

									const_str += str(od_flow_to_stop_prob[q_origin,r_dest,h_depart,i_stop,t]) + ' * f_' + str(q_origin) + '_' + str(r_dest) + '_' + str(h_depart)
									temp_count += 1

					if temp_count > 0:
						const_str += ' == '
						const_str += 'x_' + str(i_stop) + '_' +str(t)
						
						if TEST_QUADRATIC_PROG == 1:
							print()
							print("    A passby count constraint is added: ")
							print(const_str)
						# End of test print.

						m.addConstr(eval(const_str), 'c' + str(count_constraint))
						count_constraint += 1

			# Optimize model
			m.optimize()
			print('    Obj: %g' % m.objVal)

			if iteration_count == 0:
				with open("results/dynamic/obj_upper_level.csv", "w") as f:
					f.write(str(m.objVal))
			else:
				with open("results/dynamic/obj_upper_level.csv", "a") as f:
					f.write('\n')
					f.write(str(m.objVal))

			# Obtain results
			for v in m.getVars():
				temp_name = v.varName
				temp_name = temp_name.split('_')

				if temp_name[0] == 'f':
					od_flow[int(temp_name[1]),int(temp_name[2]),int(temp_name[3])] = v.x

		# Save results
		np.save('results/dynamic/od_flow_upper_level_iteration_' + str(iteration_count), od_flow)
		print()
		print('    The optimal od_flow obtained!')

		# Lower level - Solving dynamic schedule based UE assignment problem.
		print('    Lower level begins ...')
		
		# If this is not a test, then update od_flow_to_stop_prob.
		if (TEST_QUADRATIC_PROG != 1):
			od_flow_to_stop_prob, link_combined_flow = dynamic_schedule_based_ue_assignment_algorithm(tn, od_flow, od_flow_to_stop_prob)
			#np.save('results/dynamic/od_flow_to_stop_prob_upper_level_iteration_' + str(iteration_count), od_flow_to_stop_prob)
			np.save('results/dynamic/link_combined_flow_upper_level_iteration_' + str(iteration_count), link_combined_flow)
		else:
			print()
			print("    Testing quadratic prog; od_flow_to_stop_prob is not updated.")

		# Convergence test.
		if iteration_count >= 1:
			avg_mse_od_flow = 0
			for r in range(N):
				for s in range(N):
					for h_depart  in range(T):
						avg_mse_od_flow += (od_flow_last[r,s,h_depart] - od_flow[r,s,h_depart]) ** 2

						#if abs(od_flow_last[r,s,h_depart] - od_flow[r,s,h_depart]) > 0.1:
						#	print()
						#	print("    od_flow_last" + str([r,s,h_depart]) + ": " + str(od_flow_last[r,s,h_depart]))
						#	print("    od_flow" + str([r,s,h_depart]) + ": " + str(od_flow[r,s,h_depart]))

			avg_mse_od_flow /= N*N*T
			print()
			print("    Iteration: " + str(iteration_count) + " avg_mse_od_flow: " + str(avg_mse_od_flow))

			if iteration_count == 1:
				with open("results/dynamic/avg_mse_od_flow_upper_level.csv", 'w') as f:
					f.write(str(avg_mse_od_flow))
			else:
				with open("results/dynamic/avg_mse_od_flow_upper_level.csv", 'a') as f:
					f.write('\n')
					f.write(str(avg_mse_od_flow))

			avg_mse_link_flow = 0
			for link_index  in range(len(link_combined_flow)):
				avg_mse_link_flow += (link_combined_flow_last[link_index] - link_combined_flow[link_index]) ** 2

				#if abs(link_combined_flow_last[link_index] - link_combined_flow[link_index]) > 0.1:
				#	print()
				#	print("    link_combined_flow_last" + str([link_index]) + ": " + str(link_combined_flow_last[link_index]))
				#	print("    link_combined_flow" + str([link_index]) + ": " + str(link_combined_flow[link_index]))

			avg_mse_link_flow /= len(link_combined_flow)
			print()
			print("    Iteration: " + str(iteration_count) + " avg_mse_link_flow: " + str(avg_mse_link_flow))

			if iteration_count == 1:
				with open("results/dynamic/avg_mse_link_flow_upper_level.csv", 'w') as f:
					f.write(str(avg_mse_link_flow))
			else:
				with open("results/dynamic/avg_mse_link_flow_upper_level.csv", 'a') as f:
					f.write('\n')
					f.write(str(avg_mse_link_flow))

			if avg_mse_od_flow < CONVERGENCE_CRITERION and avg_mse_link_flow < CONVERGENCE_CRITERION:
				print()
				print("Convergence reached!")
				converg_flag = 1
				continue
			
		od_flow_last = deepcopy(od_flow)
		link_combined_flow_last = deepcopy(link_combined_flow)

		if iteration_count >= MAX_NUMBER_OF_ITERATIONS:
			print()
			print('    Warning! bi-level prog not converging at ' + str(MAX_NUMBER_OF_ITERATIONS) + 'th iteration!')
			sys.exit(1)

		iteration_count += 1

	return od_flow