import numpy as np

from StaticNetwork import StaticNetwork
from static_assignment_algorithm import static_assignment_algorithm

def static_assignment_algorithm_run():

	sn = StaticNetwork()

	num_stops = len(sn.stops)

	od_flow = np.zeros((num_stops, num_stops))

	# Option 1) Manual Input
	od_flow[0,3] = 100

	od_flow_to_stop_prob = static_assignment_algorithm(sn,od_flow)