# Dijkstra's algorithm for finding shortest path, whose results will be used
# for finding the initial strategy.

# Note:
# 1) It's assumed that links data format will be the same with tn.links;
# 2) Only arrival_timeance labels will be returned, since only these labels are needed for finding 
# the initial strategy.
# 3) Link cost are based on uncongested cost (travel time).

from open_config_file import open_config_file

class Graph:
	"""Input #nodes, links, and sink node index; return arrival_timeance list."""

	def __init__(self,V,links,stops_exp_2):
		# Total numer of nodes.
		self.V = V
		self.links = links
		self.stops_exp_2 = stops_exp_2

	def dijkstra_arrival_time(self, r):
		# r is the destination.

		TEST = int(open_config_file('test_Graph'))

		unlabeled_nodes = set()
		arrival_time = []
		sequence = []

		# Initialzie nodes and arrival_time.
		for i in range(self.V):
			unlabeled_nodes.add(i)
			arrival_time.append(float('inf'))
			sequence.append(float('inf'))

		# Destination has arrival_time 0.
		arrival_time[r] = 0

		# Used to update sequence.
		count = 0

		while unlabeled_nodes:

			# Find the node with smallest arrival_timeance label in unlabeled nodes set.
			min_arrival_time = float('inf')
			min_node = float('inf')

			for i in unlabeled_nodes:
				if arrival_time[i] < min_arrival_time:
					min_arrival_time = arrival_time[i]
					min_node = i

			# Remove this node from the unlabeled list, namely finalize the arrival_time.
			if min_node != float('inf'):
				unlabeled_nodes.remove(min_node)

			# If all remaining unlabeled nodes are not connected with current destination ...
			# break out of loop.
			else:
				break

			# Update the arrival_time of upstream nodes of min_node
			incoming_links = [item for item in self.links if item[3] == min_node]
			for item in incoming_links:
				if item[4] == 'Dummy':
					# This link connect to destination; the arrival time equals 
					# the time of this node.
					arrival_time[item[2]] = self.stops_exp_2[item[2]][2]

					sequence[item[2]] = count
					count += 1

				else:
					# If the arrival time of the end of this link has NOT been made pernament.
					if  arrival_time[item[2]] == float('inf'):
						arrival_time[item[2]] = arrival_time[min_node]

						sequence[item[2]] = count
						count += 1

		return arrival_time, sequence

	def dijkstra_travel_cost(self, r):
		# r is the destination.

		TEST = int(open_config_file('test_Graph'))

		unlabeled_nodes = set()
		travel_cost = []
		sequence = []

		# Initialzie nodes and travel_cost.
		for i in range(self.V):
			unlabeled_nodes.add(i)
			travel_cost.append(float('inf'))
			sequence.append(float('inf'))

		# Destination has travel_cost 0.
		travel_cost[r] = 0

		# Used to update sequence.
		count = 0

		while unlabeled_nodes:

			# Find the node with smallest travel_cost label in unlabeled nodes set.
			min_travel_cost = float('inf')
			min_node = float('inf')

			for i in unlabeled_nodes:
				if travel_cost[i] < min_travel_cost:
					min_travel_cost = travel_cost[i]
					min_node = i

			# Remove this node from the unlabeled list, namely finalize the travel_cost.
			if min_node != float('inf'):
				unlabeled_nodes.remove(min_node)

			# If remaining unlabeled nodes are not connected with current destination ...
			# break out of loop.
			else:
				break

			# Update the travel_cost of upstream nodes of min_node.
			# Note: use link travel cost.
			incoming_links = [item for item in self.links if item[3] == min_node]
			for item in incoming_links:
				if travel_cost[item[2]] > travel_cost[min_node] + item[8]:
					travel_cost[item[2]] = travel_cost[min_node] + item[8]

					sequence[item[2]] = count
					count += 1

		return travel_cost, sequence