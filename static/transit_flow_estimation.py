# main function
# Last update: May 18, 2019

from static_assignment_algorithm_run import static_assignment_algorithm_run
from static_model_run import static_model_run

# User iuput to select algorithm.
input_flag = 0
while input_flag==0:

	algorithm_selection = input("Please input: '1' for transit flow assignment," +\
		"'2' for transit flow estimation, '0' for quit: \n")

	try:
		algorithm_selection = int(algorithm_selection)
	except ValueError:
		print("ValueError! Please input again!")
	else:
		input_flag = 1

# Execute model.
if algorithm_selection == 1:
	static_assignment_algorithm_run()
elif algorithm_selection == 2:
	static_model_run()
elif algorithm_selection == 0:
	exit(0)



