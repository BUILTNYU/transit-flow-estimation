# Delete all the test print;
# they begin eith "if TEST_" and end with "# End of test print."

import sys

filepath = "dynamic_schedule_based_ue_assignment_algorithm.py"
filepath2 = "dynamic_schedule_based_ue_assignment_algorithm_clean.py"

with open(filepath) as file:
	lines = file.readlines()

sc_clean_going_on = 0
sc_new = []

for index in range(len(lines)):

	if "if TEST_" in lines[index]:
		if sc_clean_going_on == 1:
			print("error!")
			sys.exit(1)
		else:
			sc_clean_going_on = 1

	if "End of test print" in lines[index]:
		sc_clean_going_on = 0

	if sc_clean_going_on == 0:
		sc_new.append(lines[index])
	else:
		print(lines[index])

with open(filepath2, "w") as f:
	for item in sc_new:
		f.write(item)