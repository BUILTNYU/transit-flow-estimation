# Further improvement is expected.

# Congestion coeffcient function f(v/c) should satisfy:
# 1) f(0) = 1;
# 2) strcitly convex.

def congestion_penalty_coeff(vol,cap,type):
	"""input volume and capacity; output cost."""

	# Formula coeff = (vol / cap)^2 + 1

	coeff = 1
	if type == 'Bus':
		coeff = (vol / cap) ** 2 + 1.0

	elif type == 'Subway' or 'Train':
		coeff = (vol / cap) ** 2 + 1.0

	else:
		print("Unrecognized transit node!")
		coeff = (vol / cap) ** 2 + 1.0

	return coeff