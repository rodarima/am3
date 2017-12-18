import numpy as np

# Reproducible runs
np.random.seed(1)

class Problem:
	def __init__(self, N, H=24):
		self.N = N
		self.H = H
	
	def random(self):
		N = self.N
		H = self.H

		P = np.zeros([N, H])
		W = np.zeros([N, H])
		S = np.zeros([N, H])

		self.demand = np.zeros([H])

		# Say we have a probability distribution for a day, for instance it's
		# more common that in the working hours we have more people. So let's
		# try with the following probability distribution.

		counts = np.array([
		#   1  2  3  4  5  6  7  8  9  10 11 12
			1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, # AM
			5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1  # PM
		])

		prob = counts / np.sum(counts)

		# Asume that we expect around 8 hours of work per day per nurse
		expected_work = 8/H

		# Now we fill a total of total_demand hours, which is proportional to
		# the number of nurses, and a proportion of expected work time
		total_demand = int(H * N * expected_work)

		# The indices to each hour
		hi = np.arange(H)

		# Select total_demand elements with the given probability
		h_indices = np.random.choice(hi, total_demand, p=prob)

		# Then add one hour to each index in h_indices
		for h in h_indices:
			self.demand[h] += 1

		# Random limits
		self.maxConsec   = np.random.randint(3, 10)
		self.maxPresence = np.random.randint(2, 6)
		self.maxHours    = expected_work * H + np.random.randint(0, 4)
		self.minHours    = expected_work * H - np.random.randint(0, 4)

	def __str__(self):
		s = ''
		s += "Problem of N = {} nurses with H = {}\n".format(self.N, self.H)
		s += "demand:\n"
		s += "AM : {}\n".format(self.demand[0:int(self.H/2)])
		s += "PM : {}".format(self.demand[int(self.H/2):])
		return s
	
p = Problem(10)
p.random()
print(p)
