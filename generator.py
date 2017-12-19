import numpy as np
from pulp import *


class Problem:
	def __init__(self, N, H=24):
		self.N = N
		self.H = H
	
	def random(self, prob=None):
		N = self.N
		H = self.H

		self.demand = np.zeros([H])

		if prob is None and self.H == 24:

			# Say we have a probability distribution for a day, for instance it's
			# more common that in the working hours we have more people. So let's
			# try with the following probability distribution.

			counts = np.array([
			#   1  2  3  4  5  6  7  8  9  10 11 12
				1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, # AM
				5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1  # PM
			])

			prob = counts / np.sum(counts)

		elif prob is None and self.H != 24:
			s = "Please provide a probability vector for your day of {} hours".format(
				self.H)
			raise ValueError(s)

		# Asume that we expect around 8 hours of work per day per nurse
		# XXX: Well, it seems that if we use values > 4 there is a lot of
		# infeasible problems...
		expected_work = 4/H

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
		self.maxPresence = np.random.randint(8, 12)
		#self.maxHours    = int(expected_work * H) + np.random.randint(0, 4)
		#self.minHours    = int(expected_work * H) - np.random.randint(0, 4)
		self.maxHours    = 12 + np.random.randint(0, 4)
		self.minHours    = 3 - np.random.randint(0, 2)

	def __str__(self):
		s = ''
		s += "Problem of N = {} nurses with H = {}\n".format(self.N, self.H)
		s += "demand:\n"
		s += "AM : {}\n".format(self.demand[0:int(self.H/2)])
		s += "PM : {}\n".format(self.demand[int(self.H/2):])
		s += "maxConsec   : {}\n".format(self.maxConsec)
		s += "maxPresence : {}\n".format(self.maxPresence)
		s += "maxHours    : {}\n".format(self.maxHours)
		s += "minHours    : {}".format(self.minHours)
		return s
	

class Solution:

	def __init__(self, problem):
		self.problem = problem
		self.N = problem.N
		self.H = problem.H

	def solve(self, *args, **kargs):
		raise NotImplementedError()

class SolutionLP(Solution):

	def __init__(self, problem, name):
		super().__init__(problem)
		self.name = name
		self.lp = LpProblem(self.name, LpMinimize)
		self.create_vars()
		self.set_constraints()

	def solve(self, solver=None):
		if solver is None:
			solver = GLPK(msg=1, keepFiles=1, options=['--log',self.name+'-pulp.log',
					'--last', '--tmlim', '120'])

		status = self.lp.solve(solver)
		return status

	def create_vars(self):
		p = self.problem
		H = range(p.H)
		N = range(p.N)
		#dvar boolean NPresent[i in N, s in H];
		#dvar boolean NWorking[i in N, w in H];
		#dvar boolean NStart[i in N, w in H];
		self.NPresent = LpVariable.matrix("NPresent", (N, H), cat='Binary')
		self.NWorking = LpVariable.matrix("NWorking", (N, H), cat='Binary')
		self.NStart = LpVariable.matrix("NStart", (N, H), cat='Binary')

	def set_constraints(self):
		lp = self.lp
		p = self.problem
		nHours = p.H
		nNurses = p.N
		H = range(p.H)
		N = range(p.N)

		D = self.problem.demand
		NPresent = self.NPresent
		NWorking = self.NWorking
		NStart = self.NStart

		# Demand per hr met
		
		for j in H:
			lp.add(lpSum([NWorking[i][j] for i in N]) >= D[j])
		
		# Working nurses should be present
		
		for i in N:
			for h in H:
				lp.add(NWorking[i][h] - NPresent[i][h] <= 0)

		# Nurse cannot take 2 consec rests
		
		for i in N:
			for j in range(nHours - 1):
				lp.add(NWorking[i][j] + NWorking[i][j+1] >= NPresent[i][j])

		# Nurses cannot work for more than maxHours
		
		for i in N:
			lp.add(lpSum([NWorking[i][h] for h in H]) <= p.maxHours)

		# Working nurses cannot work for less than minHours
		
		for i in N:
			lp.add(
				lpSum([NWorking[i][h] for h in H]) >=
				p.minHours * lpSum([NStart[i][h] for h in H]))

		# Nurse cannot work for more than maxConsec consecutive hrs
		
		for i in N:
			for j in range(nHours - p.maxConsec):
				lp.add(lpSum([NWorking[i][j+k] for k in range(p.maxConsec)])
					<= p.maxConsec)
			
		# Nurses cannot be present for more than maxPresence hrs
		
		for i in N:
			lp.add(lpSum([NPresent[i][h] for h in H]) <= p.maxPresence)

		# Nurses can start at most once per day

		for n in N:
			lp.add(lpSum([NStart[n][h] for h in H]) <= 1)

		# The following 3 constraints are needed to set NStart[n,h] to one if
		# NPresent[n,h-1] and NPresent[n,h] are 0 and 1, respectively. So we
		# detect the nurse traveling to the hospital at the hour h. The operation
		# is just the AND of NOT NPresent[n,h-1] and NPresent[n,h]. With (NOT a)
		# as (1 - a). Only the hours in [2, nHours] are used here

		for n in N:
			for h in range(1, nHours):
				lp.add(NStart[n][h] >= 1 - NPresent[n][h-1] + NPresent[n][h] - 1)
				lp.add(NStart[n][h] <= 1 - NPresent[n][h-1])
				lp.add(NStart[n][h] <= NPresent[n][h])

		# We need to add the cases where the nurses start at hour 1. If they
		# work at h, then must travel at h, otherwise not

		for n in N:
			lp.add(NStart[n][0] == NPresent[n][0])

		# minimize function
		lp.setObjective(lpSum([[NStart[n][h] for n in N] for h in H]))

	def print(self):
		self._mat_print("NPresent", self.NPresent)
		self._mat_print("NWorking", self.NWorking)
		self._mat_print("NStart", self.NStart)

	def _mat_print(self, name, mat):
		print("{} = ".format(name))
		n,m = np.shape(np.array(mat))
		for i in range(n):
			line = ""
			for j in range(m):
				if isinstance(mat, list): elem = mat[i][j]
				else: elem = mat[i,j]
	
				if isinstance(elem, LpVariable): elem=value(elem)
				if elem == None: elem = 0
				#line += "{:6.2f} ".format(elem)
				line += "{} ".format(int(elem))
				
				#if isinstance(
	#		l = [value(mat[i][j]) for j in range(m)]
	#		line = " ".join(map(str,l))
			print("    {}".format(line))
	

p = Problem(10)
s = None
RUNS = 20
successful = 0
for i in range(RUNS):
	iteration = i+1
	# Reproducible runs
	np.random.seed(iteration)
	p.random()
	print("\rIteration {}/{}".format(iteration, RUNS),
		end='', flush=True)
	s = SolutionLP(p, 'nurses')

	solver_params = dict(keepFiles=0, msg=0)

	# "BECAUSE THERE IS NO CHOICE!"

	#solver = COIN(**solver_params)
	solver = CPLEX(**solver_params)
	#solver = GLPK(**solver_params)

	status = s.solve(solver)

	if status == 1:
		#print('Found')
		successful += 1

print()
print("Rate of feasible problems {}/{}".format(successful, RUNS))

