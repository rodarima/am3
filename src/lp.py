import numpy as np
from pulp import *

from nurses import Problem, Solution

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

		self.store_solution()

		if status < 0: return False
		return True
		#return status

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
				#lp.add(NWorking[i][j] + NWorking[i][j+1] >= NPresent[i][j])
				lp.add((NPresent[i][j] - NWorking[i][j]) +
					(NPresent[i][j+1] - NWorking[i][j+1]) <= 1)

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
				lp.add(lpSum([NWorking[i][j+k] for k in range(p.maxConsec + 1)])
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

	def store_solution(self):
		self.P = self._lpmat_to_array(self.NPresent)
		self.W = self._lpmat_to_array(self.NWorking)
		self.S = self._lpmat_to_array(self.NStart)

	def _lpmat_to_array(self, lpmat):
		n, m = np.shape(np.array(lpmat))
		mat = np.zeros([n, m])
		for i in range(n):
			for j in range(m):
				mat[i,j] = value(lpmat[i][j])
		return mat

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


