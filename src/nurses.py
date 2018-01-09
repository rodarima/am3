import random, time
import numpy as np
from functools import reduce
from pulp import *

class Problem:
	def __init__(self, seed, N, H=24):
		self.N = N
		self.H = H
		self.seed = seed
		self.random_problem(seed)

	def random_problem(self, seed, prob=None):
		N = self.N
		H = self.H

		np.random.seed(seed)

		self.demand = np.zeros([H])

		if prob is None and self.H == 24:

			# Say we have a probability distribution for a day, for instance it's
			# more common that in the working hours we have more people. So let's
			# try with the following probability distribution.

			counts = np.array([
			#	1  2  3  4	5  6  7  8	9  10 11 12
				1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, # AM
				5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1	# PM
			])

			prob = counts / np.sum(counts)

		elif prob is None and self.H != 24:
			s = "Please provide a probability vector for your day of {} hours".format(
				self.H)
			raise ValueError(s)

		# Asume that we expect around 8 hours of work per day per nurse
		# XXX: Well, it seems that if we use values > 4 there is a lot of
		# infeasible problems...
		expected_work = 8

		# About how many nurses are going to be required?
		fraction_nurses = 0.3

		# Now we fill a total of total_demand hours, which is proportional to
		# the number of nurses the expected work time.
		# We compute a fraction of that work in order to keep only some nurses
		# working.
		total_demand = int(round(N * expected_work * fraction_nurses))

		# The indices to each hour
		hi = np.arange(H)

		# Select total_demand elements with the given probability
		h_indices = np.random.choice(hi, total_demand, p=prob)

		# Then add one hour to each index in h_indices
		for h in h_indices:
			self.demand[h] += 1

		# Random limits
		self.maxPresence = expected_work + np.random.randint(0, 4)
		self.maxHours	 = expected_work + np.random.randint(0, 4)
		self.minHours	 = expected_work - np.random.randint(0, 4)
		self.maxConsec	 = self.maxHours - np.random.randint(0, 3)
		#self.maxHours	 = 12 + np.random.randint(0, 4)
		#self.minHours	 = 3 - np.random.randint(0, 2)

	def __str__(self):
		d = self.demand
		line = '-' * (8 + 2*self.H - 1) + '\n'
		s = ''
		s += 'Problem(seed={}, N={}, H={})\n'.format(self.seed, self.N, self.H)
		s += line
		s += 'hour\t'
		for h in range(self.H):
			s += '{} '.format(h//10)
		s += '\n'
		s += '\t'
		for h in range(self.H):
			s += '{} '.format(h % 10)
		s += '\n'
		s += line
		s += 'dem.\t'
		for h in range(self.H):
			s += '{} '.format(int(d[h]))
		s += '\n'
		s += line
		s += " maxConsec   : {}\n".format(self.maxConsec)
		s += " maxPresence : {}\n".format(self.maxPresence)
		s += " maxHours    : {}\n".format(self.maxHours)
		s += " minHours    : {}\n".format(self.minHours)
		s += line
		return s

class Solution:

	def __init__(self, problem):
		self.problem = problem
		self.N = problem.N
		self.H = problem.H
		self.debug = False

		# The final solution
		self.P = None
		self.W = None
		self.S = None

	def solve(self, *args, **kargs):
		raise NotImplementedError()

	def objective(self):
		return np.sum(self.S)

	def __str__(self):
		d = self.problem.demand
		line = '-' * (8 + 2*self.H - 1) + '\n'
		s = ''
		s += line
		s += 'hour\t'
		for h in range(self.H):
			s += '{} '.format(h//10)
		s += '\n'
		s += '\t'
		for h in range(self.H):
			s += '{} '.format(h % 10)
		s += '\n'
		s += line
		s += 'dem.\t'
		for h in range(self.H):
			s += '{} '.format(int(d[h]))
		s += '\n'
		s += line
		for n in range(self.N):
			s += 'n={}\t'.format(n)
			for h in range(self.H):
				st = '.'
				if self.P[n, h] == -1: st = ' '
				elif self.W[n, h] == 1: st = 'W'
				elif self.P[n, h] == 1: st = '-'
				s += '{} '.format(st)
			s += '\n'
		s += line
#		s = ''
#		s += "NPresent\n"
#		s += str(self.P) + '\n'
#		s += "NWorking\n"
#		s += str(self.W) + '\n'
#		s += "NStart\n"
#		s += str(self.S) + '\n'
		return s


	def is_feasible(self):
		p = self.problem
		nHours = p.H
		nNurses = p.N
		H = range(p.H)
		N = range(p.N)

		D = self.problem.demand

		P = self.P
		W = self.W
		S = self.S
		bad = False
		problems = []

		# Demand per hr met

		for j in H:
			if (np.sum([W[i][j] for i in N]) >= D[j]): continue
			problems.append("Demand not reached at hour {}".format(j))
			bad = True

		# Working nurses should be present

		for i in N:
			for h in H:
				if (W[i][h] - P[i][h] <= 0): continue
				problems.append(
					"Nurse {} at hour {} is working but not present".format(
					i, h))
				bad = True

		# Nurse cannot take 2 consec rests

		for i in N:
			for j in range(nHours - 1):
				if ((P[i][j] - W[i][j]) + (P[i][j+1] - W[i][j+1]) < 2): continue
				problems.append(
					"Nurse {} at hour {} is resting more than 1 hour".format(
					i, j))
				bad = True

		# Nurses cannot work for more than maxHours

		for i in N:
			if(np.sum([W[i][h] for h in H]) <= p.maxHours): continue
			problems.append("Nurse {} is working more than maxHours".format(i))
			bad = True

		# Working nurses cannot work for less than minHours

		for i in N:
			if(np.sum([W[i][h] for h in H]) >=
				p.minHours * np.sum([S[i][h] for h in H])): continue
			#if(np.sum([W[i][h] for h in H]) >= p.minHours): continue
			problems.append("Nurse {} is working less than minHours".format(i))
			bad = True

		# Nurse cannot work for more than maxConsec consecutive hrs

		for i in N:
			for j in range(nHours - p.maxConsec):
				if(np.sum([W[i][j+k] for k in range(p.maxConsec + 1)])
					<= p.maxConsec): continue
				problems.append(
					"Nurse {} is works more than maxConsec from {} to {}"
					.format(i, j, j + p.maxConsec))
				bad = True

		# Nurses cannot be present for more than maxPresence hrs

		for i in N:
			if(np.sum([P[i][h] for h in H]) <= p.maxPresence): continue
			problems.append("Nurse {} is present more than maxPresence hours".format(i))
			bad = True

		# Nurses can start at most once per day

		for n in N:
			if(np.sum([S[n][h] for h in H]) <= 1): continue
			problems.append("Nurse {} started more than one time".format(n))
			bad = True

		# The following 3 constraints are needed to set NStart[n,h] to one if
		# NPresent[n,h-1] and NPresent[n,h] are 0 and 1, respectively. So we
		# detect the nurse traveling to the hospital at the hour h. The operation
		# is just the AND of NOT NPresent[n,h-1] and NPresent[n,h]. With (NOT a)
		# as (1 - a). Only the hours in [2, nHours] are used here

		for n in N:
			for h in range(1, nHours):
				if((S[n][h] >= 1 - P[n][h-1] + P[n][h] - 1)
					and (S[n][h] <= 1 - P[n][h-1])
					and (S[n][h] <= P[n][h])): continue
				problems.append("Nurse {} at hour {} has inconsistent NStart".format(
					n, h))
				bad = True

		# We need to add the cases where the nurses start at hour 1. If they
		# work at h, then must travel at h, otherwise not

		for n in N:
			if(S[n][0] == P[n][0]): continue
			problems.append("Nurse {} at hour 0 has inconsistent NStart".format(n))
			bad = True

		if bad:
			if self.debug:
				print("Infeasible solution. Problems detected:")
				print('\n'.join(problems))
			return False

		return True

