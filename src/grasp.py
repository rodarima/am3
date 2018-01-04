import numpy as np
from pulp import *

from nurses import Problem, Solution

class SolutionGRASP(Solution):
	def __init__(self, problem, name):
		super().__init__(problem)
		self.name = name
		self.maxiter = 20
		self.debug = False
		self.states = {0:'home', 1:'working', 2:'break'}

	def init_tables(self):
		# Those tables help to speedup the GRASP greedy build phase
		self.workdone = [0] * self.H
		self.worktime = [0] * self.N
		self.presencetime = [0] * self.N
		self.consec_table = np.ones([self.N, self.H], dtype='int')
		self.assigned = set()

	def solve(self, alpha):
		# Algorithm extracted from p.284, fig 10.1, Chapter 10, Handbook of
		# Metaheuristics edited by Michel Gendreau and Jean-Yves Potvin
		best_solution = None
		for iteration in range(self.maxiter):
			print("Iteration {}".format(iteration))
			solution = self.greedy_build(alpha)
			self.store_solution(solution)
			if not self.is_feasible():
				#break
				if self.debug:
					print('Infeasible')
					print(self)
				continue
				#self.repair(solution)
			#solution = self.local_search(solution)
			#best_solution = self.update(solution, best_solution)
			best_solution = solution
			break
		return best_solution

	def greedy_build(self, alpha):
		self.init_tables()
		solution = set()
		C = self.initial_candidates()
		while len(C) != 0:
			costs, infos = self.evaluate_costs(C, solution)
			#print(costs)
			cmin = np.min(costs)
			cmax = np.max(costs)
			cmed = cmin + alpha * (cmax - cmin)
			RCL_indices = np.where(costs <= cmed)[0]
			si = np.random.choice(RCL_indices)
			s = C[si]
			#RCL = [C[i] for i in range(len(C)) if costs[i] <= cmed]
			#RCL = list(RCL)
			# Use numpy random number generator
			#si = np.random.randint(len(RCL))
			#s = RCL[si]

			if self.debug:
				psel = len(RCL_indices)/costs.shape[0]
				d = {}
				for i in RCL_indices:
					e = C[i]
					cost = costs[i]
					info = infos[i]
					print("{}: {}   {}".format(e, cost, info))
				print()
				print('  picked {} with cost {}'.format(s, costs[si]))
				print('  min={}  med={}  max={}  psel={:.3f}'.format(cmin, cmed, cmax, psel))
				print()

			self.select_candidate(solution, s)
			C = self.update_candidates(C, s)

			if self.debug:
				self.store_solution(solution)
				print(self)
				input()
		return solution

	def store_solution(self, solution):
		n, m = (self.N, self.H)
		self.P = np.zeros([n, m], dtype='int') - 1
		self.W = np.zeros([n, m], dtype='int') - 1
		self.S = np.zeros([n, m], dtype='int')

		# Fill P and W from solution
		for e in solution:
			n, h, s = e
			if s == 0:
				self.P[n, h] = 0
				self.W[n, h] = 0
			if s == 1:
				self.P[n, h] = 1
				self.W[n, h] = 1
			if s == 2:
				self.P[n, h] = 1
				self.W[n, h] = 0

		# Fill NStart from P
		for n in range(self.N):
			for h in range(1, self.H):
				if self.P[n, h-1] == 0 and self.P[n, h] == 1:
					self.S[n, h] = 1
			self.S[n, 0] = self.P[n, 0]

	def initial_candidates(self):
		'Build all the possible actions that can be selected'
		#C = set()
		C = []
		for n in range(self.N):
			for h in range(self.H):
				for s in self.states.keys():
					#C.add((n, h, s))
					C.append((n, h, s))
		return C

	def select_candidate(self, solution, e):
		solution.add(e)

		n,h,s = e

		self.assigned.add((n,h))

		if s == 1:
			self.worktime[n] += 1
			self.presencetime[n] += 1
			self.workdone[h] += 1
		elif s == 2:
			self.presencetime[n] += 1

		maxConsec = self.problem.maxConsec
		for i in range(self.H):
			len_next_consec = 0
			for k in range(1, maxConsec+1):
				if (n,i+k,1) in solution: len_next_consec+=1
				else: break

			len_prev_consec = 0
			for k in range(1, maxConsec+1):
				if (n,i-k,1) in solution: len_prev_consec+=1
				else: break

			total_possible_consec = len_prev_consec + 1 + len_next_consec
			if total_possible_consec > maxConsec:
				self.consec_table[n, i] = 0
			else:
				self.consec_table[n, i] = 1

	def evaluate_costs(self, C, solution):
		p = self.problem
		H = range(p.H)
		N = range(p.N)
		S = self.states.keys()
		costs = np.zeros([len(C)], dtype='int')
		infos = []
		inf = 1000

		for i in range(len(C)):
			e = C[i]
			cost = 0
			info = ''
			n, h, s = e

			#all_states = {(n,h,ss) for ss in S}
			#states_assigned = set.intersection(all_states, solution)
			#states_assigned = {e for e in solution if e[0] == n and e[1] == h}
			#already_assigned = len(states_assigned) > 0
			already_assigned = (n,h) in self.assigned

			#all_working_hours = {(n,hh,1) for hh in H}
			#working_hours = set.intersection(all_working_hours, solution)
			#working_time = len(working_hours)
			working_time = self.worktime[n]

			#all_breaking_hours = {(n,hh,2) for hh in H}
			#breaking_hours = set.intersection(all_breaking_hours, solution)

			#present_hours_set = set.union(working_hours, breaking_hours)
			#present_hours = len(present_hours_set)
			present_hours = self.presencetime[n]

			#all_working_nurses = {(nn,h,1) for nn in N}
			#working_nurses = set.intersection(all_working_nurses, solution)
			#work_done = len(working_nurses)
			work_done = self.workdone[h]
			demand_left = p.demand[h] - work_done
			hours_left = p.minHours - work_done
			#print(hours_left)

#			next_consec = {(n,h+k,1) for k in range(1, p.maxConsec+1)}
#			prev_consec = {(n,h-k,1) for k in range(1, p.maxConsec+1)}
#
#			next_consec = set.intersection(next_consec, solution)
#			prev_consec = set.intersection(prev_consec, solution)
#
#			len_next_consec = 0
#			for k in range(1, p.maxConsec+1):
#				if (n,h+k,1) in next_consec: len_next_consec+=1
#				else: break
#
#			len_prev_consec = 0
#			for k in range(1, p.maxConsec+1):
#				if (n,h-k,1) in prev_consec: len_prev_consec+=1
#				else: break
#
#
			#total_possible_consec = len_prev_consec + 1 + len_next_consec

			can_work_consec = self.consec_table[n, h]

			already_started = present_hours >= 1
			is_next_hour_present = ((n,h+1,1) in solution 
				or (n,h+1,2) in solution)
			is_previous_hour_present = ((n,h-1,1) in solution 
				or (n,h-1,2) in solution)

			is_next_hour_home = (n,h+1,0) in solution
			is_previous_hour_home = (n,h-1,0) in solution

			break_previous_hour = (n, h-1, 2) in solution
			break_next_hour = (n, h+1, 2) in solution

			# If we already have a assigned state, avoid
			if already_assigned:
				costs[i] = inf
				infos.append('+already-assigned ')
				#costs.append(inf)
				continue

			if s in {1, 2}: # Present at the hospital

				# If we are already the maximum time at the hospital, avoid
				if present_hours >= p.maxPresence:
					costs[i] = inf
					infos.append('+maximum-presence ')
					#costs.append(inf)
					continue

				if already_started:

					# If already started, and is next and previous hour home,
					# avoid going work again

					# If is adding more consecutive hours, prefer
					if is_next_hour_present or is_previous_hour_present:
						# But only if is some demand at that point
						if demand_left > 0:
							cost -= 333
							info += '-demand-left-consec '

					# Otherwise, try to avoid
					else:
						costs[i] = inf + 4
						infos.append('+multi-start ')
						#costs.append(inf+4)
						continue

			if s == 0: # Home
				if already_started:

					# Delay going home if is not at the hospital
					if not is_next_hour_present:
						cost += 401
						info += '+next-hour-ausent '

					else:
						cost += 401
						info += '+unknown '

				else:
					# Delay if didnt started
					cost += 500
					info += '+delay-start '

			elif s == 1: # Working

				# Avoid more work than allowed
				if working_time >= p.maxHours:
					costs[i] = inf
					infos.append('+max-hours-reached ')
					continue

				# Avoid working more than maxConsec hours
				if can_work_consec == 0:
					costs[i] = inf + 2
					infos.append('+max-consec-reached ')
					continue

				# We want a new nurse proportional to the demand
				if demand_left > 0:
					cost -= demand_left * 100
					info += '-demand-left '

				# Also try to reach minHours only if already started
				if hours_left > 0 and already_started:
					cost -= hours_left * 51
					info += '-hours-left '

				# Try to avoid work if there is no demand
				if not already_started and demand_left <= 0:
					cost += 700
					info += '+no-demand '

			elif s == 2: # Break
				# If was already resting, avoid at all costs
				if break_previous_hour or break_next_hour:
					costs[i] = inf
					infos.append('+already-resting-consec ')
					continue

				# Only rest if it's better than work
				if already_started:
					cost += 50
					info += '+avoid-resting '
				else:
					cost += 700
					info += '+avoid-first-resting '

			costs[i] = cost
			infos.append(info)
			#costs.append(cost)

		return costs, infos

	def update_candidates(self, C, selected):
		n, h, s = selected
		to_remove = {(n,h,ss) for ss in self.states}
		for ss in self.states:
			e = (n,h,ss)
			C.remove(e)
		return C
		#return C - to_remove

	def feasible(self): pass

	def repair(self): pass

	def local_search(self): pass

	def update(self): pass

