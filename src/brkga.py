import numpy as np
import time

from nurses import Problem, Solution

class BRKGA:

	# Independent of the problem

	def __init__(self):
		self.population = []

	def solve(self):

		# Build a new population
		self.populate()
		self.best_fit = float('inf')
		last_fit = float('inf')
		rep = 0

		for i in range(self.max_iter):

			fit = self.decode()

			#if self.debug:
			#	print(self.last_status)
			self.best_fit = np.min(fit)

			if self.best_fit == last_fit: rep+=1
			else:
				rep = 0
				last_fit = self.best_fit

			if rep >= self.max_rep: break

			if self.break_fit != None:
				if self.best_fit < self.break_fit: break

			#print(np.min(fit), np.mean(fit))

			ind = np.argsort(fit)

			is_elite = self.classify(fit, ind)

			mutated = self.mutants()

			crossed = self.crossover(is_elite, mutated)

			self.population = np.concatenate(
				(self.population[is_elite], crossed, mutated))
			#input()

		fit = self.decode()
		if self.debug:
			print('Best fit found {}'.format(np.min(fit)))

		return self.population[np.argmin(fit)]


	def decode(self):
		'Should return an array of fitness values, one for each individual'
		raise NotImplementedError()

	def classify(self, fit, ind):
		is_elite = np.zeros(self.size, dtype='bool')

		is_elite[ind[0:self.n_elite]] = True

		return is_elite

	def mutants(self):
		return np.random.rand(self.n_mutant, self.genes)

	def crossover(self, is_elite, mutants):
		n_cross = self.n_cross

		is_non_elite = is_elite == False
		n_elites = int(np.sum(is_elite))
		n_non_elites = self.size - n_elites

		elite_indices = np.random.choice(n_elites, size=n_cross)
		non_elite_indices = np.random.choice(n_non_elites, size=n_cross)

		crossed = np.zeros([n_cross, self.genes])

		for i in range(n_cross):
			elite = self.population[is_elite][elite_indices[i]]
			non_elite = self.population[is_non_elite][non_elite_indices[i]]

			r = np.random.rand(self.genes)
			select_elite = r <= self.p_inherit

			crossed[i] = np.choose(select_elite, [non_elite, elite])

		#print(crossed)
		return crossed

	def populate(self):
		'Fill population with random individuals'

		self.population = np.random.rand(self.size, self.genes)
		self.n_elite = int(np.round(self.p_elite * self.size))
		self.n_mutant = int(np.round(self.p_mutant * self.size))
		self.n_cross = self.size - self.n_mutant - self.n_elite


class SolutionBRKGA(Solution, BRKGA):

	def __init__(self, problem):
		Solution.__init__(self, problem)
		BRKGA.__init__(self)

		self.debug = True

		# Population size
		self.size = 10

		self.p_elite = 0.1
		self.p_mutant = 0.2
		self.p_inherit = 0.7
		self.max_iter = 10000
		self.max_rep = 500
		#self.break_fit = self.N
		self.break_fit = None

		# Number of genes
		self.genes = self.N * (3 + self.H)

	def solve(self):
		best_individual = BRKGA.solve(self)
		self.decode_solution(best_individual)

	def decode_solution(self, gene):

		if self.debug:
			print('Best solution below:')
		self.fitness_gene(gene)

		#print(solution)
		#print(presences, starts)

	def map(self, genes, min_val, max_val, cast=True):
		'Map genes to an interval [min_val, max_val], inclusive'

		diff_val = max_val - min_val
		if cast:
			val = min_val + np.floor(genes * (diff_val + 1))
			val = val.astype('int')
		else:
			val = min_val + (genes * (diff_val))
		return val


	def decode(self):
		'Interpret each individual as a solution'

		fit = np.zeros(self.size)

		for i in range(self.size):
			gene = self.population[i]
			fit[i] = self.fitness_gene(gene)

		return fit

	def fitness_gene(self, gene):
		gene = gene.reshape([self.N, 3 + self.H])

		presences = gene[:, 0]
		starts = gene[:, 1]
		breaks = gene[:, 2]
		breaks_score = gene[:, 3:]

		return self.fitness(presences, starts, breaks, breaks_score)

	def fitness(self, raw_presences, raw_starts, raw_breaks, breaks_score):
		# Lower fitness mean better solutions

		p = self.problem
		presences = self.map(raw_presences, 0, p.maxPresence)
		presences_fl = self.map(raw_presences, 0, p.maxPresence, cast=False)
		starts = self.map(raw_starts, 0, self.H - p.minHours)
		starts_fl = self.map(raw_starts, 0, self.H - p.minHours, cast=False)
		breaks = self.map(raw_breaks, 0, np.ceil(p.maxPresence/2))
		breaks_fl = self.map(raw_breaks, 0, np.ceil(p.maxPresence/2), cast=False)

		breaks_score = breaks_score.reshape([self.N, self.H])


		demand = self.problem.demand
		demand_left = demand.copy()
		offer = np.zeros(self.H, dtype='int')
		bad_demand = 0		#BD
		bad_hours = 0		#BH
		bad_consec = 0		#BC
		bad_far = 0			#BF
		bad = [0,0,0,0]
		inf = 1000
		nurses_needed = 0

		score = np.zeros(self.H, dtype='int')

		P = np.zeros([self.N, self.H], dtype='int')
		W = np.zeros([self.N, self.H], dtype='int')
		S = np.zeros([self.N, self.H], dtype='int')

		# We compute the offer
		for n in range(self.N):
			presence = presences[n]
			start = starts[n]
			break_score = breaks_score[n]

			# Only working if they work at least minHours
			if presence < self.problem.minHours:
				#print('Nurse {} works {}, so is not working'.format(n, presence))
				continue

			S[n, start] = 1

			max_presence = self.H - start
			presence = min(max_presence, presence)

			work_time = np.array(range(start, start+presence))
			#print("Nurse {} start {}".format(n, start))
			#print("Nurse {} presence {}".format(n, presence))
			#print("Nurse {} work time {}".format(n, work_time))
			nurses_needed += 1
			offer[start:start+presence] += 1
			P[n, start:start+presence] = 1
			W[n, start:start+presence] = 1

			# If works more than maxHours, infeasible
			if presence - breaks[n] > p.maxHours:
				# More presence then more bad
				badness = (presences_fl[n] - breaks_fl[n]) - p.maxHours
				bad_hours += 200 + 20 * badness
				bad[1] +=1

			# If working less than minHours, infeasible
			if presence - breaks[n] < p.minHours:
				# If less presence then less bad
				badness = p.minHours - (presences_fl[n] - breaks_fl[n])
				#far = (p.minHours - (presence_float -  breaks[n]))
				bad_hours += 200 + 20 * badness
				bad[1] +=1


			# Avoid consecutive breaks
			penalty_breaks = np.zeros(presence)
			nurse_breaks = np.zeros(presence, dtype='bool')


			# Assign breaks using break_score

			if presence > p.maxConsec:
				min_breaks = 1
			else: min_breaks = 0

			max_breaks = max(min_breaks, breaks[n])

			for b in range(max_breaks):
			#for b in range(breaks_needed):
				demand_nurse = demand[work_time] - offer[work_time]
				score = break_score[:presence] + penalty_breaks
				#print("nurse={} break scores {}".format(n, score))
				#print('Demand now {}'.format(demand_now))
				chosen_break = np.argmin(score)
				#print(chosen_break)
				break_hour = chosen_break + start
				nurse_breaks[chosen_break] = True

				# Avoid already chosen breaks
				penalty_breaks[chosen_break] = inf

				# Avoid choosing consecutive break
				if chosen_break - 1 >= 0:
					penalty_breaks[chosen_break - 1] = inf
				if chosen_break + 1 < presence:
					penalty_breaks[chosen_break + 1] = inf

				# Remove the hour as working
				offer[break_hour] -= 1
				W[n, break_hour] = 0

			# Avoid working more than maxConsec consecutive hours
			max_consec = 0
			now_consec = 0
			last_restart = start
			for h in range(presence):
				hour = start + h
				if nurse_breaks[h]:
					max_consec = max(max_consec, now_consec)
					now_consec = 0
					last_restart = h+1
				else:
					now_consec += 1
			max_consec = max(max_consec, now_consec)


			if max_consec > p.maxConsec:
				#print('nurse={} max_consec={} breaks_fl={} break_score={}'.format(
				#	n, max_consec, breaks_fl[n], break_score))
				bad_consec += 100 * (max_consec - p.maxConsec) - 10 * breaks_fl[n]
				bad[2] += 1

		for h in range(self.H):

			# If less nurses are needed
			if demand[h] > offer[h]:
				# The solution is infeasible
				bad_demand += 100 * (demand[h] - offer[h])
				bad[0] +=1

				# Test if the solution is close to add a nurse covering the
				# demand

				for n in range(self.N):
					presence = presences[n]
					presence_fl = presences_fl[n]
					start = starts[n]
					start_fl = starts_fl[n]
					end = start + presence
					end_fl = start_fl + presence_fl

					# Exclude nurses working
					if presence >= p.minHours: continue


					# If the nurse cannot cover the hour, thats bad
					if start > h:
						dist = start_fl - h
						bad_far += dist

					elif end < h:
						dist = h - end_fl
						bad_far += dist
						bad[3] += 1

					# But if is close, count only the number of hours to
					# minHours
					bad_far += p.minHours - presences_fl[n]
					bad[3] += 1


		fit = max(1, nurses_needed) + bad_demand + bad_hours + bad_consec + bad_far

		self.P = P
		self.W = W
		self.S = S

		#print(self)


		#bad1, bad2, bad3, bad4

		if fit < self.best_fit:
			if self.debug:
				print(self)
			line = 'best={:8.2f}  bad: dem={:2d}  h={:2d}  \
consec={:2d}  far={:2d}'.format(
				fit, *bad)
			#print('BAD consec={:7.2f} h={:7.2f} dem={:7.2f} far={:7.2f}'.format(
			#	bad_consec, bad_hours, bad_demand, bad_far))
			print(line)

		#print('\r{}'.format(line), end='', flush=True)
		#print(line)
		#if self.best_fit < 150 and self.best_fit > 0:
		#	print(self)
		#	input()
		#time.sleep(0.01)

		return fit

	def store_solution(self):
		self.P = None
		self.W = None
		self.S = None

if __name__ == '__main__':

	from lp import SolutionLP
	from pulp import *


	problems = 10
	solved = 0

	seed = 2

	for i in range(problems):

		while True:
			p = Problem(seed, 50)
			seed += 1
			lp = SolutionLP(p, 'nurses')
			solver_params = dict(keepFiles=0, msg=0, timelimit=20)
			solver = CPLEX(**solver_params)
			lp.solve(solver)
			if lp.is_feasible(): break

		#print('LP Solution:')
		#print(lp)

		brkga = SolutionBRKGA(p)
		brkga.solve()
		#print(p)
		print(brkga)
		if brkga.is_feasible():
			print('Congratulations, the solution is feasible')
			solved += 1
		else:
			print('Infeasible solution')

	print("Solved {} of {}".format(solved, problems))
