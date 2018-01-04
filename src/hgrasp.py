import random
import numpy as np
from functools import reduce
from pulp import *

from nurses import Solution

class SolutionHGRASP(Solution):
	def __init__(self, problem, name):
		super().__init__(problem)
		self.name = name
		self.solution = []

	def solve(self, alpha=0.7):
		random.seed(3)
		for hour, demand in sorted(enumerate(self.problem.demand), key=lambda e: e[1], reverse=True):
			candidates = [{'nurse':nurse, 'hour':hour, 'action': action, 'cost':float('inf')} for nurse in range(self.N) for action in ["work", "break", "home"]]
			while (len(candidates) != 0) and (not self.demandMet(hour) or not self.allWorkingNursesOffTheirAsses(hour)):
				candidates = list(filter(lambda e: self.isFeasible(e), candidates))
				if (len(candidates) == 0): break
				candidates = list(map(lambda e: self.computeCost(e), candidates))
				minCost = reduce(lambda e1, e2: e1 if (e1["cost"] < e2["cost"]) else e2, candidates)["cost"]
				maxCost = reduce(lambda e1, e2: e1 if (e1["cost"] > e2["cost"]) else e2, candidates)["cost"]
				boundaryCost = minCost + (maxCost - minCost) * alpha
				rcl = list(filter(lambda e: e["cost"] <= boundaryCost, candidates))
				selection = random.choice(rcl)
				self.solution.append(selection)

		self.store_solution(self.solution)
		if not self.is_feasible():
			print('Not feasible, solutionFeasible() = {}'.format(
				self.solutionFeasible()))
			print(self)
		if (not self.solutionFeasible()):
			return None
		self.runLocalSearch()
		return self.solution

	def store_solution(self, solution):
		n, m = (self.N, self.H)
		self.P = np.zeros([n, m], dtype='int')
		self.W = np.zeros([n, m], dtype='int')
		self.S = np.zeros([n, m], dtype='int')

		# Fill P and W from solution
		for e in solution:
			n = e['nurse']
			h = e['hour']
			action = e['action']
			actions = {"home":0, "work":1, "break":2}
			s = actions[action]
			if s == 1:
				self.P[n, h] = 1
				self.W[n, h] = 1
			if s == 2:
				self.P[n, h] = 1

		# Fill NStart from P
		for n in range(self.N):
			for h in range(1, self.H):
				if self.P[n, h-1] == 0 and self.P[n, h] == 1:
					self.S[n, h] = 1
			self.S[n, 0] = self.P[n, 0]

	def allWorkingNursesOffTheirAsses(self, hour):
		isFeasible = True
		previouslyAssigned = list(filter(lambda e: e["hour"] < hour, self.solution))
		if len(previouslyAssigned) == 0: isFeasible = False
		for i, element in enumerate(previouslyAssigned):
			if (len(list(filter(lambda e: e["nurse"] == element["nurse"] and e["hour"] == hour, self.solution))) == 0 and (i != len(previouslyAssigned)-1 or len(list(filter(lambda e: e["nurse"] == element["nurse"] and e["action"] == "home", self.solution)))> 0)):
				isFeasible = False
		return isFeasible

	def solutionFeasible(self):
		demandMet = len(list(filter(lambda h: self.demandMet(h), range(self.H)))) == self.H
		return demandMet

	def demandMet(self, hour):
		assignedThisHour = list(filter(lambda e: e["hour"] == hour and e["action"] == "work", self.solution))
		return len(assignedThisHour) >= self.problem.demand[hour]

	def computeCost(self, element):
		releventElementsOfSolution = list(filter(lambda e: e["nurse"] == element["nurse"], self.solution))
		workingElements = list(filter(lambda e: e["action"] == "work", releventElementsOfSolution))
		breakElements = list(filter(lambda e: e["action"] == "break", releventElementsOfSolution))
		isNotAlreadyWorking = len(workingElements) == 0
		tookABreakLastHour =  len(list(filter(lambda e: e["hour"] == element["hour"] - 1, breakElements))) > 0
		remainingHours = self.problem.maxHours - len(workingElements)
		demandLeft = self.problem.demand[element["hour"]] - len(list(filter(lambda e: e["hour"] == element["hour"], self.solution)))
		hasNotWorkedMinHours = len(workingElements) <= self.problem.minHours
		cost = isNotAlreadyWorking*50 - (1-isNotAlreadyWorking)*(element["action"] == "work")*hasNotWorkedMinHours*30 - (element["action"] == "work")*demandLeft*100 - (element["action"] == "work")*tookABreakLastHour*20
		element["cost"] = cost
		return element

	def isFeasible(self, element):
		isFeasible = True
		elementsOfSolutionWithNurse = list(filter(lambda e: e["nurse"] == element["nurse"], self.solution))
		hasAlreadyGoneHome = len(list(filter(lambda e: e["action"] == "home" and e["hour"] < element["hour"], elementsOfSolutionWithNurse))) > 0
		if hasAlreadyGoneHome:
			return False

		workingElements = list(filter(lambda e: e["action"] == "work", elementsOfSolutionWithNurse))
		if (element["action"] == "home" and len(workingElements) ==  0):
			return False

		breakElements = list(filter(lambda e: e["action"] == "break", elementsOfSolutionWithNurse))
		workedLastHour = len(list(filter(lambda e: e["hour"] == element["hour"] - 1, workingElements))) > 0

		#Can't take 2 breaks in a row
		tookABreakLastHour = len(list(filter(lambda e: e["hour"] == element["hour"] - 1, breakElements))) > 0
		if (element["action"] == "break" and not workedLastHour):
		   isFeasible = False

		if (element["action"] == "home" and (not workedLastHour and not tookABreakLastHour)):
			isFeasible = False

		isAlreadyAssignedAtThisHour = len(list(filter(lambda e: e["hour"] == element["hour"], elementsOfSolutionWithNurse))) > 0

		earlierThanMaxPresence = list(filter(lambda e: e["hour"] <= element["hour"] - self.problem.maxPresence and e["action"] != "home", elementsOfSolutionWithNurse))
		moreThanMaxPresent = len(earlierThanMaxPresence) > 0
		moreThanMaxHours = len(workingElements) > self.problem.maxHours
		moreThanMaxConsec = False
		sReleventElementsOfSolution = sorted(workingElements, key=lambda e: e["hour"], reverse=False)
		count = 0
		for i in range(1, len(sReleventElementsOfSolution)):
			if (sReleventElementsOfSolution[i]["hour"] != sReleventElementsOfSolution[i-1]["hour"] + 1):
				count = 0
				continue
			count+=1
			if (count > self.problem.maxConsec):
				moreThanMaxConsec = True

		if (isAlreadyAssignedAtThisHour or moreThanMaxPresent or moreThanMaxHours or moreThanMaxConsec):
			isFeasible = False
		return isFeasible

	def runLocalSearch(self):
		bestSolution = self.solution
		bestNumNurses = self.countWorkingNurses(bestSolution)
		continueIteration = True
		while(continueIteration):
			continueIteration = False
			neighbour = self.pickNeighbour(bestSolution)
			numNurses = self.countWorkingNurses(neighbour)
			if (numNurses < bestNumNurses):
				bestSolution = neighbour
				bestNumNurses = numNurses
				keepIterating = True

		return bestSolution

	def pickNeighbour(self, solution, policy="first_improvement", strategy="whatever"):
		return solution

	def countWorkingNurses(self, solution):
		#elementsWithWorkTag = list(filter(lambda e: e[2] == "work", solution))
		mapToListNurseIds = list(map(lambda e: e["nurse"], solution))
		return len(list(set(mapToListNurseIds)))

	def printSolution(self):
		assigned = {}
		for element in sorted(self.solution, key=lambda e: e["hour"], reverse=False):
			nurse = element["nurse"]
			if not nurse in assigned:
				assigned[nurse] = []
			assigned[nurse].append({"hour":element["hour"], "action":element["action"]})
		print("Demand is: ", self.problem.demand)
		# print("maxConsec is: ", self.problem.maxConsec)
		# print("maxHours is: ", self.problem.maxHours)
		# print("maxPresence is: ", self.problem.maxPresence)
		for nurse in assigned:
			element = assigned[nurse]
			print("Nurse ", nurse, element)

