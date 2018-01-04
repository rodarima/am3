2018J3

This source directory contains different parts that solve random problems
created from the Problem class in 'nurses.py'.

The following files are different solvers:

- lp.py : Uses a LP solver to find a feasible solution. Different solvers are
available by the Pulp package.

- hgrasp.py : The GRASP metaheuristic is applied, but iterating for each hour at
the candidate selection.

- grasp.py : The GRASP metaheuristic is applied, but the candidate can be selected
from any hour. Slower than hgrasp. Also the cost function is more complex, and
rule based.

The program 'main.py' first selects a suitable problem by checking with 'lp.py'
that there are a feasible solution, and then applies a grasp metaheuristic to
try to find a solution.
