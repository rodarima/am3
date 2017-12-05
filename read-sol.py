import numpy as np

def readsol(filename):
	"""Read a CPLEX solution file"""
	try:
		import xml.etree.ElementTree as et
	except ImportError:
		import elementtree.ElementTree as et
	solutionXML = et.parse(filename).getroot()
	solutionheader = solutionXML.find("header")
	status = int(solutionheader.get("solutionStatusValue"))

	slacks = {}
	constraints = solutionXML.find("linearConstraints")
	for constraint in constraints:
			name = constraint.get("name")
			slack = constraint.get("slack")
			slacks[name] = float(slack)

	values = {}
	for variable in solutionXML.find("variables"):
			name = variable.get("name")
			value = variable.get("value")
			values[name] = float(value)

	var_shape = {}
	for var,val in values.items():
		if not '(' in var: # is not an array
			var_shape[var] = [1]
			continue

		dim = var.count('(')
		split_var = var.replace(')','').split('(')
		name = split_var[0]
		str_indices = split_var[1:]
		indices = [int(i) for i in str_indices]
		if var in var_shape:
			last_indices = var_shape[var]
			for i in range(len(last_indices)):
				larger = max(last_indices[i], indices[i])
				indices[i] = larger

		var_shape[name] = indices

	variables = {}
	for var in var_shape:
		if len(var_shape[var]) > 1:
			variables[var] = np.zeros(var_shape[var], dtype='int') - 9999
		else:
			variables[var] = -9999 # Dummy initial var

	for var, val in values.items():
		if not '(' in var: # is not an array
			variables[var] = val
			continue

		dim = var.count('(')
		split_var = var.replace(')','').split('(')
		name = split_var[0]
		str_indices = split_var[1:]
		indices = [int(i)-1 for i in str_indices]
		variables[name][tuple(indices)] = val

	return status, variables, slacks

status, var, slacks = readsol('sol.xml')

VARS = ['NStart', 'NWorking', 'NPresent']

for name in VARS:
	print(name)
	print(var[name])
