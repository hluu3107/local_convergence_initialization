import gurobipy as gp

def min_max_flow_init(nodes, edges, outflow, lookup):
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)
	model = gp.Model('netflow')
	model.Params.LogToConsole = 0
	#vars
	flows = model.addVars(edges,name="flows",lb = -bound, ub = bound)
	max_flow = model.addVar(name="max_flow")
	#flow conservations constraints
	flows_conservation = {v: model.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}
	#max constrains
	for (u,v) in edges:
		model.addConstr(max_flow >= flows[u,v])
		model.addConstr(max_flow >= -flows[u,v]) 
	
	model.setObjective(max_flow, gp.GRB.MINIMIZE)
	model.update()
	model.optimize()
	flows = model.getAttr('x',flows)
	return assign_psi(flows, lookup)

def laminar_flow_init(nodes, edges, outflow, lookup):
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)
	model = gp.Model('netflow')
	model.Params.LogToConsole = 0
	#vars
	flows = model.addVars(edges,name="flows",lb = -bound, ub = bound)
	pressures = model.addVars(nodes, name="pressures", lb = 0)
	#flow conservations constraints
	flows_conservation = {v: model.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}
	#pressure constrains
	pressure_constrains = {(u,v): model.addConstr(flows[(u,v)] == pressures[u]-pressures[v]) for (u,v) in edges}
	outlet_pressure = model.addConstr(pressures[len(nodes)-1] == 0)
	
	model.setObjective(0, gp.GRB.MINIMIZE)
	model.update()
	model.optimize()
	flows = model.getAttr('x',flows)
	return assign_psi(flows, lookup)

def assign_psi(flows, lookup):
	psi = [0]*len(lookup)
	for (u,v),flow in flows.items():
		if u>v: u,v = v,u
		psi[lookup[(u,v)]] = flow
	return psi

def shortest_path_init(path,flow_value,lookup):	
	flows = {}
	for (u,v) in path:
		if u<v:
			flows[(u,v)] = flow_value
		else:
			flows[(v,u)] = -flow_value
	return assign_psi(flows, lookup)

def epanet_init(nodes, edges, outflow, fundamental_edges, lookup):
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)*len(edges)
	model = gp.Model('epanet')
	model.Params.LogToConsole = 0
	#flows var
	flows = model.addVars(edges,name="flows",lb = -bound, ub = bound)
	
	#flow conservation constraints
	flows_conservation = {v: model.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}

	#set fundamental edge flows to one
	for (u,v) in fundamental_edges:
		if u>v: u,v = v,u
		model.addConstr(flows[(u,v)]==1)

	model.setObjective(0, gp.GRB.MINIMIZE)
	model.update()
	model.optimize()
	flows = model.getAttr('x',flows)
	return assign_psi(flows, lookup)

