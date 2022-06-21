import numpy as np
import networkx as nx
import random
from itertools import *
from gurobipy import *
from nr import *
np.set_printoptions(threshold=sys.maxsize)

def randomGraph(n,p):
	edges = combinations(range(n), 2)
	G = nx.Graph()
	G.add_nodes_from(range(n))
	if p <= 0: return G
	if p >= 1: return nx.complete_graph(n, create_using=G)
	for _, node_edges in groupby(edges, key=lambda x: x[0]):
		node_edges = list(node_edges)
		random_edge = random.choice(node_edges)
		G.add_edge(*random_edge)
		for e in node_edges:
			if random.random() < p:
				G.add_edge(*e)
	return G

def fundamental_cycle_basis(G,lookup):
	T = nx.minimum_spanning_tree(G)
	T_edges = list(T.edges())
	m = G.number_of_edges()
	n = G.number_of_nodes()
	k = m-n+1
	A = np.zeros((m,k))
	
	fedges = []
	cycles = []
	for (u,v) in list(G.edges()):
		if (u,v) in T_edges or (v,u) in T_edges:
			continue
		if u>v: u,v = v,u
		fedges.append((u,v))
		path = nx.shortest_path(T,u,v)
		if not path:
			path = nx.shortest_path(T,v,u)
		cycles.append(path)
	counter = 0
	basis = []
	for cycle in cycles:
		cur = []
		for i in range(0,len(cycle)-1):
			u,v = cycle[i],cycle[i+1]
			cur.append((u,v))
			if u<v:
				A[lookup[(u,v)]][counter]=1
			else:
				A[lookup[(v,u)]][counter]=-1
		first,last = cycle[-1],cycle[0]
		cur.append((first,last))
		if first < last:
			A[lookup[(first,last)]][counter]=1
		else:
			A[lookup[(last,first)]][counter]=-1
		counter+=1
		basis.append(cur)
	return basis,fedges,A

def random_generation_graph(minn,maxn, q, graph_counter):
	nnodes = random.randint(minn,maxn)
	G = randomGraph(nnodes,q)
	while not nx.is_biconnected(G):
		G = randomGraph(nnodes,q)
	#nedges = G.number_of_edges()
	#cycles = nx.cycle_basis(G)
	adj_matrix=nx.convert.to_dict_of_lists(G)
	#basis = []
	lookup = {}
	counter = 0
	#ncycles = len(cycles)
	#A = np.zeros((nedges,ncycles))
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				lookup[(u,v)] = counter
				counter+=1
	fcycles,fedges,A = fundamental_cycle_basis(G,lookup)
	write_graph_to_file(adj_matrix,graph_counter)


	return G, adj_matrix, A, lookup, fcycles, fedges

def generate_graph(n,q):
	G = randomGraph(n,q)
	while not nx.is_biconnected(G):
		G = randomGraph(n,q)
	adj_matrix=nx.convert.to_dict_of_lists(G)
	lookup = {}
	counter = 0
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				lookup[(u,v)] = counter
				counter+=1
	return G,adj_matrix, lookup

def stnumber(G,source,sink):
	n = G.number_of_nodes()
	low, pre, p, sign = [0]*n, [-1]*n, [-1]*n, [-1]*n
	preorder = []
	counter = [0]
	def dfs_helper(v):
		counter[0] = counter[0]+1
		pre[v] = counter[0]
		low[v] = v
		for w  in G.adj[v]:
			if pre[w] == -1:
				preorder.append(w)
				p[w] = v
				dfs_helper(w)
				if pre[low[w]] < pre[low[v]]:
					low[v] = low[w]
			elif pre[w] < pre[low[v]]:
				low[v] = w

	pre[source] = 0
	p[sink] = source
	dfs_helper(sink)
	sign[source] = -1
	L = [source,sink]
	for v in preorder:
		if sign[low[v]]==-1:
			idx = L.index(p[v])
			L.insert(idx,v)
			sign[p[v]] = 1
		else:
			idx = L.index(p[v])
			L.insert(idx+1,v)
			sign[p[v]] = -1
	cur = -1
	st = [0]*n
	for v in L:
		cur = cur+1
		st[v] = cur
	return st 

def write_graph_to_file(A,graph_counter):
	filename = "graph" + str(graph_counter) + ".txt"
	f = open('data/'+filename, "w")
	for key,value in A.items():
		ls = str(key) + ":"
		ls += ','.join(str(v) for v in value)
		f.write(ls)
		f.write('\n')
	f.close()
	graph_counter = graph_counter + 1

def readfromfile(file,filedirectory):
	f = open(filedirectory + file,"r")
	adj_matrix = {}
	for row in f:
		if row:
			ls = row.split(':')
			adj_matrix[int(ls[0])] = list(map(int,ls[1].split(',')))
	#print(adj_matrix)
	G = nx.from_dict_of_lists(adj_matrix)
	#nedges = G.number_of_edges()
	#cycles = nx.cycle_basis(G)
	#basis = []
	lookup = {}
	counter = 0
	#ncycles = len(cycles)
	#A = np.zeros((nedges,ncycles))
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				lookup[(u,v)] = counter
				counter+=1
	#fcycles,fedges,A = fundamental_cycle_basis(G,lookup)
	fcycles,fedges,A = get_bfs_basis(G,0, lookup)
	#print(A)
	return G, adj_matrix, A, lookup, fcycles, fedges

def dfs(G,source,sink):
	n = G.number_of_nodes()
	low, pre = [float('inf')]*n, [-1]*n
	counter = [0]
	def dfs_helper(node, parent):
		counter[0]+=1
		pre[node], low[node] = counter[0], counter[0]
		for x in G[node]:
			if x==parent:
				continue
			if pre[x]==-1:
				dfs_helper(x, node)
				low[node] = min(low[node], low[x])
			else: #x is not parent of node
				low[node] = min(low[node],pre[x])
	pre[source] = 0
	dfs_helper(source, -1)
	return pre, low

def min_max_flow_init(nodes, edges, outflow,lookup):
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)
	m = Model('netflow')
	m.Params.LogToConsole = 0
	#vars
	flows = m.addVars(edges,name="flows",lb = -bound, ub = bound)
	maxflow = m.addVar(name="maxflow")
	#aux vars for abs
	absflows = m.addVars(edges,name="absflows")

	#flow conservations constraints
	flows_conservation = {v: m.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}
	#max abs constrains
	for (u,v) in edges:
		#m.addConstr(maxflow >= flows[u,v]) 
		m.addConstr(absflows[u,v]== abs_(flows[u,v]))
		m.addConstr(maxflow >= absflows[u,v]) 
	
	m.setObjective(maxflow, GRB.MINIMIZE)
	m.update()
	m.optimize()
	flows = m.getAttr('x',flows)
	psi = getPsi(flows, lookup)
	return psi

def relabelGraph(G, st):
	mapping = {}
	reverse_mapping = {}
	for i in range(len(st)):
		mapping[i] = st[i]
		reverse_mapping[st[i]] = i
	H = nx.relabel_nodes(G, mapping)
	return H, reverse_mapping

def reverseFlows(flows, reverse_mapping):
	newFlows = {}
	for (u,v), flow in flows.items():
		newu, newv = reverse_mapping[u], reverse_mapping[v]
		newFlows[(newu,newv)] = flow
	return newFlows

def compute_constants(flows,A,lookup,k):
	psi = getPsi(flows, lookup)
	init_x = [0]*k
	alpha = max(np.absolute(function(A,init_x,psi)))
	j = jacobian(A,init_x,psi)
	#if np.linalg.matrix_rank(j) < k:
		#return []
	beta = inf_norm(np.linalg.inv(j))
	result_flows, iterations = iter_newton(A,init_x,psi)
	return [alpha,beta,result_flows, iterations]

def getPsi(flows, lookup):
	psi = [0]*len(lookup)
	for (u,v),flow in flows.items():
		if u>v: u,v = v,u
		psi[lookup[(u,v)]] = flow
		#if flow !=0:
			#print(u,v,flow)
	return psi

def shortest_path_init(G,source,sink,flow_value,lookup):
	shortest = list(nx.bidirectional_shortest_path(G,source,sink))
	path = []
	flows = {}
	for i in range(len(shortest)-1):
		path.append((shortest[i],shortest[i+1]))
	#print(path)
	for (u,v) in path:
		if u<v:
			flows[(u,v)] = flow_value
		else:
			flows[(v,u)] = -flow_value
	psi = getPsi(flows, lookup)
	#print(psi)
	return psi

def laminar_flow_init(nodes, edges, outflow,lookup):
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)
	m = Model('netflow')
	m.Params.LogToConsole = 0
	#vars
	flows = m.addVars(edges,name="flows",lb = -bound, ub = bound)
	pressures = m.addVars(nodes, name="pressures", lb = 0)
	#aux vars for abs
	#absflows = m.addVars(edges,name="absflows")

	#flow conservations constraints
	flows_conservation = {v: m.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}
	#pressure constrains
	pressure_constrains = {(u,v): m.addConstr(flows[(u,v)] == pressures[u]-pressures[v]) for (u,v) in edges}
	outlet_pressure = m.addConstr(pressures[len(nodes)-1] == 0)
	
	m.setObjective(0, GRB.MINIMIZE)
	m.update()
	m.optimize()
	flows = m.getAttr('x',flows)
	psi = getPsi(flows, lookup)
	return psi

def epanet_init(nodes, edges, outflow, A, lookup, fedges):
	#k = len(edges)-len(nodes)+1	
	#counting nonzero in each row to find fundamental edges
	#rows = (A!=0).sum(1)
	#fedges = [i for i in range(len(rows)) if rows[i]==1]
	#print(len(fedges))
	#print(rows)

	# to find out the mapping between edge number and actual edge
	reverse_lookup = {}
	for key,value in lookup.items():
		reverse_lookup[value] = key
	#print(reverse_lookup)
	bound = sum(outflow[key] for key in outflow if outflow[key]>0)*len(edges)
	m = Model('epanet')
	m.Params.LogToConsole = 0
	#flows var
	flows = m.addVars(edges,name="flows",lb = -bound, ub = bound)
	
	#flow conservation constraints
	flows_conservation = {v: m.addConstr(flows.sum(v, '*') == outflow.get(v, 0) + flows.sum('*', v)) for v in nodes}

	#set fundamental edge flows to one
	#for e in fedges:
	#	m.addConstr(flows[reverse_lookup[e]]==1)
	for (u,v) in fedges:
		if u>v: u,v = v,u
		m.addConstr(flows[(u,v)]==1)

	m.setObjective(0, GRB.MINIMIZE)
	m.update()
	#m.display()
	m.optimize()

	flows = m.getAttr('x',flows)
	psi = getPsi(flows, lookup)
	return psi

def change_basis(cycles):
	curBasis = cycles.copy()
	while True:
		combinations = list(itertools.combinations(curBasis,2))
		flag = False
		for i,(c1,c2) in enumerate(combinations):
			if len(c1)>len(c2): c1,c2 = c2, c1
			change, newc2 = merge_cycle(c1,c2)
			if change:
				curBasis.remove(c2)
				curBasis.append(newc2)
				flag = True
				break
		if flag:
			continue
		else:
			break
	return curBasis

def merge_cycle(c1, c2):
	c1, c2 = set(c1), set(c2)
	diff = c1.symmetric_difference(c2)
	if len(diff)<len(c2): 
		#print("change cycle")
		return True, list(diff)
	return False, list(c2)

def get_bfs_basis(G,source, lookup):
	bfs_edges = list(nx.bfs_edges(G,source))
	bfs_tree = nx.Graph()
	bfs_tree.add_edges_from(bfs_edges)
	m = G.number_of_edges()
	n = G.number_of_nodes()
	k = m-n+1
	A = np.zeros((m,k))
	
	fedges = []
	cycles = []
	for (u,v) in list(G.edges()):
		if (u,v) in bfs_edges or (v,u) in bfs_edges:
			continue
		if u>v: u,v = v,u
		fedges.append((u,v))
		path = nx.shortest_path(bfs_tree,u,v)
		if not path:
			path = nx.shortest_path(bfs_tree,v,u)
		cycles.append(path)
	counter = 0
	basis = []
	for cycle in cycles:
		cur = []
		for i in range(0,len(cycle)-1):
			u,v = cycle[i],cycle[i+1]
			cur.append((u,v))
			if u<v:
				A[lookup[(u,v)]][counter]=1
			else:
				A[lookup[(v,u)]][counter]=-1
		first,last = cycle[-1],cycle[0]
		cur.append((first,last))
		if first < last:
			A[lookup[(first,last)]][counter]=1
		else:
			A[lookup[(last,first)]][counter]=-1
		counter+=1
		basis.append(cur)
	return basis,fedges,A

def get_edge_cyclebasis(cycles, lookup):
	A = np.zeros((len(lookup),len(cycles)))
	counter = 0
	for cycle in cycles:
		cur = []
		for (u,v) in cycle:
			if u<v:
				A[lookup[(u,v)]] = 1
			else:
				A[lookup[(v,u)]] = -1		
	return A
