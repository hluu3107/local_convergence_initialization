

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

def randomgenerator():
	#test_cycles_routine(10,0.3)
	graph_counter = 201
	for n in range(210,510,10):
		numtest = 10
		prob = random.uniform(0.01,0.1)
		for i in range(numtest):
			G, adj_matrix, A, lookup, fcycles, fedges = random_generation_graph(n,n, prob, graph_counter)
			graph_counter+=1
