from graph_helper import *

def test_routine(minV,maxV,prob):
	# 3 cycles graph
	# adj_matrix = {0:[1,2,3], 1:[0,2,3], 2:[0,1,3], 3:[0,1,2]}
	# A = np.transpose(np.array([[-1,-1,1,0,0,0],[0,0,-1,-1,1,0],[0,1,0,1,0,-1]]))
	# G = nx.from_dict_of_lists(adj_matrix)
	# lookup = {(0,1):0, (0,2):1, (0,3):2, (1,2):3, (1,3):4, (2,3):5}

	#random graph generator
	#minV, prob = 10, 0.3
	G, adj_matrix,A, lookup, fedges = random_generation_graph(minV,minV, prob)

	#print(adj_matrix)
	flow_value = 1
	n, m = len(adj_matrix), len(lookup)
	k = m-n+1
	source, sink = 0, n-1
	outflow = {source: flow_value, sink: -flow_value}
	#st = stnumber(G,source,sink)
	#H, reverse_mapping = relabelGraph(G, st)
	#adj_matrix_H = nx.convert.to_dict_of_lists(H)
	#nodes = list(H.nodes)
	nodes = [n for n in adj_matrix.keys()]
	edges = []
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				edges.append((u,v))

	#minimize the maximum flow initialization
	psi1 = min_max_flow_init(nodes, edges, outflow,lookup)
	x1 = [0]*k
	alpha1 = max(np.absolute(function(A,x1,psi1)))
	r1,i1 = iter_newton(A,x1,psi1)
	#print(psi1)
	#print(get_flows(A,r1,psi1))
	
	#laminar flow initialization
	psi2 = laminar_flow_init(nodes,edges,outflow,lookup)
	x2 = [0]*k
	alpha2 = max(np.absolute(function(A,x2,psi2)))
	r2,i2 = iter_newton(A,x2,psi2)

	#shortest path initialization
	psi3 = shortest_path_init(G,source,sink,flow_value,lookup)
	x3 = [0]*k
	alpha3 = max(np.absolute(function(A,x3,psi3)))
	r3,i3 = iter_newton(A,x3,psi3)


	#epanet init initialization
	psi4 = epanet_init(nodes, edges, outflow, A, lookup, fedges)
	x4 = [0]*k
	alpha4 = max(np.absolute(function(A,x4,psi4)))
	r4,i4 = iter_newton(A,x4,psi4)
	#print(psi4)
	#print(get_flows(A,r4,psi4))

	result = [len(nodes),len(edges), i1, i2, i3, i4, alpha1, alpha2, alpha3,alpha4]
	return result

def shortesttest(minV,maxV,prob):
	G, adj_matrix,A, lookup = random_generation_graph(minV,minV, prob)

	#print(adj_matrix)
	flow_value = 1
	n, m = len(adj_matrix), len(lookup)
	k = m-n+1
	source, sink = 0, n-1
	outflow = {source: flow_value, sink: -flow_value}
	#st = stnumber(G,source,sink)
	#H, reverse_mapping = relabelGraph(G, st)
	#adj_matrix_H = nx.convert.to_dict_of_lists(H)
	#nodes = list(H.nodes)
	nodes = [n for n in adj_matrix.keys()]
	edges = []
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				edges.append((u,v))
	#shortest path initialization
	psi3 = shortest_path_init(G,source,sink,flow_value,lookup)
	x3 = [0]*k
	alpha3 = max(np.absolute(function(A,x3,psi3)))
	r3,i3 = iter_newton(A,x3,psi3)

def testfromfile(f):
	G, adj_matrix,A, lookup,fedges = readfromfile(f)
	#print(adj_matrix)
	flow_value = 1
	n, m = len(adj_matrix), len(lookup)
	k = m-n+1
	source, sink = 0, n-1
	outflow = {source: flow_value, sink: -flow_value}
	#st = stnumber(G,source,sink)
	#H, reverse_mapping = relabelGraph(G, st)
	#adj_matrix_H = nx.convert.to_dict_of_lists(H)
	#nodes = list(H.nodes)
	nodes = [n for n in adj_matrix.keys()]
	edges = []
	for u,neighbors in adj_matrix.items():
		for v in neighbors:
			if u<v:
				edges.append((u,v))

	#epanet init initialization
	psi4 = epanet_init(nodes, edges, outflow, A, lookup,fedges)
	x4 = [0]*k
	alpha4 = max(np.absolute(function(A,x4,psi4)))
	r4,i4 = iter_newton(A,x4,psi4)
	print(i4)

def main():
	f = open("result.txt", "a")
	for n in range(10,200,10):
		numtest = 10
		prob = random.uniform(0.1,0.2)
		minV,maxV = n,n
		res = []
		for i in range(numtest):
			cur = test_routine(minV,maxV,prob)
			if cur:
				res.append(cur)
				#f.write(','.join(str(i) for i in res))
				#f.write('\n')
		#print(res)
		mean_res = np.mean(res,axis=0)
		f.write(','.join(str(i) for i in mean_res))
		f.write('\n')
	f.close()
			
def main1():
	f = 'graph1.txt'
	testfromfile(f)

if __name__ == '__main__':
    main()