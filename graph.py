import networkx as nx
import numpy as np
import random
from itertools import combinations, groupby

class Graph:
	
	def __init__(self):
		self.n, self.m, self.k = 0,0, 0
		self.p = 0
		self.graph = None
		self.adjacency_matrix, self.cycle_matrix = {}, None
		self.nodes, self.edges = [], []
		self.lookup = {}
		self.fundamental_edges = []

	def create_random_biconnected_graph(self,n,p):
		self.n = n
		self.p = p
		self.graph = self.__generate_random_graph()
		while not nx.is_biconnected(self.graph):
			self.graph = self.__generate_random_graph()
		#save adjacency matrix
		self.adjacency_matrix = nx.convert.to_dict_of_lists(new_graph)
		self.m = self.graph.number_of_edges()
		self.k = self.m - self.n + 1
		self.__create_edges_label()
		self.__generate_params()

	def create_graph_from_file(self,input_file):
		file = open(input_file,"r")
		for row in file:
			if row:
				tokens = row.split(':')
				self.adjacency_matrix[int(tokens[0])] = list(map(int,tokens[1].split(',')))
		self.graph = nx.from_dict_of_lists(self.adjacency_matrix)
		self.n = self.graph.number_of_nodes()
		self.m = self.graph.number_of_edges()
		self.k = self.m - self.n + 1
		self.__create_edges_label()
		self.__generate_params()
		file.close()

	def write_graph_to_file(self,output_file):
		for key,value in self.adjacency_matrix.items():
			current_string = str(key) + ":"
			current_string += ','.join(str(v) for v in value)
			current_string += '\n'
			output_file.write(current_string)

	def get_shortest_path(self,source,sink):
		shortest_path = list(nx.bidirectional_shortest_path(self.graph,source,sink))
		path = []
		for (e1,e2) in zip(shortest_path[:-2],shortest_path[1:]):
			path.append((e1,e2))
		return path

	def assign_psi(self,flows):
		psi = [0]*self.m
		for (u,v),flow in flows.items():
			if u>v: u,v = v,u
			psi[self.lookup[(u,v)]] = flow
		return psi

	def __generate_random_graph(self):
		edges = combinations(range(n), 2)
		new_graph = nx.Graph()
		new_graph.add_nodes_from(range(n))
		if p<=0: return new_graph
		if p>=1: return nx.complete_graph(n,create_using=graph)
		for _, node_edges in groupby(edges, key=lambda x:x[0]):
			node_edges = list(node_edges)
			random_edge = random.choice(node_edges)
			new_graph.add_edge(random_edge)
			for e in node_edges:
				if random.random() < p:
					new_graph.add_edge(e)		
		return new_graph

	def __create_edges_label(self):
		counter = 0
		for u,neighbors in self.adjacency_matrix.items():
			for v in neighbors:
				if u<v:
					self.lookup[(u,v)] = counter
					counter+=1
	
	def __generate_params(self):
		self.nodes = [n for n in self.adjacency_matrix.keys()]
		for u,neighbors in self.adjacency_matrix.items():
			for v in neighbors:
				if u<v:
					self.edges.append((u,v))
		self.__bfs_basic()

	def __dfs(self,source=0):
		low, pre = [float('inf')]*self.n, [-1]*self.n
		counter = [0]
		def __dfs_helper(self,node, parent):
			counter[0]+=1
			pre[node], low[node] = counter[0], counter[0]
			for x in G[node]:
				if x==parent:
					continue
				if pre[x]==-1:
					self.__dfs_helper(x, node)
					low[node] = min(low[node], low[x])
				else: #x is not parent of node
					low[node] = min(low[node],pre[x])
		pre[source] = 0
		self.__dfs_helper(source, -1)
		return pre, low

	def __bfs_basic(self,source=0):
		bfs_tree_edges = list(nx.bfs_edges(self.graph,source))
		bfs_tree = nx.Graph()
		bfs_tree.add_edges_from(bfs_tree_edges)
		self.cycle_matrix = np.zeros((self.m,self.k))
		for (u,v) in self.edges:
			if (u,v) in bfs_tree_edges or (v,u) in bfs_tree_edges:
				continue
			self.fundamental_edges.append((u,v))
		counter = 0
		for (u,v) in self.fundamental_edges:
			path = nx.shortest_path(bfs_tree,u,v)
			#print(path)
			for (x,y) in zip(path,path[1:]+[path[0]]):
				#print(x,y)
				xy_label = self.lookup[(x,y)] if x<y else self.lookup[(y,x)]
				if u<v: self.cycle_matrix[xy_label][counter] = 1
				else: self.cycle_matrix[xy_label][counter] = -1
			counter+=1





	
		
	
