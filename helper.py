import networkx as nx
import numpy as np
import random
from itertools import combinations, groupby, zip

class Graph:
	
	def __init__(self):
		self.n, self.m, self.k = 0,0, 0
		self.p = 0
		self.graph = None
		self.adjacency_matrix, self.cycle_matrix = None, None
		self.lookup = {}

	def create_random_biconnected_graph(self,n,p):
		self.n = n
		self.p = p
		self.graph = self._generate_random_graph()
		while not nx.is_biconnected(self.graph):
			self.graph = self.__generate_random_graph()
		self.m = self.graph.number_of_edges()
		self.k = self.m - self.n + 1
		#save adjacency matrix
		self.adjacency_matrix = nx.convert.to_dict_of_lists(new_graph)
		self.__create_edges_label()
	
	def create_graph_from_file(self,input_file):
		for row in input_file:
			if row:
				tokens = row.split(':')
				self.adjacency_matrix[int(tokens[0])] = list(map(int,tokens[1].split(',')))
		self.graph = nx.from_dict_of_lists(self.adjacency_matrix)
		self.n = self.graph.number_of_nodes()
		self.m = self.graph.number_of_edges()
		self.k = self.m - self.n + 1
		self.__create_edges_label()

	def write_graph_to_file(self,output_file):
		for key,value in self.adjacency_matrix.items():
			current_string = str(key) + ":"
			current_string += ','.join(str(v) for v in value)
			current_string += '\n'
			output_file.write(current_string)

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
		bfs_tree = nx.Graph()
		bfs_tree.add_edges_from(list(nx.bfs_edges(self.graph,source)))
		self.cycle_matrix = np.zeros((self.m,self.k))
		fundamental_edges = list(set(self.graph.edges())^set(bfs_tree))
		cycles = []
		counter = 0
		for (u,v) in fundamental_edges:
			if u>v: u,v = v,u
			path = nx.shortest_path(bfs_tree,u,v)
			for (x,y) in zip(path,path[1:]+[u]):
				xy_label = self.lookup[(x,y)] if x<y else self.lookup[(y,x)]
				if u<v: self.cycle_matrix[xy_label][counter] = 1
				else: self.cycle_matrix[xy_label][counter] = -1
			counter+=1




	
		
	
