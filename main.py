from graph import Graph
from iterative_procedure import IterativeProcedure 
from flows_init import *
from os import listdir

def main():
	input_file_directory = "data4_2/"
	output_file = open("result.txt","a")
	for file in listdir(input_file_directory):
		print(file)
		result = process_file(input_file_directory+file)
		output_file.write(file + ',')
		output_file.write(','.join(str(i) for i in result))
		output_file.write('\n')
	output_file.close()

def process_file(file):
	g = Graph()
	g.create_graph_from_file(file)
	flow_value = 1
	source, sink = 0, g.n-1
	outflow = {source: flow_value, sink: -flow_value}
	ip = IterativeProcedure()
	ip.A = g.cycle_matrix

	#shortest path initialization
	print("Shortest path init")
	shortest_path = g.get_shortest_path(source,sink)
	psi1 = shortest_path_init(shortest_path, flow_value, g.lookup)
	ip.psi = psi1
	r1,i1 = ip.newton()
	alpha1 = ip.alpha

	#minimize the maximum flow initialization
	print("Minimize max flow init")
	psi2 = min_max_flow_init(g.nodes, g.edges, outflow, g.lookup)
	ip.psi = psi2
	r2,i2 = ip.newton()
	alpha2 = ip.alpha
	
	#laminar flow initialization
	print("Laminar flow init")
	psi3 = laminar_flow_init(g.nodes, g.edges, outflow, g.lookup)
	ip.psi = psi3
	r3,i3 = ip.newton()
	alpha3 = ip.alpha

	#epanet init initialization
	print("Epanet init")
	psi4 = epanet_init(g.nodes, g.edges, outflow, g.fundamental_edges, g.lookup)
	ip.psi = psi4
	r4,i4 = ip.newton()
	alpha4 = ip.alpha

	result = [file,g.n,g.m,g.k, i1, i2, i3, i4, alpha1, alpha2, alpha3,alpha4]
	return result

if __name__ == '__main__':
    main()