import numpy as np
#import scipy.sparse as sp
from gurobipy import *

def iter_newton(A,x,psi,imax=100,tol=1e-6):
	iter = 1
	bound = sum(v for v in psi if v>0)*len(A)*2
	for i in range(imax):
		J = jacobian(A,x,psi)
		Y = function(A,x,psi)

		#if Jacobian is singular
		if np.linalg.matrix_rank(J) < len(x):
			#print("i: ",i)
			#print("not full rank")
			x = solve_singular(J, Y, bound)
			#print(x)
		else:
			dx = np.linalg.solve(J,-Y)
			x = np.add(x,dx)
		iter+=1
		if np.linalg.norm(Y) < tol:
			break
	#print("numberof iterations ", iter)
	return x,iter

def ssquare(a):
	return np.multiply(np.absolute(a),a)

def inf_norm(matrix):
    return max(abs(row.sum()) for row in matrix)

def get_flows(A,x,psi):
	q = np.add(psi,np.dot(A,x))
	return q

def function(A,x,psi):
	return np.dot(A.transpose(),ssquare(get_flows(A,x,psi)))

def jacobian(A,x,psi):
	abs_flows = np.absolute(get_flows(A,x,psi))
	diag_flows = np.diag(abs_flows)
	J = 2*np.dot(np.dot(A.transpose(),diag_flows),A)
	return J

def solve_singular(J, Y, bound):
	#print('Singular J')
	m = Model('singular')
	m.Params.LogToConsole = 0
	#var
	x = [i for i in range(len(J))]
	#adjustments = m.addVars(x,name="adjustments")
	adjustments = m.addMVar(shape=len(x),name="adjustments", lb = -bound, ub = bound)
	absAdj = m.addMVar(shape=len(x),name="absAdj")
	maxAdj = m.addVar(name="maxAdj")

	#constraint Jx = -Y
	m.addConstr(J @ adjustments == - Y)
	# for i in range(len(x)):
	# 	eq = LinExpr(J[i,:], [adjustments[j] for j in colidx])
	# 	m.addConstr(eq==-Y[i])
	#aux constraint max 
	for vadj,vabs in zip(adjustments.tolist(),absAdj.tolist()):
		m.addConstr(vabs==abs_(vadj))
		m.addConstr(maxAdj>=vabs)

	m.setObjective(maxAdj, GRB.MINIMIZE)
	
	m.update()
	#m.display()
	m.optimize()
	return adjustments.X











