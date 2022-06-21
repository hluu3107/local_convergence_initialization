import numpy as np
#import scipy.sparse as sp
from gurobipy import *

def iter_newton(A,x,psi,imax=100,tol=1e-6):
	iter = 0
	#bound = sum(v for v in psi if v>0)*len(A)
	bound = 2
	#print(bound)
	while iter<=imax:
		J = jacobian(A,x,psi)
		Y = function(A,x,psi)
		if np.linalg.norm(Y) < tol:
			break
		#if Jacobian is singular
		rankJ = np.linalg.matrix_rank(J)
		if rankJ < len(x):
			print("i: ",iter)
			print("rank J: ", rankJ)
			dx = solve_singular(J, Y, bound)
		else:
			dx = np.linalg.solve(J,-Y)
		#print(np.add(np.dot(dx,J),Y))
		x = np.add(x,dx)
		#print(x)
		iter+=1
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

def solve_singular(J,Y,bound):
	m = Model('singular')
	m.Params.LogToConsole = 0
	#var
	x = [i for i in range(len(Y))]
	adjustments = m.addVars(x,name="adjustments", lb = -bound, ub = bound)
	maxAdj = m.addVar(name="maxAdj")

	eqs_idx = []
	for i,row in enumerate(J):
		if any(row)!=0:
			eqs_idx.append(i)

	#constraint Jx = -Y
	for i in range(len(x)):
		eq = LinExpr(J[i,:],[adjustments[j] for j in range(len(x))])
		m.addConstr(eq==-Y[i])

	#aux constraint max 
	for i in range(len(x)):
		m.addConstr(maxAdj>=adjustments[i])
		m.addConstr(maxAdj>=-adjustments[i])

	m.setObjective(maxAdj, GRB.MINIMIZE)
	
	m.update()
	#m.display()
	#m.write('model.lp')
	m.optimize()
	#print(m.status)
	res = m.getAttr('x',adjustments)
	return res.values()

def solve_singular1(J,Y,bound):
	m = Model('singular')
	m.Params.LogToConsole = 0
	#var
	x = [i for i in range(len(Y))]
	adjustments = m.addVars(x,name="adjustments", lb = -bound, ub = bound)
	absAdj = m.addVars(x,name="absAdj")
	maxAdj = m.addVar(name="maxAdj")

	eqs_idx = []
	for i,row in enumerate(J):
		if any(row)!=0:
			eqs_idx.append(i)

	#constraint Jx = -Y
	for i in range(len(x)):
		eq = LinExpr(J[i,:],[adjustments[j] for j in range(len(x))])
		m.addConstr(eq==-Y[i])

	#aux constraint max 
	for i in range(len(x)):
		m.addConstr(absAdj[i]==abs_(adjustments[i]))
		m.addConstr(maxAdj>=absAdj[i])

	m.setObjective(maxAdj, GRB.MINIMIZE)
	
	m.update()
	#m.display()
	#m.write('model.lp')
	m.optimize()
	#print(m.status)
	res = m.getAttr('x',adjustments)
	return res.values()

def solve_singular1(J, Y, bound):
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
	m.display()
	m.optimize()
	return adjustments.X











