import gurobipy as gp
import numpy as np

class IterativeProcedure:
	
	def __init__(self):
		self.psi = None, None
		# edge-cycle matrix
		self.A = None
		self.imax = 100
		self.tol=1e-6
		self.alpha = 0

	def __function(self,x):
		return np.dot(self.A.transpose(),self.__sign_square(self.get_flows(x)))

	def __jacobian(self,x):
		abs_flows = np.absolute(self.get_flows(x))
		diag_flows = np.diag(abs_flows)
		return 2*np.dot(np.dot(self.A.transpose(),diag_flows),self.A)

	def __sign_square(self,q):
		return np.multiply(np.absolute(q),q)

	def __matrix_norm(self,matrix):
		return max(abs(row.sum()) for row in matrix)

	def __vector_norm(self,vector):
		return max(np.absolute(vector))
	
	def __solve_singular(self, J, F, bound):
		model = gp.Model('singular')
		model.Params.LogToConsole = 0
		#var
		size = len(F)
		x = [i for i in range(size)]
		adjustments = model.addVars(x,name="adjustments", lb = -bound, ub = bound)
		max_adjustment = model.addVar(name="max_adjustment")

		eqs_idx = []
		for i,row in enumerate(J):
			if any(row)!=0:
				eqs_idx.append(i)

		#constraint jx = -f
		for i in range(len(x)):
			eq = gp.LinExpr(J[i,:],[adjustments[j] for j in range(len(x))])
			model.addConstr(eq==-F[i])

		#aux constraint max 
		for i in range(len(x)):
			model.addConstr(max_adjustment>=adjustments[i])
			model.addConstr(max_adjustment>=-adjustments[i])

		model.setObjective(max_adjustment, gp.GRB.MINIMIZE)
		model.update()
		model.optimize()
		x = model.getAttr('x',adjustments).values()
		return x

	def get_flows(self, x):
		return np.add(self.psi,np.dot(self.A,x))

	def newton(self):
		icounter = 0
		bound = sum(v for v in self.psi if v>0)
		k = len(self.A[0]) 
		x = [0]*k
		self.alpha = self.__vector_norm(self.__function(x))
		while icounter <= self.imax:
			J = self.__jacobian(x)
			F = self.__function(x)
			if np.linalg.norm(F) < self.tol:
				break
			#if Jacobian is singular
			rank_jacobian = np.linalg.matrix_rank(J)
			if rank_jacobian < k:
				dx = self.__solve_singular(J, F, bound)
			else:
				dx = np.linalg.solve(J,-F)
			x = np.add(x,dx)
			icounter+=1
		return x,icounter


