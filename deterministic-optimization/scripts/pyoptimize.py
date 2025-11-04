# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
import sys, traceback

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time
import pdb

# =============================================================================
# Extension modules
# =============================================================================
from pyOpt import Optimization
from pyOpt import SLSQP
from pyOpt import ALGENCAN

# Import TACS for analysis
from tacs import TACS, elements, constitutive, functions
from assembly import TACSRotorAssemblyFourBladed

# Optimization settings
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--smoothness', type=str , default='yes', help='yes/no')
parser.add_argument('--logfile'   , type=str , default='stresscon-massmin-smooth', help='log file for optimizer')
parser.add_argument('--algorithm' , type=str , default='SLSQP', help='SLSQP/ALGENCAN')
args = parser.parse_args()

# Create the rosenbrock function class
class PyoptRotorAssembly:
    def __init__(self, comm,
                 problem, num_vars,
                 smoothness=False):
        self.comm = comm
        self.nvars = num_vars
        self.problem = problem

        # Get tacs from 'problem' and create structural objective and
        # constraints
        tacs = problem.tacs            
        self.funcs = []
        self.funcs.append(functions.StructuralMass(tacs))
        self.funcs.append(functions.KSFailure(tacs, 100000.0))
        #self.funcs.append(functions.InducedFailure(problem.tacs, 100000.0))
        self.problem.integrator.setFunctions(self.funcs, num_vars)

        self.smoothness = smoothness
        if self.smoothness is True:
            self.ncon = len(self.funcs) - 1 + 2*(num_vars-1)
            if self.ncon <= 0:
                raise "not many variables to enforce smoothness constaints"
            self.delta = 1.0e-3 # 1mm smoothness constraint
        else:
            self.ncon = len(self.funcs) - 1

        # The design history file
        self.x_hist = []

        # Control redundant evalution of cost and constraint
        # gradients. Evaluate the gradient only if X has changed from
        # previous X.
        self.currentX = None

        # Space for current function and gradient values
        self.funcVals = np.zeros(len(self.funcs), TACS.dtype)
        self.gradVals = np.zeros(len(self.funcs)*self.nvars, TACS.dtype)
        
        return
   
    def evalObjCon(self, x):
        '''
        Evaluate the objective and constraint
        '''
        assert(np.all(x>0.0)==1)
        
        # Set the fail flag
        fail = 0

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Call the solver
        try:
            self.currentX = x        
            self.funcVals = self.problem.getFuncGrad(np.array(x),
                                                     self.funcs,
                                                     self.gradVals)
        except:
            print "exception occurred in funceval"
            traceback.print_exc(file=sys.stdout)
            fail = 1
            stop

        # Store the objective function value
        fobj = self.funcVals[0]
        
        # Store the constraint values
        con = [0.0]*self.ncon

        # 1 - sigma/sigma_max >= 0
        con[0] = 4.0*self.funcVals[1] - 1.0
        print con[0]
        
        if self.smoothness is True:
            con[1:self.ncon] = self.evalSmoothness(x)

        # Return the values
        return fobj, con, fail

    def evalObjConGradient(self, x, g, A):
        '''
        Evaluate the objective and constraint gradient
        '''
        assert(np.all(x>0.0)==1)
        
        # Set the fail flag
        fail = 0

        print "evaluating gradient"

        # Evaluate gradients if this is a new design point
        if np.array_equal(self.currentX,x) is False:
            try:
                print "Info: evaluating gradients at new x:", x
                self.currentX = x
                self.funcVals = self.problem.getFuncGrad(np.array(x),
                                                         self.funcs,
                                                         self.gradVals)
            except:
                print "exception occurred in evaluation of gradient"
                traceback.print_exc(file=sys.stdout)
                fail = 1
                stop

        # Set the objective gradient
        g = [0.0]*self.nvars        
        g[0:self.nvars] = self.gradVals[0:self.nvars]

        # Set the constraint gradient
        A = np.zeros([self.ncon, self.nvars], TACS.dtype)
        A[0,:] = 4.0*self.gradVals[self.nvars:2*self.nvars]
        
        # Append the gradient of smoothness constraints
        if self.smoothness is True:
            sgrad = self.evalSmoothnessGrad(x)
            for c in xrange(1,self.ncon):                
                A[c,:] = sgrad[c-1,:]
        
        return g, A, fail
    
    def evalSmoothness(self, x):        
        # There are as many smoothness cons as nvars-1
        con = np.zeros(2*(self.nvars-1), TACS.dtype)
        j = 0 # tracks variable
        k = 0 # tracks constraint
        for i in xrange(self.nvars-1):
            if j == 47: # or j == 95 or j == 143 or j == 191:
                j += 1
            con[k  ] = x[j]-x[j+1] - self.delta
            con[k+1] = x[j+1]-x[j] - self.delta
            k += 2
            j += 1
        return con

    def evalSmoothnessGrad(self,x):
        grad = np.zeros([2*(self.nvars-1),self.nvars], TACS.dtype)
        j = 0
        k = 0
        for i in xrange(self.nvars-1):
            if j == 47: # or j == 95 or j == 143 or j == 191:
                j += 1
            grad[k,j    ] =  1.0
            grad[k,j+1  ] = -1.0
            grad[k+1,j  ] = -1.0
            grad[k+1,j+1] =  1.0
            j += 1
            k += 2
        return grad

if __name__ == "__main__":

    print "hello"
    
## ######################################################################
## # Setup the optimization
## ######################################################################

problem = TACSRotorAssemblyFourBladed(MPI.COMM_WORLD)

num_vars = 48

# Set DV bounds
min_thickness  = np.array([5.0e-3]*num_vars) # minimum thickness of elements in m
max_thickness  = np.array([3.0e-2]*num_vars) # maximum thickness of elemetns in m

lb = np.zeros(num_vars, TACS.dtype)
ub = np.zeros(num_vars, TACS.dtype)
for i in xrange(num_vars):        
     lb[i] = min_thickness[i]
     ub[i] = max_thickness[i]
     
smoothness = False
if args.smoothness == 'yes':
	smoothness = True
        x = np.zeros(num_vars, TACS.dtype)

x = np.zeros(num_vars, TACS.dtype)
if smoothness is False:
    # start with a random design
    np.random.seed(seed=7)
    xrand = np.random.rand(num_vars)
    for i in xrange(num_vars):
        x[i] = min_thickness[i] + (max_thickness[i]-min_thickness[i])*1.0
else:
    # start with uniform thickness throughout
    for i in xrange(num_vars):
        x[i] = ub[i] #0.5*(lb[i] + ub[i])

lb[47] = 1.5e-2
x[47] = 1.5e-2

optproblem = PyoptRotorAssembly(MPI.COMM_WORLD, problem, num_vars, smoothness)
opt_prob = Optimization(args.logfile, optproblem.evalObjCon)

# Add functions
opt_prob.addObj('mass')
opt_prob.addCon('stress', type='i')
if smoothness is True:
    for i in xrange(num_vars-1):
        opt_prob.addCon('Smoothness %i a' % i,type='i')
        opt_prob.addCon('Smoothness %i b' % i,type='i')

# Add variables
for i in xrange(num_vars):
    opt_prob.addVar('thickness %i' % i, type='c',
                    value= x[i],
                    lower= lb[i],
                    upper= ub[i])

# Optimization algorithm
if args.algorithm == 'ALGENCAN':
    opt = ALGENCAN()
    opt.setOption('iprint',2)
    opt.setOption('epsfeas',1e-4)
    opt.setOption('epsopt',1e-3)
else:
    opt = SLSQP(pll_type='POA')
    opt.setOption('MAXIT',999)

opt(opt_prob,
    sens_type=optproblem.evalObjConGradient,
    disp_opts=True,
    store_hst=True,
    hot_start=False)

if optproblem.comm.Get_rank() ==0:   
    print opt_prob.solution(0)
    opt_prob.write2file(disp_sols=True)
