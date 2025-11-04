# Import some utilities
from pathlib import Path
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
#from pyOpt import SLSQP
from pyOpt import ALGENCAN

# Import TACS for analysis
from tacs import TACS, functions
from collective import TACSRotorAssemblyFourBladedCollective
from lateral_cyclic import TACSRotorAssemblyFourBladedLatCyclic
from longitudinal_cyclic import TACSRotorAssemblyFourBladedLonCyclic

# Optimization settings
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smoothness', type=str, default='yes', help='yes/no')
parser.add_argument('--logfile'   , type=str, default='stresscon-massmin-smooth', help='log file for optimizer')
parser.add_argument('--algorithm' , type=str, default='ALGENCAN', help='SLSQP/ALGENCAN')
args = parser.parse_args()

# Pickle for writing data
import pickle
from postopt import PostOpt

# centralize solver artifacts (ALGENCAN history, logs)
SOLVER_ARTIFACTS = Path(__file__).resolve().parents[1] / "solver_artifacts"
# ensure directory exists when running fresh
SOLVER_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Create the rosenbrock function class
class PyoptRotorAssembly:
    def __init__(self, comm,
                 scenarios,
                 num_vars,
                 smoothness=True):
        self.comm = comm
        self.nvars = num_vars
        self.scenarios = scenarios
        self.funcs = []

        # Get tacs from each 'scenario' and create structural
        # objective and constraints
        snum = 0
        for scenario in self.scenarios:
            sfuncs = []
            # Use mass with only one scenario
            if snum == 0 :
                sfuncs.append(functions.StructuralMass(scenario.tacs))
            sfuncs.append(functions.KSFailure(scenario.tacs, 100000.0))
            scenario.integrator.setFunctions(sfuncs, num_vars)
            self.funcs.append(sfuncs)
            snum += 1

        self.smoothness = smoothness

        if self.smoothness is True:
            self.ncon = len(self.scenarios) + 2*(num_vars-1)
            if self.ncon <= 0:
                raise "not many variables to enforce smoothness constaints"
            self.delta = 1.0e-3 # 1mm smoothness constraint
            print(self.ncon, "constraints")
        else:
            self.ncon = len(self.scenarios)

        # The design history file
        self.x_hist = []

        # Control redundant evalution of cost and constraint
        # gradients. Evaluate the gradient only if X has changed from
        # previous X.
        self.currentX = None

        # Space for current function and gradient values
        self.funcVals = np.zeros([3, 2], TACS.dtype)
        self.gradVals = np.zeros([3, 2*self.nvars], TACS.dtype)

        # Store constraint history and objective
        self.hist = []

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

        snum = 0
        for scenario in self.scenarios:
            try:
                print(" >> scenario", snum)
                self.currentX = x
                funclist = self.funcs[snum]
                self.funcVals[snum,:] = scenario.getFuncGrad(np.array(x),
                                                             funclist,
                                                             self.gradVals[snum,:],
                                                             True)
                snum += 1
            except:
                print( "exception occurred in funceval")
                traceback.print_exc(file=sys.stdout)
                fail = 1
                stop

        # Store the objective function value
        fobj = self.funcVals[0,0]

        # Store the constraint values
        con = [0.0]*self.ncon

        # 1 - sigma/sigma_max >= 0
        con[0] = 4.0*self.funcVals[0,1] - 1.0
        con[1] = 4.0*self.funcVals[1,0] - 1.0
        con[2] = 4.0*self.funcVals[2,0] - 1.0

        print("failures:", con[0], con[1], con[2])

        if self.smoothness is True:
            con[3:self.ncon] = self.evalSmoothness(x)

        print("fobj", fobj)
        print("con", con)

        # Return the values
        return fobj, con, fail

    def evalObjConGradient(self, x, g, A):
        '''
        Evaluate the objective and constraint gradient
        '''
        assert(np.all(x>0.0)==1)

        # Set the fail flag
        fail = 0

        print("evaluating gradient")

        # Same design point. So do only reverse mode
        if np.array_equal(self.currentX,x) is True:
            snum = 0
            for scenario in self.scenarios:
                print(" >> scenario at same x", snum)
                self.currentX = x
                funclist = self.funcs[snum]

                # March backward and evaluate gradient
                self.gradVals[snum,:] = 0.0
                scenario.integrator.integrateAdjoint()
                scenario.integrator.getGradient(self.gradVals[snum,:])

                snum += 1
        else:
            # New design point. So do forward and reverse modes
            snum = 0
            for scenario in self.scenarios:
                print(" >> scenario at new x", snum)
                self.currentX = x
                funclist = self.funcs[snum]
                self.funcVals[snum,:] = scenario.getFuncGrad(np.array(x),
                                                             funclist,
                                                             self.gradVals[snum,:],
                                                             False)
                snum += 1

        # Set the objective gradient
        g = [0.0]*self.nvars
        g[0:self.nvars] = self.gradVals[0,0:self.nvars]

        # Set the constraint gradient
        A = np.zeros([self.ncon, self.nvars], TACS.dtype)
        A[0,:] = 4.0*self.gradVals[0, 1*self.nvars:2*self.nvars]
        A[1,:] = 4.0*self.gradVals[1, 0*self.nvars:1*self.nvars]
        A[2,:] = 4.0*self.gradVals[2, 0*self.nvars:1*self.nvars]

        # Append the gradient of smoothness constraints
        if self.smoothness is True:
            sgrad = self.evalSmoothnessGrad(x)
            for c in range(3,self.ncon):
                A[c,:] = sgrad[c-3,:]

        #print "g=", g
        #print "A=", A

        # Create a list with function and constraint values at this
        # iteration
        hist = [ len(self.hist) + 1,
                 self.funcVals[0,0],
                 4.0*self.funcVals[0,1] - 1.0,
                 4.0*self.funcVals[1,0] - 1.0,
                 4.0*self.funcVals[2,0] - 1.0 ]

        # Append to global list
        self.hist.append(hist)

        print("hist", hist)

        return g, A, fail

    def evalSmoothness(self, x):
        # There are as many smoothness cons as nvars-1
        con = np.zeros(2*(self.nvars-1), TACS.dtype)
        j = 0 # tracks variable
        k = 0 # tracks constraint
        for i in range(self.nvars-1):
            if j == 47: # or j == 95 or j == 143 or j == 191:
                j += 1
            #con[k  ] = x[j]-x[j+1] - self.delta
            #con[k+1] = x[j+1]-x[j] - self.delta

            con[k  ] = (x[j]   - x[j+1])/self.delta - 1.0
            con[k+1] = (x[j+1] - x[j]  )/self.delta - 1.0

            k += 2
            j += 1
        return con

    def evalSmoothnessGrad(self,x):
        grad = np.zeros([2*(self.nvars-1),self.nvars], TACS.dtype)
        j = 0
        k = 0
        for i in range(self.nvars-1):
            if j == 47: # or j == 95 or j == 143 or j == 191:
                j += 1
            grad[k,j    ] = 1.0/self.delta
            grad[k,j+1  ] = -1.0/self.delta
            grad[k+1,j  ] = -1.0/self.delta
            grad[k+1,j+1] = 1.0/self.delta
            j += 1
            k += 2
        return grad

if __name__ == "__main__":

    print("Multiscenario optimization of rotor blade thickness")

    ## ######################################################################
    ## # Setup the optimization
    ## ######################################################################

    coll = TACSRotorAssemblyFourBladedCollective(MPI.COMM_WORLD)
    lonc = TACSRotorAssemblyFourBladedLonCyclic(MPI.COMM_WORLD)
    latc = TACSRotorAssemblyFourBladedLatCyclic(MPI.COMM_WORLD)

    problist = [coll, lonc, latc]

    num_vars = 48

    # Set DV bounds
    min_thickness = np.array([1.0e-2]*num_vars) # minimum thickness of elements in m
    max_thickness = np.array([2.0e-2]*num_vars) # maximum thickness of elemetns in m

    lb = np.zeros(num_vars, TACS.dtype)
    ub = np.zeros(num_vars, TACS.dtype)
    for i in range(num_vars):
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
        for i in range(num_vars):
            x[i] = min_thickness[i] + (max_thickness[i]-min_thickness[i])*xrand[i]
    else:
        # start with uniform thickness throughout
        for i in range(num_vars):
            x[i] = ub[i] # 0.5*(lb[i] + ub[i])

    ## x = np.array([0.005000, 0.005000, 0.005000, 0.005000, 0.005000,
    ##               0.005000, 0.005000, 0.005000, 0.005000, 0.005000,
    ##               0.005000, 0.005000, 0.005000, 0.005000, 0.005000,
    ##               0.005000, 0.005000, 0.005000, 0.005000, 0.005000,
    ##               0.005000, 0.005000, 0.005000, 0.005000, 0.005000,
    ##               0.005000, 0.005000, 0.005050, 0.005179, 0.005307,
    ##               0.005430, 0.005550, 0.005666, 0.005779, 0.005887,
    ##               0.005987, 0.006089, 0.006195, 0.006242, 0.007000,
    ##               0.008000, 0.009000, 0.010000, 0.011000, 0.012000,
    ##               0.013000, 0.014000, 0.015000], TACS.dtype)

    optproblem = PyoptRotorAssembly(MPI.COMM_WORLD, problist, num_vars,
                                    smoothness)
    opt_prob = Optimization(args.logfile, optproblem.evalObjCon)

    # Add functions
    opt_prob.addObj('mass')
    opt_prob.addCon('collective stress', type='i')
    opt_prob.addCon('longitudinal stress', type='i')
    opt_prob.addCon('lateral stress', type='i')
    if smoothness is True:
        for i in range(num_vars-1):
            opt_prob.addCon('Smoothness %i a' % i,type='i')
            opt_prob.addCon('Smoothness %i b' % i,type='i')

    # Add variables
    for i in range(num_vars):
        opt_prob.addVar('thickness %i' % i, type='c',
                        value= x[i],
                        lower= lb[i],
                        upper= ub[i])

    # Optimization algorithm
    history_base = SOLVER_ARTIFACTS / "ALGENCAN"
    history_base_str = str(history_base)

    if args.algorithm == 'ALGENCAN':
        opt = ALGENCAN()
        opt.setOption('iprint',2)
        opt.setOption('epsfeas',1e-4)
        opt.setOption('epsopt',1e-3)
    else:
        opt = SLSQP(pll_type='POA')
        opt.setOption('MAXIT',999)

    if args.algorithm == 'ALGENCAN':
        hot_start_path = history_base_str if (history_base.with_suffix('.bin').exists() and history_base.with_suffix('.cue').exists()) else False
        opt(opt_prob,
            sens_type=optproblem.evalObjConGradient,
            disp_opts=True,
            store_hst=history_base_str,
            hot_start=hot_start_path)
    else:
        opt(opt_prob,
            sens_type=optproblem.evalObjConGradient,
            disp_opts=True,
            store_hst=True,
            hot_start=True)

    if optproblem.comm.Get_rank() ==0:
        print (opt_prob.solution(0))
        opt_prob.write2file(disp_sols=True)

    if optproblem.comm.Get_rank() ==0:
        print("storing optimization history")
        with open('opt.hist', 'wb') as fp:
            pickle.dump(optproblem.hist, fp)

        # Plot using post processor
        # hist = np.array(optproblem.hist)
        # PostOpt.plot_history(hist, "history.pdf")
