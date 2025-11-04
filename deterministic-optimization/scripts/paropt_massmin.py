# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
import sys, traceback

# Import ParOpt
from paropt import ParOpt

# Import TACS for analysis
from pyoptimize import PyoptRotorAssembly
from assembly import TACSRotorAssemblyFourBladed

# Set DV bounds
min_thickness  = 1.0e-2
max_thickness  = 3.0e-2 

# Create the rosenbrock function class
class ParoptRotorAssemblyOpt(ParOpt.pyParOptProblem):
    def __init__(self, pyopt):
        # Set the communicator pointer
        self.pyopt = pyopt
        
        # The design history file
        self.x_hist = []

        super(ParoptRotorAssemblyOpt, self).__init__(pyopt.comm,
                                                     pyopt.nvars,
                                                     pyopt.ncon)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''
        Set the values of the bounds for design variables
        '''
        # Set the bounds
        for i in xrange(self.pyopt.nvars):        
            lb[i] = 1.0e-2
            ub[i] = 3.0e-2
            x[i] = 3.0e-2

        return

    def evalObjCon(self, x):
        
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        fobj, con, fail = self.pyopt.evalObjCon(np.array(x))
        
        return fail, fobj, con
    
if __name__ == "__main__":
    
    rotor = TACSRotorAssemblyFourBladed(MPI.COMM_WORLD)
    num_vars = 48
    smoothness = True
    pyoptproblem = PyoptRotorAssembly(MPI.COMM_WORLD, rotor, num_vars, smoothness)
    
    ##################################################################
    # Create the optimization problem for Paropt
    ##################################################################
    
    problem = ParoptRotorAssemblyOpt(pyoptproblem)

    # Set up the optimization problem using Paropt
    max_lbfgs = 20
    opt = ParOpt.pyParOpt(problem, max_lbfgs, ParOpt.BFGS)
    #opt.resetQuasiNewtonHessian()
    #opt.setInitBarrierParameter(0.1)
    #opt.setUseLineSearch(1)
    opt.setMaxMajorIterations(100)
    opt.setGradientCheckFrequency(0, 1.0e-8)
    opt.setOutputFile("4bladeopt_output.log")
    opt.optimize()
    
    # Get the final design point
    xopt = opt.getOptimizedPoint()

    # Print the design variables
    print "xopt=", np.array(xopt)
