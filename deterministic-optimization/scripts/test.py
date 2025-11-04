import sys
sys.path.append('../')

import numpy as np
np.set_printoptions(precision=16)

from mpi4py import MPI
from tacs import functions, TACS, elements
from tacs_builder import TACSDynamicsProblem, TACSBodyType

#elements.setElementFDStepSize(1e-30)

#######################################################################
# Create a TACSDynamicsProblem
# problem.tacs --> created after initialization
# problem.helper --> helper to create TACS
# problem.integrator --> instance of integrator
#######################################################################

# Create two bladed rotor assembly problem
class TACSRotorAssemblyFourBladed(TACSDynamicsProblem):
    def __init__(self, comm, blade_type=TACSBodyType.SHELL):
        # invoke super class constructor
        super(self.__class__, self).__init__(comm)
        
        ##############################################################
        # Get the helper object for the problem. Add rigid/flex bodies
        # and constraints to the analysis using this 'helper'
        # instance. Helper instance keeps track of xpts, conn, ptr,
        # elem_list needed for creation of TACS.
        ##############################################################
    
        dtype = TACS.dtype
    
        ##############################################################
        # Define problem parameters
        ##############################################################
        
        speed = 109.12 # angular speed of rotor blades
        o     = np.array([0.0, 0.0, 0.0], dtype)   # origin point
        ex    = np.array([1.0, 0.0, 0.0], dtype)   # reference x axis
        ey    = np.array([0.0, 1.0, 0.0], dtype)   # reference y axis
        ez    = np.array([0.0, 0.0, 1.0], dtype)   # reference z axis
        
        #shaft = self.builder.rigidBody("hub4b.inp")
        #self.builder.addRevoluteDriver(ez, speed, shaft)

        # blade 0
        blade0 = self.builder.body("blade0.inp", blade_type)

        # Initialize TACS and integrator
        self.initialize()
        
        return

if __name__ == "__main__":
    # Create the rotor assembly problem
    problem = TACSRotorAssemblyFourBladed(MPI.COMM_WORLD, TACSBodyType.SHELL)

    # Create design variable values
    nvars = 192
    dvs = np.array([1.5e-2]*nvars, TACS.dtype)

    funcs = []
    funcs.append(functions.InducedFailure(problem.tacs, 100000.0))
    funcs.append(functions.KSFailure(problem.tacs, 100000.0))
    funcs.append(functions.Compliance(problem.tacs))
    funcs.append(functions.StructuralMass(problem.tacs))
    problem.march()
