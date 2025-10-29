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
class TACSRotorAssemblyFourBladedCollective(TACSDynamicsProblem):
    def __init__(self, comm, blade_type=TACSBodyType.RIGID,
                 flexLink=False):
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

        # central shaft
        shaft = self.builder.rigidBody("hub4b.inp")
        self.builder.addRevoluteDriver(ez, speed, shaft)

        # swash plates
        lsp = self.builder.rigidBody("lsp.inp")
        usp = self.builder.rigidBody("usp4b.inp")
        self.builder.addRevoluteConstraint(np.array([0.0, 0.0, 0.4], dtype), ez, lsp, usp)

        # sphere
        sphere = self.builder.rigidBody("sphere.inp")
        self.builder.addSphericalConstraint(np.array([0.0, 0.0, 0.4]), lsp, sphere)
        self.builder.addCylindricalConstraint(np.array([0.0, 0.0, 0.4]), ez, sphere)

        # pitch links
        lpl30 = self.builder.rigidBody("lpl30.inp")
        usp_lpl30_ball_point = np.array([0.278167359696, 0.160600, 0.43], dtype)
        self.builder.addSphericalConstraint(usp_lpl30_ball_point, usp, lpl30)

        lpl120 = self.builder.rigidBody("lpl120.inp")
        usp_lpl120_ball_point = np.array([-0.160600, 0.278167359696, 0.43], dtype)
        self.builder.addSphericalConstraint(usp_lpl120_ball_point, usp, lpl120)

        lpl210 = self.builder.rigidBody("lpl210.inp")
        usp_lpl210_ball_point = np.array([-0.278167359696, -0.160600, 0.43], dtype)
        self.builder.addSphericalConstraint(usp_lpl210_ball_point, usp, lpl210)

        lpl300 = self.builder.rigidBody("lpl300.inp")
        usp_lpl300_ball_point = np.array([0.160600, -0.278167359696, 0.43], dtype)
        self.builder.addSphericalConstraint(usp_lpl300_ball_point, usp, lpl300)

        upl30 = self.builder.rigidBody("upl30.inp")
        lpl30_upl30_hinge_point = np.array([0.278167359696, 0.160600, 0.90], dtype)
        self.builder.addSphericalConstraint(lpl30_upl30_hinge_point, lpl30, upl30)

        upl120 = self.builder.rigidBody("upl120.inp")
        lpl120_upl120_hinge_point = np.array([-0.160600, 0.278167359696, 0.90], dtype)
        self.builder.addSphericalConstraint(lpl120_upl120_hinge_point, lpl120, upl120)

        upl210 = self.builder.rigidBody("upl210.inp")
        lpl210_upl210_hinge_point = np.array([-0.278167359696, -0.160600, 0.90], dtype)
        self.builder.addSphericalConstraint(lpl210_upl210_hinge_point, lpl210, upl210)

        upl300 = self.builder.rigidBody("upl300.inp")
        lpl300_upl300_hinge_point = np.array([0.160600, -0.278167359696, 0.90], dtype)
        self.builder.addSphericalConstraint(lpl300_upl300_hinge_point, lpl300, upl300)

        shaft_upl30_hinge_point = np.array([0.231, 0.0, 0.9], dtype)
        self.builder.addRevoluteConstraint(shaft_upl30_hinge_point, ex, shaft, upl30)

        shaft_upl120_hinge_point = np.array([0.0, 0.231, 0.9], dtype)
        self.builder.addRevoluteConstraint(shaft_upl120_hinge_point, ey, shaft, upl120)

        shaft_upl210_hinge_point = np.array([-0.231, 0.0, 0.9], dtype)
        self.builder.addRevoluteConstraint(shaft_upl210_hinge_point, ex, shaft, upl210)

        shaft_upl300_hinge_point = np.array([0.0, -0.231, 0.9], dtype)
        self.builder.addRevoluteConstraint(shaft_upl300_hinge_point, ey, shaft, upl300)

        # blade 0
        bcap0 = self.builder.rigidBody("bcap0.inp")
        self.builder.addRigidLink(upl30, bcap0)
        blade0 = self.builder.body("blade0.inp", blade_type)
        if flexLink is not True:
            self.builder.addRigidLink(bcap0, blade0)
        else:
            moment_flag = 1
            loc = np.array([0.44,0.0,0.9])
            orig = elements.GibbsVector(loc[0], loc[1], loc[2])
            xaxis = elements.GibbsVector(loc[0]+1, loc[1], loc[2])
            yaxis = elements.GibbsVector(loc[0], loc[1]+1, loc[2])
            frame = elements.RefFrame(orig, xaxis, yaxis)
            self.builder.addFlexLink(bcap0, blade0, loc, frame, moment_flag)

        # blade 90
        bcap90 = self.builder.rigidBody("bcap90.inp")
        self.builder.addRigidLink(upl120, bcap90)
        blade90 = self.builder.body("blade90.inp", blade_type)
        if flexLink is not True:
            self.builder.addRigidLink(bcap90, blade90)
        else:
            moment_flag = 2
            loc = np.array([0.0, 0.44, 0.9])
            orig = elements.GibbsVector(loc[0], loc[1], loc[2])
            xaxis = elements.GibbsVector(loc[0]+1, loc[1], loc[2])
            yaxis = elements.GibbsVector(loc[0], loc[1]+1, loc[2])
            frame = elements.RefFrame(orig, xaxis, yaxis)
            self.builder.addFlexLink(bcap90, blade90, loc, frame, moment_flag)

        # blade 180
        bcap180 = self.builder.rigidBody("bcap180.inp")
        self.builder.addRigidLink(upl210, bcap180)
        blade180 = self.builder.body("blade180.inp", blade_type)
        if flexLink is not True:
            self.builder.addRigidLink(bcap180, blade180)
        else:
            moment_flag = 1
            loc = np.array([-0.44, 0.0, 0.9])
            orig = elements.GibbsVector(loc[0], loc[1], loc[2])
            xaxis = elements.GibbsVector(loc[0]+1, loc[1], loc[2])
            yaxis = elements.GibbsVector(loc[0], loc[1]+1, loc[2])
            frame = elements.RefFrame(orig, xaxis, yaxis)
            self.builder.addFlexLink(bcap180, blade180, loc, frame, moment_flag)

        # blade 270
        bcap270 = self.builder.rigidBody("bcap270.inp")
        self.builder.addRigidLink(upl300, bcap270)
        blade270 = self.builder.body("blade270.inp", blade_type)
        if flexLink is not True:
            self.builder.addRigidLink(bcap270, blade270)
        else:
            moment_flag = 2
            loc = np.array([0.0, -0.44, 0.9])
            orig = elements.GibbsVector(loc[0], loc[1], loc[2])
            xaxis = elements.GibbsVector(loc[0]+1, loc[1], loc[2])
            yaxis = elements.GibbsVector(loc[0], loc[1]+1, loc[2])
            frame = elements.RefFrame(orig, xaxis, yaxis)
            self.builder.addFlexLink(bcap270, blade270, loc, frame, moment_flag)

        # push rod at 90 degrees
        prod90 = self.builder.rigidBody("prod90.inp")
        lsp_prod90_ball_point = np.array([0.000000,  0.400400, 0.380000], dtype)
        self.builder.addSphericalConstraint(lsp_prod90_ball_point, prod90, lsp)

        # push rod at 180 degrees
        prod180 = self.builder.rigidBody("prod180.inp")
        lsp_prod180_ball_point = np.array([-0.40040,  0.000000, 0.380000], dtype)
        self.builder.addSphericalConstraint(lsp_prod180_ball_point, prod180, lsp)

        # push rod at 270 degrees
        prod270 = self.builder.rigidBody("prod270.inp")
        lsp_prod270_ball_point = np.array([0.000000, -0.400400, 0.380000], dtype)
        self.builder.addSphericalConstraint(lsp_prod270_ball_point, prod270, lsp)

        dirn = 0.05*ez
        self.builder.addMotionDriver(dirn, 0.25*speed, prod90)

        dirn = 0.05*ez
        self.builder.addMotionDriver(dirn, 0.25*speed, prod180)

        dirn = 0.05*ez
        self.builder.addMotionDriver(dirn, 0.25*speed, prod270)

        bsp = self.builder.rigidBody("bsp.inp")
        self.builder.addFixedConstraint(o, bsp)

        # push horns
        lph = self.builder.rigidBody("lph.inp")
        bsp_lph_hinge_point = np.array([0.4004, 0.0, 0.05], dtype)
        self.builder.addRevoluteConstraint(bsp_lph_hinge_point, ey, lph, bsp)

        uph = self.builder.rigidBody("uph.inp")
        lph_uph_hinge_point = np.array([0.565400, 0.000000, 0.215000], dtype)
        self.builder.addRevoluteConstraint(lph_uph_hinge_point, ey, lph, uph)

        uph_lsp_ball_point = np.array([0.400400, 0.000000, 0.380000])
        self.builder.addSphericalConstraint(uph_lsp_ball_point, uph, lsp)

        # Initialize TACS and integrator
        self.initialize()

        return

if __name__ == "__main__":
    # Create the rotor assembly problem
    problem = TACSRotorAssemblyFourBladedCollective(MPI.COMM_WORLD,
                                                    TACSBodyType.RIGID,
                                                    False)
    # Create design variable values
    nvars = 48
    dvs = np.array([2.0e-2]*nvars, TACS.dtype)
    problem.tacs.setDesignVars(dvs)

    funcs = []
    funcs.append(functions.InducedFailure(problem.tacs, 100000.0))
    funcs.append(functions.KSFailure(problem.tacs, 100000.0))
    funcs.append(functions.Compliance(problem.tacs))
    funcs.append(functions.StructuralMass(problem.tacs))

    problem.integrator.integrate()
