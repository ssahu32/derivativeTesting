from __future__ import print_function
import os

from pprint import pprint
import numpy as np
from mpi4py import MPI

from tacs import TACS, functions, constitutive, elements, pyTACS, problems
from pyfuntofem.tacs_interface import TacsSteadyInterface

class wedgeTACS(TacsSteadyInterface):
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(wedgeTACS,self).__init__(comm, tacs_comm, model)

        assembler = None
        self.tacs_proc = False
        if comm.Get_rank() < n_tacs_procs:
            self.tacs_proc = True

            # Instantiate FEASolver
            structOptions = {
                'printtiming':True,
            }

            bdfFile = os.path.join(os.path.dirname(__file__), 'stiffPanel7.dat')
            FEASolver = pyTACS(bdfFile, options=structOptions, comm=tacs_comm)

            # Material properties
            rho = 2780.0        # density kg/m^3
            E = 73.1e9          # Young's modulus (Pa)
            nu = 0.33           # Poisson's ratio
            kcorr = 5.0/6.0     # shear correction factor
            ys = 324.0e6        # yield stress
            specific_heat = 920.096
            cte = 24.0e-6
            kappa = 230.0

            tInput = 0.001*np.ones(12)

            def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
                elemIndex = kwargs['propID'] - 1
                t = tInput[elemIndex]

                prop = constitutive.MaterialProperties(rho=rho, specific_heat=specific_heat,
                                                       E=E, nu=nu, ys=ys, cte=cte, kappa=kappa)
                con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

                refAxis = np.array([1.0, 0.0, 0.0])

                elemList = []
                transform = None
                for elemDescript in elemDescripts:
                    if elemDescript in ['CQUAD4', 'CQUADR']:
                        elem = elements.Quad4ThermalShell(transform, con)
                    else:
                        print("Uh oh, '%s' not recognized" % (elemDescript))
                    elemList.append(elem)

                # Add scale for thickness dv
                scale = [100.0]
                return elemList, scale

            # Set up elements and TACS assembler
            FEASolver.initialize(elemCallBack)
            assembler = FEASolver.assembler

            # # Add back pressure
            # forces = assembler.createVec()
            # force_array = forces.getArray()

            # # Panel pressure loading
            # force_array[2::7] += 50*np.cos(np.radians(5))
            # force_array[0::7] += -50*np.sin(np.radians(5))
            
            # assembler.setBCs(forces)

        self._initialize_variables(assembler, thermal_index=6)

        self.initialize(model.scenarios[0],model.bodies)

    def post_export_f5(self):
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_STRAINS |
                TACS.OUTPUT_STRESSES |
                TACS.OUTPUT_EXTRAS)
        f5 = TACS.ToFH5(self.assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
        f5.writeToFile('stiffPanel.f5')
