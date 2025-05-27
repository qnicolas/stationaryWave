###########################################################################################
######### Sigma-level stationary wave model on the sphere                         #########
######### Inspired from Ting & Yu (1998, JAS).                                    #########
######### Model has a variable number of sigma levels (N)                         #########
######### and uses spherical harmonics in the horizontal.                         #########
######### Prognostic variables are horizontal velocity at each level (u,v)_i,     #########
######### temperature T_i, and perturbation-log-surface-pressure lnps.            #########
#########                                                                         #########
######### For now, the model is linear (nonlinear extension would be quite        #########
######### straightforward, contact qnicolas@berkeley.edu)                         #########
#########                                                                         #########
######### A steady-state should be reached after a few days of integration.       #########
#########                                                                         #########
######### To run: change SNAPSHOTS_DIR, then                                      #########
######### mpiexec -n {number of cores} python stationarywave_realgill.py {0 or 1} #########
######### the last argument stands for restart (0 for initial run, 1 for          #########
######### restart run)                                                            #########
###########################################################################################


import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import warnings
from mpi4py import MPI

import xarray as xr

SNAPSHOTS_DIR = "/Users/qnicolas/stationaryWave/data/" # Change this to your desired output directory

# Simulation units 
# The main point is to do the calculations on a sphere of unit radius, 
# which likely has better numerical properties than using a radius of 6.37122e6 
meter = 1 / 6.37122e6
hour = 1.
second = hour / 3600
day = hour*24
Kelvin = 1.
kilogram = 1.

# Global constants
consts = {'R0': 6.37122e6 * meter,  # Earth radius in meters
          'Omega': 2*np.pi/86400 / second,  # Earth's rotation rate in rad/s
          'g': 9.81 *  meter / second**2,  # Gravitational acceleration in m/s^2
          'cp': 1004. * meter**2 / second**2 / Kelvin,  # Specific heat at constant pressure in J/(kg*K)
          'Rd': 287. * meter**2 / second**2 / Kelvin,  # Specific gas constant for dry air in J/(kg*K)
          }

class StationaryWaveProblem:
    def __init__(self, resolution=32, Nsigma=12, linear=True, zonal_basic_state=True):
        self.resolution = resolution
        self.Nsigma = Nsigma
        self.zonal_basic_state = zonal_basic_state
        self.linear = linear

        # For debugging purposes, this is how to access the rank and size of the MPI communicator
        # world_rank = MPI.COMM_WORLD.Get_rank()
        # world_size = MPI.COMM_WORLD.Get_size()
        # print(f"Running on rank {world_rank} of {world_size}"
    
    def setup_bases(self):
        """Setup the spherical coordinates and bases for the simulation"""
        dtype = np.float64
        Nphi = self.resolution * 2  
        Ntheta = self.resolution 

        # Dealiasing factor indicates how many grid points to use in physical space
        if self.linear:
            dealias = (1, 1)
        else:
            dealias = (3/2, 3/2) # Delaiasing factor appropriate for order 2 nonlinearlities 

        # Bases
        self.coords = d3.S2Coordinates('phi', 'theta')
        self.dist = d3.Distributor(self.coords, dtype=dtype)#
        self.full_basis = d3.SphereBasis(self.coords, (Nphi, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)
        self.zonal_basis = d3.SphereBasis(self.coords, (1, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)

    # The following three functions are used in the formulation of the problem equations
    def _dlnps_dt(self):
        """Function to compute the log surface pressure tendency"""
        sum_us    = "+".join([f"u{j}"  for j in range(1,self.Nsigma+1)])
        sum_ubars = "+".join([f"ubar{j}" for j in range(1,self.Nsigma+1)])
        return f"(- deltasigma * (div({sum_us}) + ({sum_us})@grad(lnpsbar) + ({sum_ubars})@grad(lnps)))"

    def _sigmadot(self,i):
        """Function to compute the sigma-vertical-velocity at level i"""
        partialsum_us = "+".join([f"u{j}"  for j in range(1,i+1)])
        partialsum_ubars = "+".join([f"ubar{j}" for j in range(1,i+1)])
        return f"(-{i}*deltasigma*{self._dlnps_dt()} - deltasigma * (div({partialsum_us}) + ({partialsum_us})@grad(lnpsbar) + ({partialsum_ubars})@grad(lnps)))"

    def _Phiprime(self,i):
        """Function to compute the geopotential height (perturbation relative to surface geopotential height) at full level i (0 is model top, N is surface)"""
        if i==self.Nsigma:
            return 0.
        else:
            partialsum = "+".join([f"T{j}/({j}-0.5)" for j in range(i+1,self.Nsigma+1)])
            return f"(Rd * ({partialsum}))"

    def setup_problem(self):
        """Setup the vertical grid, all fields and problem equations"""
        # Vertical grid
        deltasigma = 1/self.Nsigma
        sigma = (np.arange(self.Nsigma) + 0.5)*deltasigma # Half levels
        self.deltasigma = deltasigma 
        self.sigma = sigma

        # hyperdiffusion & Rayleigh damping 
        nu = 40e15 * meter**4 / second       # Ting&Yu 1998 use 10^17
        epsilon = np.ones(self.Nsigma)/(15.*day)   #See equation 1 of Ting&Yu 1998
        epsilon[self.Nsigma-1] = 1/(0.2*day)       #See equation 1 of Ting&Yu 1998
        epsilon[self.Nsigma-2] = 1/(1.0*day)       #See equation 1 of Ting&Yu 1998

        # cross product by zhat times sin(latitude) for Coriolis force
        zcross = lambda A: d3.MulCosine(d3.skew(A))

        # Useful constants
        kappa = consts['Rd'] / consts['cp']  
        Rd = consts['Rd'] 
        Omega = consts['Omega'] 

        # Fields
        u_names        = [f"u{i}" for i in range(1,self.Nsigma+1)]             # Perturbation velocity
        T_names        = [f"T{i}" for i in range(1,self.Nsigma+1)]             # Perturbation temperature
        Qdiab_names    = [f"Qdiab{i}" for i in range(1,self.Nsigma+1)]         # Diabatic forcing

        ubar_names        = [f'ubar{i}' for i in range(1,self.Nsigma+1)]       # Basic-state velocity
        Tbar_names        = [f'Tbar{i}' for i in range(1,self.Nsigma+1)]       # Basic-state temperature
        sigmadotbar_names = [f"sigmadotbar{i}" for i in range(1,self.Nsigma)]  # Basic-state sigma-velocity
        
        # Instantiate prognostic fields
        us           = {name: self.dist.VectorField(self.coords, name=name, bases=self.full_basis) for name in u_names } # Perturbation velocity
        Ts           = {name: self.dist.Field(name=name, bases=self.full_basis) for name in T_names }                    # Perturbation temperature
        lnps         = self.dist.Field(name='lnps'  , bases=self.full_basis)                                             # Perturbation log-surface-pressure

        # Instantiate forcing fields
        Qdiabs       = {name: self.dist.Field(name=name, bases=self.full_basis) for name in Qdiab_names } # Diabatic forcing
        Phisfc       =  self.dist.Field(name='Phisfc', bases=self.full_basis)                      # Surface geopotential

        # Instantiate basic-state fields
        if self.zonal_basic_state:
            basis_basestate = self.zonal_basis
        else:
            basis_basestate = self.full_basis  
        ubars        = {name: self.dist.VectorField(self.coords, name=name, bases=basis_basestate) for name in ubar_names } # Basic-state velocity
        Tbars        = {name: self.dist.Field(name=name, bases=basis_basestate) for name in Tbar_names }                    # Basic-state temperature
        sigmadotbars = {name: self.dist.Field(name=name, bases=basis_basestate) for name in sigmadotbar_names }             # Basic-state sigma-velocity
        lnpsbar      =  self.dist.Field(name='lnpsbar'  , bases=basis_basestate)                                            # Basic-state log-surface-pressure

        # Gather all 3D variables 
        self.vars  = {**us, **Ts, **Qdiabs, **ubars, **Tbars, **sigmadotbars, 'lnps': lnps, 'lnpsbar': lnpsbar, 'Phisfc': Phisfc}

        ###################################################
        ################ PROBLEM EQUATIONS ################
        ###################################################

        self.namespace = (globals() | locals() | self.vars)
        self.problem = d3.IVP(list(us.values()) + list(Ts.values()) + [lnps,] , namespace=self.namespace)

        # log pressure equation
        if self.zonal_basic_state: #All terms with non-constant coefficients from the basic state go on the LHS
            self.problem.add_equation(f"dt(lnps) - {self._dlnps_dt()} = 0")
        else:
            self.problem.add_equation(f"dt(lnps) = {self._dlnps_dt()}")

        for i in range(1,self.Nsigma+1):
            # Build terms that involve vertical differentiation/staggering - different treatment for upper and lower boundaries
            if i==1:
                vert_advection_mom_1 = f"( sigmadotbar{i}*(u{i+1}-u{i})    )/deltasigma/2"
                vert_advection_mom_2 = f"( {self._sigmadot(i)}*(ubar{i+1}-ubar{i}) )/deltasigma/2"
                vert_advection_T_1   = f"( sigmadotbar{i}*(T{i+1}-T{i})    )/deltasigma/2"
                vert_advection_T_2   = f"( {self._sigmadot(i)}*(Tbar{i+1}-Tbar{i}) )/deltasigma/2"
                expansion = f"kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({self._sigmadot(i)})/2 + T{i}*(sigmadotbar{i})/2)"
            elif i==self.Nsigma:
                vert_advection_mom_1 = f"( sigmadotbar{i-1}*(u{i}-u{i-1})    )/deltasigma/2"
                vert_advection_mom_2 = f"( {self._sigmadot(i-1)}*(ubar{i}-ubar{i-1}) )/deltasigma/2"
                vert_advection_T_1   = f"( sigmadotbar{i-1}*(T{i}-T{i-1})    )/deltasigma/2"
                vert_advection_T_2   = f"( {self._sigmadot(i-1)}*(Tbar{i}-Tbar{i-1}) )/deltasigma/2"
                expansion = f"kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({self._sigmadot(i-1)})/2 + T{i}*(sigmadotbar{i-1})/2)"
            else:
                vert_advection_mom_1 = f"( sigmadotbar{i}*(u{i+1}-u{i}) + sigmadotbar{i-1}*(u{i}-u{i-1}) )/deltasigma/2"
                vert_advection_mom_2 = f"( {self._sigmadot(i)}*(ubar{i+1}-ubar{i}) + {self._sigmadot(i-1)}*(ubar{i}-ubar{i-1}) )/deltasigma/2"
                vert_advection_T_1   = f"( sigmadotbar{i}*(T{i+1}-T{i}) + sigmadotbar{i-1}*(T{i}-T{i-1}) )/deltasigma/2"
                vert_advection_T_2   = f"( {self._sigmadot(i)}*(Tbar{i+1}-Tbar{i}) + {self._sigmadot(i-1)}*(Tbar{i}-Tbar{i-1}) )/deltasigma/2"
                expansion = f"kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({self._sigmadot(i)}+{self._sigmadot(i-1)})/2 + T{i}*(sigmadotbar{i}+sigmadotbar{i-1})/2)"
                
            LHS_mom = f"dt(u{i}) \
                + epsilon[{i-1}] * u{i} \
                + nu * lap(lap(u{i})) \
                + grad( ({self._Phiprime(i-1)}+{self._Phiprime(i)}) / 2 ) \
                + 2 * Omega * zcross(u{i})"
            
            LHS_T = f"dt(T{i}) \
                + epsilon[{i-1}] * T{i} \
                + nu * lap(lap(T{i}))"

            minus_RHS_mom = f"( ubar{i}@grad(u{i})\
                + u{i}@grad(ubar{i})\
                + {vert_advection_mom_1}\
                + {vert_advection_mom_2} \
                + Rd * Tbar{i} * grad(lnps)\
                + Rd * T{i} * grad(lnpsbar))"
            
            minus_RHS_T = f"( ubar{i}@grad(T{i}) + u{i}@grad(Tbar{i})\
                + {vert_advection_T_1}\
                + {vert_advection_T_2}\
                - {expansion}\
                - kappa * Tbar{i} * {self._dlnps_dt()}\
                - kappa * T{i} * ubar{i}@grad(lnpsbar)\
                - kappa * Tbar{i} * u{i}@grad(lnpsbar)\
                - kappa * Tbar{i} * ubar{i}@grad(lnps) )"
            
            forcing_mom = "-grad(Phisfc)"

            forcing_T = f"Qdiab{i}"

            if self.zonal_basic_state: #All terms with non-constant coefficients from the basic state go on the LHS
                # Momentum equation
                self.problem.add_equation(LHS_mom + "+" + minus_RHS_mom + " = " + forcing_mom)
                # Thermodynamic equation
                self.problem.add_equation(LHS_T + "+" + minus_RHS_T + " = " + forcing_T)
            else:
                # Momentum equation
                self.problem.add_equation(LHS_mom + " = " + forcing_mom + " - " + minus_RHS_mom)
                # Thermodynamic equation
                self.problem.add_equation(LHS_T + " = " + forcing_T + " - " + minus_RHS_T)

    def initialize_basic_state_with_zeros(self):
        for i in range(1,self.Nsigma+1):
            self.vars[f'ubar{i}']['g'] = 0.
            self.vars[f'Tbar{i}']['g'] = (200. + self.sigma[i-1] * 100.) * Kelvin  # Example: linear decrease from 300K at surface to 200K at top

        # Basic-state log-surface pressure
        self.vars['lnpsbar']['g'] = 0.

        # Basic-state sigmadot (omega/ps - sigma * u.grad(lnps))
        for i in range(1,self.Nsigma):
            self.vars[f'sigmadotbar{i}']['g'] = 0.

    def initialize_basic_state_from_sigma_data(self,input_data):
        """Initialize the basic state fields.
        This function is used to initialize the basic state fields from a given 
        input data (given as xarray.Dataset) that has (sigma,lat,lon) coordinates.
        """
        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat_deg = (np.pi / 2 - theta + 0*phi) * 180 / np.pi 
        lon_deg = (phi-np.pi) * 180 / np.pi

        if self.zonal_basic_state:
            assert input_data.U.dims == ('sigma', 'lat'), "Input data must have dimensions (sigma, lat) for zonal basic state."
            local_grid_xr = xr.DataArray(np.zeros(len(lat_deg[0])),coords={'lat':lat_deg[0]},dims=['lat'])
            target_shape = (1, len(lat_deg[0]))
        else:
            assert input_data.U.dims == ('sigma', 'lat', 'lon'), "Input data must have dimensions (sigma, lat, lon) for non-zonal basic state."
            local_grid_xr = xr.DataArray(np.zeros(lat_deg.shape),coords={'lat':lat_deg,'lon':lon_deg},dims=['lat','lon'])
            target_shape = lat_deg.shape

        sigma_full = np.arange(1,self.Nsigma) * self.deltasigma
        sigma_full_xr = xr.DataArray(sigma_full,coords={'sigma':sigma_full},dims=['sigma'])
        sigma_half_xr = xr.DataArray(self.sigma,coords={'sigma':self.sigma},dims=['sigma'])

        basicstate_halflevs = input_data.interp_like(sigma_half_xr * local_grid_xr, method='linear')
        basicstate_fulllevs = input_data.interp_like(sigma_full_xr * local_grid_xr, method='linear')

        # Basic-state temperature, wind
        for i in range(1,self.Nsigma+1):
            self.vars[f'ubar{i}']['g'] = np.stack([ basicstate_halflevs.U.isel(sigma=i-1).data,
                                                   -basicstate_halflevs.V.isel(sigma=i-1).data ], axis=0).reshape((2,*target_shape)) * meter / second
            self.vars[f'Tbar{i}']['g'] = basicstate_halflevs.T.isel(sigma=i-1).data.reshape(target_shape) * Kelvin

        # Basic-state log-surface pressure
        self.vars['lnpsbar']['g'] = np.log(basicstate_halflevs.SP.data/1e5).reshape(target_shape)

        # Basic-state sigmadot (omega/ps - sigma * u.grad(lnps))
        omega_ov_ps = basicstate_fulllevs.W/basicstate_fulllevs.SP * 1/second
        for i in range(1,self.Nsigma):
            self.vars[f'sigmadotbar{i}']['g'] = omega_ov_ps.isel(sigma=i-1).data.reshape(target_shape)
            self.vars[f'sigmadotbar{i}'] = self.vars[f'sigmadotbar{i}']\
                - sigma_full[i-1] * (self.vars[f'ubar{i}']+self.vars[f'ubar{i+1}']) @ d3.grad(self.vars['lnpsbar']) / 2

    def initialize_forcings(self):
        """Initialize the forcing fields"""
        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat = np.pi / 2 - theta + 0*phi
        lon = phi-np.pi

        # Topographic forcing
        self.vars['Phisfc']['g'] = 0. * meter * consts['g'] 
        
        # Heating forcing
        Q0 = 2. * Kelvin / day # 2 K/day heating rate
        for i in range(1,self.Nsigma+1):
            deltalat = 10 * np.pi/180
            deltalon = 35 * np.pi/180
            self.vars[f'Qdiab{i}']['g'] = Q0 * np.sin(np.pi * self.sigma[i-1]) * np.pi/2 * np.cos(np.pi * lat / (2 * deltalat))**2 * np.cos(np.pi * lon / (2 * deltalon))**2 * (np.abs(lat)<deltalat) * (np.abs(lon)<deltalon)

    def _configure_snapshots(self):

        # Add all state variables, vorticity, forcings, and background state to snapshots
        # You may want to omit some of these to reduce output size
        self.snapshots.add_task(self.vars['lnps'])
        self.snapshots.add_task(self.vars['lnpsbar'])
        self.snapshots.add_task(self.vars['Phisfc'] / (meter**2 / second**2), name = 'Phisfc') 

        for i in range(1,self.Nsigma+1):
            # u, T perturbations
            self.snapshots.add_task(self.vars[f'u{i}'] / (meter / second), name=f'u{i}')
            self.snapshots.add_task(self.vars[f'T{i}'] / Kelvin, name=f'T{i}')

            # Vorticity
            self.snapshots.add_task(-d3.div(d3.skew(self.vars[f'u{i}'])) / (1/second), name=f'zeta{i}') 

            # Diabatic heating
            if not self.zonal_basic_state:  # otherwise a bug in Dedalus prevents us from outputting Qdiab for now
                self.snapshots.add_task(self.vars[f'Qdiab{i}'] / Kelvin * day, name=f'Qdiab{i}')

            # Basic-state variables
            self.snapshots.add_task(self.vars[f'ubar{i}'] / (meter / second), name=f'ubar{i}')
            self.snapshots.add_task(self.vars[f'Tbar{i}'] / Kelvin, name=f'Tbar{i}')
            if i < self.Nsigma:  # sigmadotbar is only defined for levels 1 to N-1:
                self.snapshots.add_task(self.vars[f'sigmadotbar{i}'] / (1/second), name=f'sigmadotbar{i}')
            
        # Add sigma and pressure velocity to snapshots  
        p0 = 1e5 * kilogram / meter / second ** 2  # Reference surface pressure in Pa
        for i in range(1,self.Nsigma+1):
            # What I'm doing here is to calculate sigmadot at level i. 
            # The sigmadot function does exactly that, but outputs a string that goes in the equation formulation
            # What we simply need to do is to transform this string into an actual expression that acts on the fields
            # This is what 'eval' does.
            # However with the current namespace, the operators 'div' and 'grad' are not imported
            # Thus I simpy replace them with d3.div and d3.grad
            sigmadot_i = eval(self._sigmadot(i).replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
            self.snapshots.add_task(sigmadot_i / (1/second), name=f'sigmadot{i}') 
            
            if not self.zonal_basic_state: #otherwise a bug in Dedalus prevents us from outputting omega for now
                # basic-state pressure velocity
                sigma_i = (i-0.5)*self.deltasigma
                sigmadotbar_i = self.vars[f"sigmadotbar{i}"]
                Dlnpsbar_Dt = eval(f"(ubar{i}+ubar{i+1})@d3.grad(lnpsbar)/2",self.namespace)
                omegabar_i = p0 * np.exp(self.vars["lnpsbar"]) * ( sigma_i * Dlnpsbar_Dt + sigmadotbar_i)
                
                # total pressure velocity
                Dlnpstot_Dt = eval(f"{self._dlnps_dt()}\
                                    + (ubar{i}+ubar{i+1})@grad(lnpsbar)/2\
                                    + (u{i}+u{i+1})@grad(lnpsbar)/2\
                                    + (ubar{i}+ubar{i+1})@grad(lnps)/2".replace('div','d3.div').replace('grad','d3.grad')
                                  ,self.namespace)
                
                omegatot_i = p0 * np.exp(self.vars["lnpsbar"]+self.vars["lnps"]) * ( sigma_i * Dlnpstot_Dt + sigmadotbar_i + sigmadot_i)
                
                # output perturbation pressure velocity
                self.snapshots.add_task((omegatot_i-omegabar_i) / (kilogram / meter / second**3), name=f'omega{i}')
            
    def integrate(self, restart=False, restart_id='s1', use_CFL=False, safety_CFL=0.8, timestep=400, stop_sim_time=5*86400):
        """Integrate the problem. 
        args:
            - restart: bool, whether this is a restart run
            - restart_id: str, identifier for the restart snapshot (e.g. 's1', 's2', etc.)
            - use_CFL: bool, whether to use an adaptive timestep based on the CFL condition
            - safety_CFL: float, fraction of max timestep allowed by the CFL condition to use 
            - timestep: float, initial timestep in seconds
            - stop_sim_time: float, simulation time in seconds until which to integrate
        """
        # Solver
        self.solver = self.problem.build_solver(d3.RK222)
        self.solver.stop_sim_time = stop_sim_time * second
        timestep = timestep * second

        if not restart:
            file_handler_mode = 'overwrite'
        else:
            write, initial_timestep = self.solver.load_state(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
            timestep = min(timestep, initial_timestep)
            file_handler_mode = 'append'

        snapshot_id = f'stationarywave_{self.Nsigma}level_T{self.resolution}_realgill'


        # Save snapshots every 6h of model time
        self.snapshots = self.solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=3600 * second, mode=file_handler_mode)
        self._configure_snapshots()

        # CFL
        CFL = d3.CFL(self.solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1, max_dt = 0.2)
        # Need to add all velocities and sigmadots - not sure if worth it
        # CFL.add_velocity(us[0])

        # Main loop
        with warnings.catch_warnings():
            warnings.filterwarnings('error',category=RuntimeWarning)
            try:
                logger.info('Starting main loop')
                while self.solver.proceed:
                    if use_CFL:
                        timestep = CFL.compute_timestep()
                    self.solver.step(timestep)
                    if (self.solver.iteration-1) % 20 == 0:
                        logger.info('Iteration=%i, Time=%e, dt=%e' %(self.solver.iteration, self.solver.sim_time, timestep))
            except:
                logger.info('Last dt=%e' %(timestep))
                logger.error('Exception raised, triggering end of main loop.')
                raise
            finally:
                self.solver.log_stats()

# ####################################################
# ###### SAVE THIS .py FILE TO OUTPUT DIRECTORY ######
# ####################################################
# import os;import shutil;
# from pathlib import Path
# if self.dist.comm.rank == 0:
#     Path(SNAPSHOTS_DIR+snapshot_id).mkdir(parents=True, exist_ok=True)
#     shutil.copyfile(os.path.abspath(__file__), SNAPSHOTS_DIR+snapshot_id+'/'+os.path.basename(__file__))

# if __name__ == "__main__":
#     idealgill_linear = StationaryWaveProblem(resolution=8, Nsigma=4, linear=True, zonal_basic_state=True)
#     idealgill_linear.setup_bases()
#     idealgill_linear.setup_problem()
#     idealgill_linear.initialize_basic_state_with_zeros()
#     idealgill_linear.initialize_forcings()
#     idealgill_linear.integrate(restart=False, use_CFL=False, timestep=400, stop_sim_time=20*86400)

