##########################################################################################
# Sigma-level stationary wave model on the sphere                                        #
# Inspired from Ting & Yu (1998, JAS).                                                   #
# Model has a variable number of sigma levels (N)                                        #
# and uses spherical harmonics in the horizontal.                                        #
# Prognostic variables are horizontal velocity at each level (u,v)_i,                    #
# temperature T_i, and perturbation-log-surface-pressure lnps.                           #
#                                                                                        #
# TODO: Nonlinear extension, add some attributes to the output,                          #
# add nonuniform sigma levels                                                            #
#                                                                                        #
# A steady-state should be reached after a few days of integration, typically 20-50      #
#                                                                                        #
# Example run script:                                                                    #
#                                                                                        #
## sph_resolution = 16; Nsigma = 5; linear = True; zonal_basic_state = True              #
## output_dir = "data/"; case_name = "test"                                              #
## exampleRun = StationaryWaveProblem(sph_resolution, Nsigma, linear, zonal_basic_state, #
##                                    output_dir, case_name)                             #
## basicstate = xr.open_dataset("example_basicstate.nc")                                 #
## exampleRun.initialize_basic_state_from_pressure_data(basicstate)                      #
## forcing = xr.open_dataset("example_forcing.nc")                                       #
## exampleRun.initialize_forcings_from_pressure_data(forcing)                            #
## exampleRun.integrate()                                                                #
#                                                                                        #
# To run in parallel, call                                                               #
# mpiexec -n {number of cores} python example_script.py                                  #
# Author: Q. Nicolas                                                                     #
##########################################################################################

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import warnings
from mpi4py import MPI

import xarray as xr

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
    """
    Class to setup and run a stationary wave problem on the sphere.
    
    Handles the setup of spherical coordinates and bases, defines the problem equations,
    initializes the basic state and forcing fields, and integrates the problem.
    """
    def __init__(self, resolution, Nsigma, linear, zonal_basic_state, output_dir, case_name, 
                 hyperdiffusion_coefficient=1e17, 
                 rayleigh_damping_timescale=25, 
                 newtonian_cooling_timescale=15
                ):
        """
        Initialize the stationary wave problem with given parameters.

        Parameters:
        ------------
        resolution : int
            Maximum degree & order in the spherical harmonic expansion. Akin to the order of a triangular truncation.
        Nsigma : int
            Number of half sigma levels in the vertical grid.
        linear : bool 
            Whether to include nonlinear terms in the equations.
        zonal_basic_state : bool
            Whether the basic state is zonally-invariant (i.e., only depends on latitude).
        output_dir : str
            Directory where output files will be saved.
        case_name : str
            Name of the run, used to name output files.
        hyperdiffusion_coefficient : float
            Hyperdiffusion coefficient in m^4/s (default: 1e17).
        rayleigh_damping_timescale : float or array-like; default: 25
            Rayleigh damping timescale in days. If one value is passed, it is applied uniformly in the vertical.
            Otherwise, must be an array of size Nsigma listing values from the topmost to the bottom-most levels.
        newtonian_cooling_timescale : float or array-like; default: 15
            Newtonian cooling timescale in days. If one value is passed, it is applied uniformly in the vertical.
            Otherwise, must be an array of size Nsigma listing values from the topmost to the bottom-most levels.            
        """
        self.resolution = resolution
        self.Nsigma = Nsigma
        self.deltasigma = 1/self.Nsigma
        
        self.zonal_basic_state = zonal_basic_state
        self.linear = linear
        self.hyperdiffusion_coefficient = hyperdiffusion_coefficient * meter**4 / second
        self.output_dir = output_dir
        self.case_name = case_name

        rayleigh_damping_timescale = np.array(rayleigh_damping_timescale)
        if len(rayleigh_damping_timescale.shape) == 0: # one float/int value was passed
            self.rayleigh_damping_coefficients = np.ones(self.Nsigma)/(rayleigh_damping_timescale*day)
        else:
            assert len(rayleigh_damping_timescale) == Nsigma,\
            f"rayleigh_damping_timescale must be one value or an array of size Nsigma, got array of length {len(rayleigh_damping_timescale)}"
            self.rayleigh_damping_coefficients = 1/(rayleigh_damping_timescale*day) 

        newtonian_cooling_timescale = np.array(newtonian_cooling_timescale)
        if len(newtonian_cooling_timescale.shape) == 0: # one float/int value was passed
            self.newtonian_cooling_coefficients = np.ones(self.Nsigma)/(newtonian_cooling_timescale*day)
        else:
            assert len(newtonian_cooling_timescale) == Nsigma,\
            f"newtonian_cooling_timescale must be one value or an array of size Nsigma, got array of length {len(newtonian_cooling_timescale)}"
            self.newtonian_cooling_coefficients = 1/(newtonian_cooling_timescale*day) 

        # Setup the horizontal grid and vertical grid
        self._setup_horizontal_grid()
        self._setup_vertical_grid()

        # Setup problem variables and equations
        self._setup_problem()
        
        # For debugging purposes, this is how to access the rank and size of the MPI communicator
        # world_rank = MPI.COMM_WORLD.Get_rank()
        # world_size = MPI.COMM_WORLD.Get_size()
        # print(f"Running on rank {world_rank} of {world_size}"
    
    def _setup_horizontal_grid(self):
        """
        Setup the spherical coordinates and bases for the simulation, as well as the vertical grid.
        """
        dtype = np.float64
        Nphi = self.resolution * 2  
        Ntheta = self.resolution 

        # Dealiasing factor indicates how many grid points to use in physical space
        if self.linear:
            dealias = (1, 1)
        else:
            dealias = (3/2, 3/2) # Delaiasing factor appropriate for order 2 nonlinearities 

        # Spherical coordinates / bases
        self.coords = d3.S2Coordinates('phi', 'theta')
        self.dist = d3.Distributor(self.coords, dtype=dtype)#
        self.full_basis = d3.SphereBasis(self.coords, (Nphi, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)
        self.zonal_basis = d3.SphereBasis(self.coords, (1, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)

    def _setup_vertical_grid(self):
        """
        Setup the vertical grid for the simulation. We use uniformly spaced sigma levels.
        """
        self.sigma = (np.arange(self.Nsigma) + 0.5) * self.deltasigma # Half levels
        self.sigma_full = np.arange(1,self.Nsigma) * self.deltasigma # Full levels, excluding surface and model top

    # The following three functions are used in the formulation of the problem equations
    def _dlnps_dt(self):
        """
        Outputs a str that expresses the log surface pressure tendency as a function of problem variables
        (perturbation log surface pressure lnps, basic-state log surface pressure lnpsbar,
        perturbation velocities u, basic-state velocities ubar).
        """
        sum_us    = "+".join([f"u{j}"  for j in range(1,self.Nsigma+1)])
        sum_ubars = "+".join([f"ubar{j}" for j in range(1,self.Nsigma+1)])
        return f"(- deltasigma * (div({sum_us}) + ({sum_us})@grad(lnpsbar) + ({sum_ubars})@grad(lnps)))"

    def _utilde(self):
        sum_us    = "+".join([f"u{j}"  for j in range(1,self.Nsigma+1)])
        return f"deltasigma * ({sum_us})"
    
    def _ubartilde(self):
        sum_ubars = "+".join([f"ubar{j}" for j in range(1,self.Nsigma+1)])
        return f"deltasigma * ({sum_ubars})"
    
    def _sigmadot(self,i):
        """
        Outputs a str that expresses the sigma-vertical-velocity at level i as a function of problem variables
        (perturbation log surface pressure lnps, basic-state log surface pressure lnpsbar,
        perturbation velocities u, basic-state velocities ubar).

        Parameters:
        -----------
        i : int
            The index of the half sigma level (1 to Nsigma).
        """
        partialsum_us = "+".join([f"u{j}"  for j in range(1,i+1)])
        partialsum_ubars = "+".join([f"ubar{j}" for j in range(1,i+1)])

        int_u_minus_utilde     = f"deltasigma * ({partialsum_us} - {i} * {self._utilde()})"
        int_u_minus_utilde_bar = f"deltasigma * ({partialsum_ubars} - {i} * {self._ubartilde()})"
        int_d_minus_dtilde     = f"div({int_u_minus_utilde})"

        return f"(- ({int_u_minus_utilde})@grad(lnpsbar) - ({int_u_minus_utilde_bar})@grad(lnps) - {int_d_minus_dtilde})"

    def _Phiprime(self,i):
        """
        Outputs a str that expresses the perturbation geopotential height at full level i as a function of 
        perturbartion temperarture T.
    
        Parameters:
        -----------
        i : int
            The index of the half sigma level (1 to Nsigma).
        """
        #"""Function to compute the geopotential height (perturbation relative to surface geopotential height) at full level i (0 is model top, N is surface)"""
        if i==self.Nsigma:
            return 0.
        else:
            partialsum = "+".join([f"T{j}/({j}-0.5)" for j in range(i+1,self.Nsigma+1)])
            return f"(Rd * ({partialsum}))"

    def _setup_problem(self):
        """
        Setup the vertical grid, instantiate all fields, and lay out the problem equations.
        """
        # hyperdiffusion, Rayleigh damping & Newtonian cooling
        nu = self.hyperdiffusion_coefficient
        epsilon_mom = self.rayleigh_damping_coefficients 
        epsilon_T = self.newtonian_cooling_coefficients 
        epsilon_zonalmean = 1 / (3 * day)

        # cross product by zhat times sin(latitude) for Coriolis force
        zcross = lambda A: d3.MulCosine(d3.skew(A))

        # Constants that appear in the problem equations
        sigma = self.sigma
        deltasigma = self.deltasigma
        kappa = consts['Rd'] / consts['cp']  
        Rd = consts['Rd'] 
        Omega = consts['Omega'] 

        # Fields
        u_names        = [f"u{i}" for i in range(1,self.Nsigma+1)]             # Perturbation velocity
        T_names        = [f"T{i}" for i in range(1,self.Nsigma+1)]             # Perturbation temperature

        EMFD_names     = [f"EMFD{i}" for i in range(1,self.Nsigma+1)]          # Eddy momentum flux divergence
        Qdiab_names    = [f"Qdiab{i}" for i in range(1,self.Nsigma+1)]         # Diabatic forcing
        EHFD_names     = [f"EHFD{i}" for i in range(1,self.Nsigma+1)]          # Eddy heat flux divergence
        
        ubar_names        = [f'ubar{i}' for i in range(1,self.Nsigma+1)]       # Basic-state velocity
        Tbar_names        = [f'Tbar{i}' for i in range(1,self.Nsigma+1)]       # Basic-state temperature
        sigmadotbar_names = [f"sigmadotbar{i}" for i in range(1,self.Nsigma)]  # Basic-state sigma-velocity
        
        # Instantiate prognostic fields
        us           = {name: self.dist.VectorField(self.coords, name=name, bases=self.full_basis) for name in u_names } # Perturbation velocity
        Ts           = {name: self.dist.Field(name=name, bases=self.full_basis) for name in T_names }                    # Perturbation temperature
        lnps         = self.dist.Field(name='lnps'  , bases=self.full_basis)                                             # Perturbation log-surface-pressure

        # Instantiate forcing fields
        EMFDs        = {name: self.dist.VectorField(self.coords, name=name, bases=self.full_basis) for name in EMFD_names } # Eddy momentum flux divergence
        Qdiabs       = {name: self.dist.Field(name=name, bases=self.full_basis) for name in Qdiab_names }                   # Diabatic forcing
        EHFDs        = {name: self.dist.Field(name=name, bases=self.full_basis) for name in EHFD_names }                    # Eddy heat flux divergence
        Phisfc       =  self.dist.Field(name='Phisfc', bases=self.full_basis)                                               # Surface geopotential

        # Instantiate basic-state fields
        if self.zonal_basic_state:
            basis_basestate = self.zonal_basis
        else:
            basis_basestate = self.full_basis  
        ubars        = {name: self.dist.VectorField(self.coords, name=name, bases=basis_basestate) for name in ubar_names } # Basic-state velocity
        Tbars        = {name: self.dist.Field(name=name, bases=basis_basestate) for name in Tbar_names }                    # Basic-state temperature
        sigmadotbars = {name: self.dist.Field(name=name, bases=basis_basestate) for name in sigmadotbar_names }             # Basic-state sigma-velocity
        lnpsbar      =  self.dist.Field(name='lnpsbar'  , bases=basis_basestate)                                            # Basic-state log-surface-pressure
        
        # Create field that is identically equal to one
        # This is used when relaxing the zonal mean, as Dedalus isn't happy if you 
        # put fields that live on the zonal basis in the main equations (all fields 
        # have to live on the full basis)
        one =    self.dist.Field(name='one'  , bases=self.full_basis)
        one['g'] = 1.0  

        # Gather all 3D variables 
        self.vars  = {**us, **Ts, **Qdiabs, **EMFDs, **EHFDs, **ubars, **Tbars, **sigmadotbars, 'lnps': lnps, 'lnpsbar': lnpsbar, 'Phisfc': Phisfc}

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
                + epsilon_mom[{i-1}] * u{i} \
                + nu * lap(lap(u{i})) \
                + grad( ({self._Phiprime(i-1)}+{self._Phiprime(i)}) / 2 ) \
                + 2 * Omega * zcross(u{i})"
            
            LHS_T = f"dt(T{i}) \
                + epsilon_T[{i-1}] * T{i} \
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
                - kappa * T{i} * (ubar{i} - {self._ubartilde()})@grad(lnpsbar)\
                - kappa * Tbar{i} * (u{i} - {self._utilde()})@grad(lnpsbar)\
                - kappa * Tbar{i} * (ubar{i} - {self._ubartilde()})@grad(lnps)\
                + kappa * Tbar{i} * div({self._utilde()})\
                + kappa * T{i} * div({self._ubartilde()}) )"
                # - kappa * Tbar{i} * {self._dlnps_dt()}\
                # - kappa * T{i} * ubar{i}@grad(lnpsbar)\
                # - kappa * Tbar{i} * u{i}@grad(lnpsbar)\
                # - kappa * Tbar{i} * ubar{i}@grad(lnps) )"
            
            forcing_mom = f"-grad(Phisfc) - EMFD{i}"

            forcing_T = f"Qdiab{i} - EHFD{i}"

            zonal_mean_relaxation_u = f"- epsilon_zonalmean * Average(u{i},'phi') * one"
            zonal_mean_relaxation_T = f"- epsilon_zonalmean * Average(T{i},'phi') * one"

            if self.zonal_basic_state: #All terms with non-constant coefficients from the basic state go on the LHS
                # Momentum equation
                self.problem.add_equation(f"{LHS_mom} + {minus_RHS_mom} = {zonal_mean_relaxation_u} + {forcing_mom}")
                # Thermodynamic equation
                self.problem.add_equation(f"{LHS_T} + {minus_RHS_T} = {zonal_mean_relaxation_T} + {forcing_T}")
            else:
                # Momentum equation
                self.problem.add_equation(f"{LHS_mom} = {zonal_mean_relaxation_u} + {forcing_mom} - {minus_RHS_mom}")
                # Thermodynamic equation
                self.problem.add_equation(f"{LHS_T} = {zonal_mean_relaxation_T} + {forcing_T} - {minus_RHS_T}")

    def initialize_basic_state_from_sigma_data(self,input_data):
        """
        Initialize the basic state fields from input data that are defined on the same sigma levels as the model.
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.

        Parameters:
        -----------
        input_data : xarray.Dataset
            Dataset containing the basic state variables (U, V, W, T, SP) on sigma levels.
            U is zonal velocity in m/s, V is meridional velocity in m/s, W is the pressure velocity in Pa/s,
            T is temperature in K, SP is surface pressure in Pa.
            SP must have dimensions (lat, lon) or (lat,) for a zonal basic state.
            U,V,T must have dimensions (sigma_half, lat, lon) or (sigma_half, lat) for a zonal basic state.
            W must have dimensions (sigma_full, lat, lon) or (sigma_full, lat) for a zonal basic state.
        """
        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat_deg = (np.pi / 2 - theta + 0*phi) * 180 / np.pi 
        lon_deg = (phi-np.pi) * 180 / np.pi

        assert np.allclose(input_data.sigma_half.data, self.sigma     ), "Input data sigma levels must match model sigma levels."
        assert np.allclose(input_data.sigma_full.data, self.sigma_full), "Input data sigma levels must match model sigma levels."
        
        if self.zonal_basic_state:
            assert set(input_data.dims) == {'sigma_half', 'sigma_full', 'lat'}, "Input data must have dimensions {sigma_half, sigma_full, lat} for zonal basic state."
            local_grid_xr = xr.DataArray(np.zeros(len(lat_deg[0])),coords={'lat':lat_deg[0]},dims=['lat'])
            target_shape = (1, len(lat_deg[0]))
        else:
            assert set(input_data.dims) == {'sigma_half', 'sigma_full', 'lat', 'lon'}, "Input data must have dimensions {sigma_half, sigma_full, lat, lon} for non-zonal basic state."
            local_grid_xr = xr.DataArray(np.zeros(lat_deg.shape),coords={'lon':lon_deg[:,0],'lat':lat_deg[0]},dims=['lon','lat'])
            target_shape = lat_deg.shape

        input_data_itp = input_data.interp_like(local_grid_xr, method='linear').transpose('sigma_half','sigma_full',*local_grid_xr.dims)

        # Basic-state temperature, wind
        for i in range(1,self.Nsigma+1):
            self.vars[f'ubar{i}']['g'] = np.stack([ input_data_itp.U.isel(sigma_half=i-1).data,
                                                   -input_data_itp.V.isel(sigma_half=i-1).data ], axis=0).reshape((2,*target_shape)) * meter / second
            self.vars[f'Tbar{i}']['g'] = input_data_itp.T.isel(sigma_half=i-1).data.reshape(target_shape) * Kelvin

        # Basic-state log-surface pressure
        self.vars['lnpsbar']['g'] = np.log(input_data_itp.SP.data/1e5).reshape(target_shape)

        # Basic-state sigmadot (omega/ps - sigma * u.grad(lnps))
        omega_ov_ps = input_data_itp.W/input_data_itp.SP * 1/second
        for i in range(1,self.Nsigma):
            self.vars[f'sigmadotbar{i}']['g'] = omega_ov_ps.isel(sigma_full=i-1).data.reshape(target_shape)
            self.vars[f'sigmadotbar{i}'] = self.vars[f'sigmadotbar{i}']\
                - self.sigma_full[i-1] * (self.vars[f'ubar{i}']+self.vars[f'ubar{i+1}']) @ d3.grad(self.vars['lnpsbar']) / 2
            
    def initialize_basic_state_from_pressure_data(self,input_data):
        """
        Initialize the basic state fields from input data that are defined on arbitrary pressure levels.
        Typically, this is reanalysis data. This function interpolates the data to the model's sigma levels
        before calling initialize_basic_state_from_sigma_data.

        Parameters:
        -----------
        input_data : xarray.Dataset
            Dataset containing the basic state variables (U, V, W, T, SP) on pressure levels. 
            U is zonal velocity in m/s, V is meridional velocity in m/s, W is the pressure velocity in Pa/s,
            T is temperature in K, SP is surface pressure in Pa.
            SP must have dimensions (lat, lon) or (lat,) for a zonal basic state.
            U,V,W,T must have dimensions (pressure, lat, lon) or (pressure, lat) for a zonal basic state.
            pressure is supposed to be in hPa to match most reanalysis datasets.
        """

        input_data = input_data.assign_coords(sigma=input_data.pressure/input_data.SP*100)
        input_data_halflevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma,sig,y),
                                             input_data.sigma,
                                             input_data[['U','V','T']],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_half',),),
                                             vectorize=True)
        input_data_fulllevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma_full,sig,y),
                                             input_data.sigma,
                                             input_data[['W',]],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_full',),),
                                             vectorize=True)

        input_data_vitp = xr.merge((input_data_halflevs.assign_coords(sigma_half=self.sigma),
                                    input_data_fulllevs.assign_coords(sigma_full=self.sigma_full),
                                    input_data.SP)
                                    )

        self.initialize_basic_state_from_sigma_data(input_data_vitp)

    def initialize_forcings_from_sigma_data(self,input_data):
        """
        Initialize the forcing fields from input data that are defined on the same sigma levels as the model.
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.

        Parameters:
        -----------
        input_data : xarray.Dataset
            Dataset containing the forcing fields (ZSFC, QDIAB, EHFD, EMFD_U, EMFD_V).
            ZSFC is the surface height in m and must have dimensions (lat, lon)
            QDIAB is the diabatic heating rate in K/s and must have dimensions (sigma_half, lat, lon).
            EHFD is the eddy heat flux divergence in K/s and must have dimensions (sigma_half, lat, lon).
            EMFD_U is the eddy divergence of zonal momentum flux in m/s^2 and must have dimensions (sigma_half, lat, lon).
            EMFD_V is the eddy divergence of meridional momentum flux in m/s^2 and must have dimensions (sigma_half, lat, lon).
        """
        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat_deg = (np.pi / 2 - theta + 0*phi) * 180 / np.pi 
        lon_deg = (phi-np.pi) * 180 / np.pi

        assert set(input_data.dims) == {'sigma_half', 'lat', 'lon'}, "Input forcings must have dimensions {sigma_half, lat, lon}."
        assert np.allclose(input_data.sigma_half.data, self.sigma), "Input forcings sigma levels must match model sigma levels."

        local_grid_xr = xr.DataArray(np.zeros(lat_deg.shape),coords={'lon':lon_deg[:,0],'lat':lat_deg[0]},dims=['lon','lat'])

        input_data_itp = input_data.interp_like(local_grid_xr, method='linear').transpose('sigma_half','lon','lat')
        
        # Topographic forcing
        self.vars['Phisfc']['g'] = input_data_itp.ZSFC.data * meter * consts['g'] 
        # Diabatic heating and eddy flux forcing
        for i in range(1,self.Nsigma+1):
            self.vars[f'Qdiab{i}']['g'] = input_data_itp.QDIAB.isel(sigma_half=i-1).data * Kelvin / second
            self.vars[f'EHFD{i}']['g'] = input_data_itp.EHFD.isel(sigma_half=i-1).data * Kelvin / second
            self.vars[f'EMFD{i}']['g'] = np.stack([ input_data_itp.EMFD_U.isel(sigma_half=i-1).data,
                                                   -input_data_itp.EMFD_V.isel(sigma_half=i-1).data ], axis=0) * meter / second ** 2

    def initialize_forcings_from_pressure_data(self,input_data):
        """
        Initialize the forcing fields from input data that are defined on arbitrary pressure levels.
        Typically, this is reanalysis data. This function interpolates the data to the model's sigma levels
        before calling initialize_forcings_from_sigma_data.

        Parameters:
        -----------
        input_data : xarray.Dataset
            Dataset containing the forcing fields (ZSFC, QDIAB, EHFD, EMFD_U, EMFD_V)
            on pressure levels, plus surface pressure SP.
            pressure is supposed to be in hPa to match most reanalysis datasets.
            SP is the surface pressure in Pa and must have dimensions (lat, lon)
            ZSFC is the surface height in m and must have dimensions (lat, lon)
            QDIAB is the diabatic heating rate in K/day and must have dimensions (pressure, lat, lon).
            EHFD is the eddy heat flux divergence in K/s and must have dimensions (pressure, lat, lon).
            EMFD_U is the eddy divergence of zonal momentum flux in m/s^2 and must have dimensions (pressure, lat, lon).
            EMFD_V is the eddy divergence of meridional momentum flux in m/s^2 and must have dimensions (pressure, lat, lon).
        """

        input_data = input_data.assign_coords(sigma=input_data.pressure/input_data.SP*100)
        input_data_halflevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma,sig,y),
                                             input_data.sigma,
                                             input_data[['QDIAB','EHFD','EMFD_U','EMFD_V']],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_half',),),
                                             vectorize=True)

        input_data_vitp = xr.merge((input_data_halflevs.assign_coords(sigma_half=self.sigma),
                                    input_data.ZSFC)
                                    )

        self.initialize_forcings_from_sigma_data(input_data_vitp)

    def _calc_omega(self,i):
        """
        Diagnose pressure velocity.
        
        Parameters:
        -----------
        i : int
            The index of the full sigma level (1 to Nsigma-1).
        """
        p0 = 1e5 * kilogram / meter / second ** 2  # Reference surface pressure in Pa

        # basic-state pressure velocity
        sigma_full_i = i*self.deltasigma
        sigmadot_i = eval(self._sigmadot(i).replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
        sigmadotbar_i = self.vars[f"sigmadotbar{i}"]
        Dlnpsbar_Dt = eval(f"(ubar{i}+ubar{i+1})@d3.grad(lnpsbar)/2",self.namespace)
        omegabar_i = p0 * np.exp(self.vars["lnpsbar"]) * ( sigma_full_i * Dlnpsbar_Dt + sigmadotbar_i)
        
        # total pressure velocity
        Dlnpstot_Dt = eval(f"{self._dlnps_dt()}\
                            + (ubar{i}+ubar{i+1})@grad(lnpsbar)/2\
                            + (u{i}+u{i+1})@grad(lnpsbar)/2\
                            + (ubar{i}+ubar{i+1})@grad(lnps)/2".replace('div','d3.div').replace('grad','d3.grad')
                          ,self.namespace)
        
        omegatot_i = p0 * np.exp(self.vars["lnpsbar"]+self.vars["lnps"]) * ( sigma_full_i * Dlnpstot_Dt + sigmadotbar_i + sigmadot_i)
        
        return omegatot_i-omegabar_i

    def _configure_snapshots(self):
        """ 
        Define output variables. This version outputs all state variables, forcings, and background state,
        plus vorticity and pressure velocity at each half level.
        You may want to omit some of these to reduce output size.
        """
        self.snapshots.add_task(self.vars['lnps'])
        self.snapshots.add_task(self.vars['lnpsbar'])
        self.snapshots.add_task(self.vars['Phisfc'] / (meter**2 / second**2), name = 'Phisfc') 

        for i in range(1,self.Nsigma+1):
            # u, T perturbations
            self.snapshots.add_task(self.vars[f'u{i}'] / (meter / second), name=f'u{i}')
            self.snapshots.add_task(self.vars[f'T{i}'] / Kelvin, name=f'T{i}')

            # Vorticity
            self.snapshots.add_task(-d3.div(d3.skew(self.vars[f'u{i}'])) / (1/second), name=f'zeta{i}') 

            # Forcings
            self.snapshots.add_task(self.vars[f'Qdiab{i}'] / Kelvin * second, name=f'Qdiab{i}')
            self.snapshots.add_task(self.vars[f'EHFD{i}'] / Kelvin * second, name=f'EHFD{i}')
            self.snapshots.add_task(self.vars[f'EMFD{i}'] / (meter / second**2), name=f'EMFD{i}')

            # Basic-state variables
            self.snapshots.add_task(self.vars[f'ubar{i}'] / (meter / second), name=f'ubar{i}')
            self.snapshots.add_task(self.vars[f'Tbar{i}'] / Kelvin, name=f'Tbar{i}')
            if i < self.Nsigma:  # sigmadotbar is only defined for levels 1 to N-1:
                self.snapshots.add_task(self.vars[f'sigmadotbar{i}'] / (1/second), name=f'sigmadotbar{i}')
            
        # Add sigma and pressure velocity to snapshots  
        for i in range(1,self.Nsigma):
            # What I'm doing here is to calculate sigmadot at level i. 
            # The sigmadot function does exactly that, but outputs a string that goes in the equation formulation
            # What we simply need to do is to transform this string into an actual expression that acts on the fields
            # This is what 'eval' does.
            # However with the current namespace, the operators 'div' and 'grad' are not imported
            # Thus I simpy replace them with d3.div and d3.grad
            sigmadot_i = eval(self._sigmadot(i).replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
            self.snapshots.add_task(sigmadot_i / (1/second), name=f'sigmadot{i}') 
            
            if not self.zonal_basic_state: #otherwise a bug in Dedalus prevents us from outputting omega for now
                self.snapshots.add_task(self._calc_omega(i) / (kilogram / meter / second**3), name=f'omega{i}')
            
    def integrate(self, restart=False, restart_id='s1', use_CFL=False, safety_CFL=0.7, timestep=400, stop_sim_time=5*86400):
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
            write, initial_timestep = self.solver.load_state(self.output_dir+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
            timestep = min(timestep, initial_timestep)
            file_handler_mode = 'append'

        snapshot_id = f'stationarywave_{self.Nsigma}level_T{self.resolution}_{self.case_name}'

        # Save snapshots every 6h of model time
        self.snapshots = self.solver.evaluator.add_file_handler(self.output_dir+snapshot_id, sim_dt=6 * 3600 * second, mode=file_handler_mode)  
        self._configure_snapshots()

        # CFL
        CFL = d3.CFL(self.solver, initial_dt=timestep, cadence=20, safety=safety_CFL, min_dt = timestep, max_change = 1.5)
        # Need to add all velocities and sigmadots - not sure if worth it
        for i in range(1,self.Nsigma+1):
            CFL.add_velocity(self.vars[f'u{i}'])
            if not self.zonal_basic_state:
                CFL.add_velocity(self.vars[f'ubar{i}'])
        for i in range(1,self.Nsigma):
            # sigmadot_i = eval(self._sigmadot(i).replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
            # CFL.add_frequency(sigmadot_i/self.deltasigma/2)
            if not self.zonal_basic_state:
                CFL.add_frequency(self.vars[f'sigmadotbar{i}']/self.deltasigma/2)

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
