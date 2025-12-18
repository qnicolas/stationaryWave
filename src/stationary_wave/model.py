##########################################################################################
# Sigma-level stationary wave model on the sphere                                        #
# Inspired from Ting & Yu (1998, JAS).                                                   #
# Model has a variable number of sigma levels (N)                                        #
# and uses spherical harmonics in the horizontal.                                        #
# Prognostic variables are horizontal velocity at each level (u,v)_i,                    #
# temperature T_i, and perturbation-log-surface-pressure lnps.                           #
#                                                                                        #
# A steady-state should be reached after a few days of integration, typically 20-50      #
#                                                                                        #
# Example run script:                                                                    #
#                                                                                        #
## sph_resolution = 16; Nsigma = 5; linear = True; zonal_basic_state = True              #
## output_dir = "data/"; case_name = "test"                                              #
## exampleRun = StationaryWaveProblem(sph_resolution, np.linspace(0,1,Nsigma+1),         #
##                                    linear, zonal_basic_state,                         #
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
import time
import datetime

import xarray as xr
from pathlib import Path
from .tools import prime, lon_360_to_180, lon_180_to_360

# Simulation units 
# The main point is to do the calculations on a sphere of unit radius, 
# which has better numerical properties than using a radius of 6.37122e6 
# Time is also expressed in hours
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
    def __init__(self, resolution, sigma_full, linear, zonal_basic_state, output_dir, case_name, 
                 remove_zonal_mean = True,
                 hyperdiffusion_coefficient=1e17, 
                 rayleigh_damping_timescale=25, 
                 newtonian_cooling_timescale=15
                ):
        """
        Initialize the stationary wave problem with given parameters.

        Parameters
        ----------
        resolution : int
            Maximum degree & order in the spherical harmonic expansion. Akin to the order of a triangular truncation.
        sigma_full : array-like
            Vertical grid, given as an array of the full sigma levels (sorted increasingly, from 0 to 1).
        linear : bool 
            If True, exclude nonlinear terms in the equations.
        zonal_basic_state : bool
            Whether the basic state is zonally-invariant (i.e., only depends on latitude).
        output_dir : str
            Directory where output files will be saved.
        case_name : str
            Name of the run, used to name output files.
        remove_zonal_mean : bool, optional
            Whether to remove the zonal mean from the forcing fields and damp zonal-mean of perturbation fields. 
            Default: True.
        hyperdiffusion_coefficient : float, optional
            Hyperdiffusion coefficient in m^4/s. Default: 1e17.
        rayleigh_damping_timescale : float or array-like, optional
            Rayleigh damping timescale in days. If one value is passed, it is applied uniformly in the vertical.
            Otherwise, must be an array of size Nsigma listing values from the topmost to the bottom-most levels.
            Default: 25.
        newtonian_cooling_timescale : float or array-like, optional
            Newtonian cooling timescale in days. If one value is passed, it is applied uniformly in the vertical.
            Otherwise, must be an array of size Nsigma listing values from the topmost to the bottom-most levels.   
            Default: 15.         
        """
        self.resolution = resolution

        assert np.all(np.diff(sigma_full) >= 0), "sigma_full must be sorted increasingly."
        assert (len(sigma_full) >= 3) and sigma_full[0]==0. and sigma_full[-1]==1., "sigma_full must have at least three values, including 0 and 1"
        self.sigma_full = np.array(sigma_full)
        self.Nsigma = len(sigma_full)-1
        
        self.zonal_basic_state = zonal_basic_state
        self.linear = linear
        self.hyperdiffusion_coefficient = hyperdiffusion_coefficient * meter**4 / second
        self.output_dir = output_dir
        self.case_name = case_name
        self.remove_zonal_mean = remove_zonal_mean

        rayleigh_damping_timescale = np.array(rayleigh_damping_timescale)
        if len(rayleigh_damping_timescale.shape) == 0: # one float/int value was passed
            self.rayleigh_damping_coefficients = np.ones(self.Nsigma)/(rayleigh_damping_timescale*day)
        else:
            assert len(rayleigh_damping_timescale) == self.Nsigma,\
            f"rayleigh_damping_timescale must be one value or an array of size Nsigma ({self.Nsigma}), got array of length {len(rayleigh_damping_timescale)}"
            self.rayleigh_damping_coefficients = 1/(rayleigh_damping_timescale*day) 

        newtonian_cooling_timescale = np.array(newtonian_cooling_timescale)
        if len(newtonian_cooling_timescale.shape) == 0: # one float/int value was passed
            self.newtonian_cooling_coefficients = np.ones(self.Nsigma)/(newtonian_cooling_timescale*day)
        else:
            assert len(newtonian_cooling_timescale) == self.Nsigma,\
            f"newtonian_cooling_timescale must be one value or an array of size Nsigma ({self.Nsigma}), got array of length {len(newtonian_cooling_timescale)}"
            self.newtonian_cooling_coefficients = 1/(newtonian_cooling_timescale*day) 

        # Relaxation coefficient for zonal-mean fields
        self.zonalmean_relaxation_coefficient = 1 / (3 * day)

        # Setup the horizontal grid and vertical grid
        self._setup_horizontal_grid()
        self._setup_vertical_grid()

        # Setup problem variables and equations
        self._setup_problem()
        
        # For debugging purposes, this is how to access the rank and size of the MPI communicator
        # (First, need to "from mpi4py import MPI" at the top of the file)
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
            dealias = (2, 2) # Dealiasing factor appropriate for order 3 nonlinearities 

        # Spherical coordinates / bases
        self.coords = d3.S2Coordinates('phi', 'theta')
        self.dist = d3.Distributor(self.coords, dtype=dtype)#
        self.full_basis = d3.SphereBasis(self.coords, (Nphi, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)
        self.zonal_basis = d3.SphereBasis(self.coords, (1, Ntheta), radius=consts['R0'], dealias=dealias, dtype=dtype)

    def _setup_vertical_grid(self):
        """
        Calculate half sigma levels and layer depths from the full sigma levels
        """
        self.sigma_half = (self.sigma_full[1:] + self.sigma_full[:-1])/2 # Half levels
        self.deltasigma_full = self.sigma_full[1:] - self.sigma_full[:-1]
        self.deltasigma_half = self.sigma_half[1:] - self.sigma_half[:-1]

    # The following functions are used in the formulation of the problem equations
    def _utilde(self):
        """
        Expresses the vertical average of the perturbation horizontal velocity.
        """
        vint_us    = "+".join([f"self.deltasigma_full[{j-1}] * u{j}"  for j in range(1,self.Nsigma+1)])
        return f"({vint_us})"
    
    def _ubartilde(self):
        """
        Expresses the vertical average of the basic-state horizontal velocity.
        """
        vint_ubars = "+".join([f"self.deltasigma_full[{j-1}] * ubar{j}" for j in range(1,self.Nsigma+1)])
        return f"({vint_ubars})"
    
    def _dlnps_dt(self,component):
        """
        Expresses the log surface pressure tendency as a function of problem variables
        (perturbation log surface pressure lnps, basic-state log surface pressure lnpsbar,
        perturbation velocities u, basic-state velocities ubar).

        Parameters
        ----------
        component : str
            Either 'linear', 'nonlinear', or 'full', indicating which part of the tendency to return.
            If 'full' and self.linear is True, only the linear part is returned.
        """
        linear_tendency = f"(- div({self._utilde()}) - {self._utilde()}@grad(lnpsbar) - {self._ubartilde()}@grad(lnps))"
        nonlinear_tendency = f"(- {self._utilde()}@grad(lnps))"
        if component == 'linear':
            return linear_tendency
        elif component == 'nonlinear':
            return nonlinear_tendency
        elif component == 'full':
            if self.linear:
                return linear_tendency
            else:
                return f"( {linear_tendency} + {nonlinear_tendency} )"

    def _sigmadot(self,i,component):
        """
        Expresses the sigma-vertical-velocity at level i as a function of problem variables
        (perturbation log surface pressure lnps, basic-state log surface pressure lnpsbar,
        perturbation velocities u, basic-state velocities ubar).

        Parameters
        ----------
        i : int
            The index of the half sigma level (1 to Nsigma).
        component : str
            Either 'linear', 'nonlinear', or 'full', indicating which part of sigmadot to return.
            If 'full' and self.linear is True, only the linear part is returned.
        """
        partial_vint_us = "+".join([f"self.deltasigma_full[{j-1}] * u{j}"  for j in range(1,i+1)])
        partial_vint_ubars = "+".join([f"self.deltasigma_full[{j-1}] * ubar{j}" for j in range(1,i+1)])

        int_u_minus_utilde     = f"({partial_vint_us} - self.sigma_full[{i}] * {self._utilde()})"
        int_u_minus_utilde_bar = f"({partial_vint_ubars} - self.sigma_full[{i}] * {self._ubartilde()})"
        int_d_minus_dtilde     = f"div({int_u_minus_utilde})"

        linear_sigmadot = f"( - ({int_u_minus_utilde})@grad(lnpsbar) - ({int_u_minus_utilde_bar})@grad(lnps) - {int_d_minus_dtilde} )"
        nonlinear_sigmadot = f"( - ({int_u_minus_utilde})@grad(lnps) )"

        if component == 'linear':
            return linear_sigmadot
        elif component == 'nonlinear':
            return nonlinear_sigmadot
        elif component == 'full':
            if self.linear:
                return linear_sigmadot
            else:
                return f"( {linear_sigmadot} + {nonlinear_sigmadot} )"
    

    def _vert_advection_full_level(self,i,var,component):
        """
        Expresses the vertical advection of a variable at full level i as a function of 
        the variable at half levels i and i+1, and the sigma-vertical-velocity at full level i

        Parameters
        ----------
        i : int
            The index of the full sigma level (0 to Nsigma).
        var : str
            The name of the variable to advect ('u' or 'T').
        component : str
            Either 'linear' or 'nonlinear', indicating which part of the vertical advection to return.
        """

        if component == 'linear':
            return f"( sigmadotbar{i} * ({var}{i+1}-{var}{i}) + {self._sigmadot(i,'linear')} * ({var}bar{i+1}-{var}bar{i}) ) \
                                                   / self.deltasigma_half[{i-1}]"
        elif component == 'nonlinear':
            return f"( {self._sigmadot(i,'full')} * ({var}{i+1}-{var}{i}) + {self._sigmadot(i,'nonlinear')} * ({var}bar{i+1}-{var}bar{i}) ) \
                                                   / self.deltasigma_half[{i-1}]"

    def _Phiprime(self,i):
        """
        Expresses the perturbation geopotential height at full level i as a function of 
        perturbation temperature T.
    
        Parameters
        ----------
        i : int
            The index of the half sigma level (1 to Nsigma).
        """
        #"""Function to compute the geopotential height (perturbation relative to surface geopotential height) at full level i (0 is model top, N is surface)"""
        if i==self.Nsigma:
            return 0.
        else:
            partialsum = "+".join([f"self.deltasigma_full[{j-1}] * T{j}/self.sigma_half[{j-1}]" for j in range(i+1,self.Nsigma+1)])
            return f"(Rd * ({partialsum}))"

    def _setup_problem(self):
        """
        Setup the vertical grid, instantiate all fields, and lay out the problem equations.
        """
        # cross product by zhat times sin(latitude) for Coriolis force
        zcross = lambda A: d3.MulCosine(d3.skew(A))

        # Constants that appear in the problem equations
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
        # This is used whenever adding terms that live on the zonal basis 
        # with terms that live on the full basis, due to a bug in Dedalus
        one =    self.dist.Field(name='one'  , bases=self.full_basis)
        one['g'] = 1.0  

        # Gather all 3D variables 
        self.vars  = {**us, **Ts, **Qdiabs, **EMFDs, **EHFDs, **ubars, **Tbars, **sigmadotbars, 'lnps': lnps, 'lnpsbar': lnpsbar, 'Phisfc': Phisfc, 'one': one}

        ###################################################
        ################ PROBLEM EQUATIONS ################
        ###################################################

        self.namespace = (globals() | locals() | self.vars)
        self.problem = d3.IVP(list(us.values()) + list(Ts.values()) + [lnps,] , namespace=self.namespace)

        # log surface pressure equation
        if self.zonal_basic_state: #All linear terms with non-constant coefficients go on the LHS
            nonlin = "0" if self.linear else self._dlnps_dt('nonlinear')
            self.problem.add_equation(f"dt(lnps) - {self._dlnps_dt('linear')} = {nonlin}")
        else:
            self.problem.add_equation(f"dt(lnps) = {self._dlnps_dt('full')}")

        for i in range(1,self.Nsigma+1):
            # Build terms that involve sigma-dot (expansion and vertical advection), because they require different treatment at upper and lower boundaries

            # Expansion terms, contributions from full levels i and i-1
            expansion_linear_lower    = f"kappa/self.sigma_half[{i-1}] * ( {self._sigmadot(i,'linear')} * Tbar{i} + sigmadotbar{i} * T{i} )"
            expansion_nonlinear_lower = f"kappa/self.sigma_half[{i-1}] * ( {self._sigmadot(i,'full')} * T{i} + {self._sigmadot(i,'nonlinear')}  * Tbar{i} )"
            expansion_linear_upper    = f"kappa/self.sigma_half[{i-1}] * ( {self._sigmadot(i-1,'linear')} * Tbar{i} + sigmadotbar{i-1} * T{i} )"
            expansion_nonlinear_upper = f"kappa/self.sigma_half[{i-1}] * ( {self._sigmadot(i-1,'full')} * T{i} + {self._sigmadot(i-1,'nonlinear')}  * Tbar{i} )"

            # Combine contributions from above and below, taking care of first and last levels
            if i==1:
                vert_advection_mom_linear    = f" {self._vert_advection_full_level(i,'u','linear')} / 2"
                vert_advection_mom_nonlinear = f" {self._vert_advection_full_level(i,'u','nonlinear')} / 2"
                vert_advection_T_linear      = f" {self._vert_advection_full_level(i,'T','linear')} / 2"
                vert_advection_T_nonlinear   = f" {self._vert_advection_full_level(i,'T','nonlinear')} / 2"
                expansion_linear = f" {expansion_linear_lower} / 2"
                expansion_nonlinear = f" {expansion_nonlinear_lower} / 2"
            elif i==self.Nsigma:
                vert_advection_mom_linear    = f" {self._vert_advection_full_level(i-1,'u','linear')} / 2"
                vert_advection_mom_nonlinear = f" {self._vert_advection_full_level(i-1,'u','nonlinear')} / 2"
                vert_advection_T_linear      = f" {self._vert_advection_full_level(i-1,'T','linear')} / 2"
                vert_advection_T_nonlinear   = f" {self._vert_advection_full_level(i-1,'T','nonlinear')} / 2"
                expansion_linear = f" {expansion_linear_upper} / 2"
                expansion_nonlinear = f" {expansion_nonlinear_upper} / 2"
            else:
                vert_advection_mom_linear    = f"( {self._vert_advection_full_level(i,'u','linear')} + {self._vert_advection_full_level(i-1,'u','linear')} ) / 2"
                vert_advection_mom_nonlinear = f"( {self._vert_advection_full_level(i,'u','nonlinear')} + {self._vert_advection_full_level(i-1,'u','nonlinear')} ) / 2"
                vert_advection_T_linear      = f"( {self._vert_advection_full_level(i,'T','linear')} + {self._vert_advection_full_level(i-1,'T','linear')} ) / 2"
                vert_advection_T_nonlinear   = f"( {self._vert_advection_full_level(i,'T','nonlinear')} + {self._vert_advection_full_level(i-1,'T','nonlinear')} ) / 2"
                expansion_linear = f"( {expansion_linear_lower} + {expansion_linear_upper} ) / 2"
                expansion_nonlinear = f"( {expansion_nonlinear_lower} + {expansion_nonlinear_upper} ) / 2"

            # Calculate d(Tbar)/d(ln(sigma)), which is needed in the temperature equation, using first-order finite differences
            if i==1:
                dtbardlnsigma = f"((Tbar{i+1}-Tbar{i}) / self.deltasigma_half[{i-1}] * self.sigma_half[{i-1}])" # Forward difference
            else:
                dtbardlnsigma = f"((Tbar{i}-Tbar{i-1}) / self.deltasigma_half[{i-2}] * self.sigma_half[{i-1}])" # Backward difference

            # Assemble momentum and thermodynamic equations
            LHS_mom = f"dt(u{i}) \
                + self.rayleigh_damping_coefficients[{i-1}] * u{i} \
                + self.hyperdiffusion_coefficient * lap(lap(u{i})) \
                + grad( ({self._Phiprime(i-1)}+{self._Phiprime(i)}) / 2 ) \
                + 2 * Omega * zcross(u{i})"
            
            LHS_T = f"dt(T{i}) \
                + self.newtonian_cooling_coefficients[{i-1}] * T{i} \
                + self.hyperdiffusion_coefficient * lap(lap(T{i}))"

            linear_terms_mom = f"( ubar{i}@grad(u{i})\
                + u{i}@grad(ubar{i})\
                + {vert_advection_mom_linear}\
                + Rd * Tbar{i} * grad(lnps)\
                + Rd * T{i} * grad(lnpsbar))"

            nonlinear_terms_mom = "" if self.linear\
                else f" - ( u{i}@grad(u{i}) + {vert_advection_mom_nonlinear} + Rd * T{i} * grad(lnps) )"
            
            linear_terms_T = f"( ubar{i}@grad(T{i}) \
                + u{i}@grad(Tbar{i})\
                + {vert_advection_T_linear}\
                - {expansion_linear}\
                - kappa * T{i} * (ubar{i} - {self._ubartilde()})@grad(lnpsbar)\
                - kappa * Tbar{i} * (u{i} - {self._utilde()})@grad(lnpsbar)\
                - kappa * Tbar{i} * (ubar{i} - {self._ubartilde()})@grad(lnps)\
                + kappa * Tbar{i} * div({self._utilde()})\
                + kappa * T{i} * div({self._ubartilde()})\
                - self.newtonian_cooling_coefficients[{i-1}] * lnps * {dtbardlnsigma}\
                - self.hyperdiffusion_coefficient * {dtbardlnsigma} * lap(lap(lnps))  )"
            
            nonlinear_terms_T =  "" if self.linear\
                else f" - ( u{i}@grad(T{i})\
                + {vert_advection_T_nonlinear}\
                - {expansion_nonlinear}\
                - kappa * T{i} * (u{i} - {self._utilde()})@grad(lnpsbar)\
                - kappa * T{i} * (ubar{i} - {self._ubartilde()})@grad(lnps)\
                - kappa * Tbar{i} * (u{i} - {self._utilde()})@grad(lnps)\
                - kappa * T{i} * (u{i} - {self._utilde()})@grad(lnps)\
                + kappa * T{i} * div({self._utilde()}) )"
            
            forcing_mom = f"- grad(Phisfc) - EMFD{i}"

            forcing_T = f"Qdiab{i} - EHFD{i}"

            zonal_mean_relaxation_u = f"- self.zonalmean_relaxation_coefficient * Average(u{i},'phi') * one" if self.remove_zonal_mean else ""
            zonal_mean_relaxation_T = f"- self.zonalmean_relaxation_coefficient * Average(T{i},'phi') * one"  if self.remove_zonal_mean else ""

            if self.zonal_basic_state: #All terms with non-constant coefficients from the basic state go the LHS
                # Momentum equation
                self.problem.add_equation(f"{LHS_mom} + {linear_terms_mom} = {forcing_mom} {zonal_mean_relaxation_u} {nonlinear_terms_mom}")
                # Thermodynamic equation
                self.problem.add_equation(f"{LHS_T} + {linear_terms_T} = {forcing_T} {zonal_mean_relaxation_T} {nonlinear_terms_T}")
            else:
                # Momentum equation
                self.problem.add_equation(f"{LHS_mom} = {forcing_mom} {zonal_mean_relaxation_u} {nonlinear_terms_mom} - {linear_terms_mom}")
                # Thermodynamic equation
                self.problem.add_equation(f"{LHS_T} = {forcing_T} {zonal_mean_relaxation_T} {nonlinear_terms_T} - {linear_terms_T}")

    def _preprocess_input_pressure_data(self,input_data):
        """Make sure pressure levels are increasing and reach high and low enough"""
        if input_data.pressure[0] > input_data.pressure[1]:
            input_data = input_data.reindex({'pressure':list(reversed(input_data['pressure']))})

        input_data = input_data.assign_coords(sigma=input_data.pressure/input_data.SP*100)
        # Warn if top/bottom extrapolation will occur. 
        if self.dist.comm.rank == 0:
            if input_data.sigma.isel(pressure=0).max() > self.sigma_half[0]:
                logger.warning("For some grid points in your input data, the topmost sigma level is {:.4f}, which lies lower than the model's topmost sigma level of {:.4f}. " \
                            "The data will be extrapolated using the nearest value".format(input_data.sigma.isel(pressure=0).max().data, self.sigma_half[0]))
            if input_data.sigma.isel(pressure=-1).min() < self.sigma_half[-1]:
                logger.warning("For some grid points in your input data, the bottommost sigma level is {:.4f}, which lies higher than the model's bottommost sigma level of {:.4f}. "
                            "The data will be extrapolated using the nearest value".format(input_data.sigma.isel(pressure=-1).min().data, self.sigma_half[-1]))
        return input_data
        
    def _preprocess_input_sigma_data(self,input_data):
        """Make sure data have the correct latitude range and add longitude points at +-180 degrees if needed"""
        input_data = input_data.sortby(input_data['lat'])

        # Check latitude range
        _, theta = self.dist.local_grids(self.full_basis)
        lat_max = (np.pi / 2 - theta[0,-1]) * 180 / np.pi
        assert (input_data.lat.max()>=lat_max) and input_data.lat.min()<=lat_max,\
              "Input data latitude range is too narrow. With this resolution, it must range at least from {:.2f} to {:.2f} degrees.".format(-lat_max, lat_max)
        
        
        if 'lon' in input_data.dims:
            input_data = input_data.sortby(input_data['lon'])
            
            # Change latitude from [0,360] to [-180,180] if needed
            if input_data.lon.max() <= 360 and input_data.lon.min() >= 0:
                input_data = lon_360_to_180(input_data)
            elif input_data.lon.max() <= 180 and input_data.lon.min() >= -180:
                pass
            else:
                raise ValueError("Input data longitude values must be in the range [-180,180] or [0,360].")
            
            # Calculate values at lon = 180 to fill in missing edge point if needed
            empty_da_180 = xr.DataArray([0.],coords={'lon':[180.]})
            input_data_180 = lon_180_to_360(input_data).interp_like(empty_da_180)

            # Fill in missing edge points at lon = -180 and lon = 180 if needed
            if input_data.lon.max() < 180:
                input_data = xr.concat([input_data, input_data_180], dim='lon')
            if input_data.lon.min() > -180:
                input_data_180 = input_data_180.assign_coords({'lon':[-180.]})
                input_data = xr.concat([input_data_180,input_data], dim='lon')

        return input_data

    def initialize_basic_state_from_sigma_data(self,input_data):
        """
        Initialize the basic state fields from input data that are defined on the same sigma levels as the model.
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.
        However, the function does not extrapolate outside the given latitude range: if the given latitude range
        is too narrow, an error will be raised.
        For non-zonal basic states, the longitude dimension ('lon') must be included and lie within [-180,180] or [0,360] degrees.


        Parameters
        ----------
        input_data : xarray.Dataset
            Dataset containing the basic state variables (U, V, W, T, SP) on sigma levels.
            U is zonal velocity in m/s, V is meridional velocity in m/s, W is the pressure velocity in Pa/s,
            T is temperature in K, SP is surface pressure in Pa.
            SP must have dimensions (lat, lon) or (lat,) for a zonal basic state.
            U,V,T must have dimensions (sigma_half, lat, lon) or (sigma_half, lat) for a zonal basic state.
            W must have dimensions (sigma_full, lat, lon) or (sigma_full, lat) for a zonal basic state.
        """
        input_data = self._preprocess_input_sigma_data(input_data)
        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat_deg = (np.pi / 2 - theta + 0*phi) * 180 / np.pi 
        lon_deg = (phi-np.pi) * 180 / np.pi

        assert np.allclose(input_data.sigma_half.data, self.sigma_half), "Input data half sigma levels must match model half sigma levels."
        assert np.allclose(input_data.sigma_full.data, self.sigma_full[1:-1]), "Input data full sigma levels must match model full sigma levels (excluding sigma=0 and sigma=1)."
        
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
            alpha_itp = (self.sigma_half[i] - self.sigma_full[i]) / (self.sigma_half[i] - self.sigma_half[i-1])
            self.vars[f'sigmadotbar{i}'] = self.vars[f'sigmadotbar{i}']\
                - self.sigma_full[i] * (alpha_itp * self.vars[f'ubar{i}'] + (1-alpha_itp) * self.vars[f'ubar{i+1}']) @ d3.grad(self.vars['lnpsbar'])
            self.vars[f'sigmadotbar{i}'] = self.vars[f'sigmadotbar{i}'].evaluate() # not sure if this is necessary
            
    def initialize_basic_state_from_pressure_data(self,input_data):
        """
        Initialize the basic state fields from input data that are defined on arbitrary pressure levels.
        Typically, this is reanalysis data. This function interpolates the data to the model's sigma levels
        before calling initialize_basic_state_from_sigma_data.
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.
        However, the function does not extrapolate outside the given latitude range: if the given latitude range
        is too narrow, an error will be raised.
        For non-zonal basic states, the longitude dimension ('lon') must be included and lie within [-180,180] or [0,360] degrees.
        pressure must be in hPa (to match most reanalysis datasets).

        Parameters
        ----------
        input_data : xarray.Dataset
            Dataset containing the basic state variables (U, V, W, T, SP) on pressure levels. 
            U is zonal velocity in m/s, V is meridional velocity in m/s, W is the pressure velocity in Pa/s,
            T is temperature in K, SP is surface pressure in Pa.
            SP must have dimensions (lat, lon) or (lat,) for a zonal basic state.
            U,V,W,T must have dimensions (pressure, lat, lon) or (pressure, lat) for a zonal basic state.
        """
        input_data = self._preprocess_input_pressure_data(input_data)
        input_data_halflevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma_half,sig,y),
                                             input_data.sigma,
                                             input_data[['U','V','T']],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_half',),),
                                             vectorize=True)
        input_data_fulllevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma_full[1:-1],sig,y),
                                             input_data.sigma,
                                             input_data[['W',]],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_full',),),
                                             vectorize=True)

        input_data_vitp = xr.merge((input_data_halflevs.assign_coords(sigma_half=self.sigma_half),
                                    input_data_fulllevs.assign_coords(sigma_full=self.sigma_full[1:-1]),
                                    input_data.SP)
                                    )

        self.initialize_basic_state_from_sigma_data(input_data_vitp)

    def initialize_forcings_from_sigma_data(self,input_data):
        """
        Initialize the forcing fields from input data that are defined on the same sigma levels as the model.
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.
        However, the function does not extrapolate outside the given latitude range: if the given latitude range
        is too narrow, an error will be raised.
        The longitude dimension ('lon') must lie within [-180,180] or [0,360] degrees.

        Parameters
        ----------
        input_data : xarray.Dataset
            Dataset containing the forcing fields (ZSFC, QDIAB, EHFD, EMFD_U, EMFD_V).
            ZSFC is the surface height in m and must have dimensions (lat, lon)
            QDIAB is the diabatic heating rate in K/s and must have dimensions (sigma_half, lat, lon).
            EHFD is the eddy heat flux divergence in K/s and must have dimensions (sigma_half, lat, lon).
            EMFD_U is the eddy divergence of zonal momentum flux in m/s^2 and must have dimensions (sigma_half, lat, lon).
            EMFD_V is the eddy divergence of meridional momentum flux in m/s^2 and must have dimensions (sigma_half, lat, lon).
        """
        input_data = self._preprocess_input_sigma_data(input_data)

        # Get coordinate values
        phi, theta = self.dist.local_grids(self.full_basis)
        lat_deg = (np.pi / 2 - theta + 0*phi) * 180 / np.pi 
        lon_deg = (phi-np.pi) * 180 / np.pi

        assert np.allclose(input_data.sigma_half.data, self.sigma_half), "Input forcings sigma levels must match model sigma levels."

        local_grid_xr = xr.DataArray(np.zeros(lat_deg.shape),coords={'lon':lon_deg[:,0],'lat':lat_deg[0]},dims=['lon','lat'])

        input_data_itp = input_data.interp_like(local_grid_xr, method='linear').transpose('sigma_half','lon','lat')

        # Remove zonal mean from forcings
        if self.remove_zonal_mean:
            input_data_itp = prime(input_data_itp)
        
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
        The lat-lon grid of the input data can be arbitrary and will be interpolated to the model's grid.
        However, the function does not extrapolate outside the given latitude range: if the given latitude range
        is too narrow, an error will be raised.
        The longitude dimension ('lon') must lie within [-180,180] or [0,360] degrees.
        pressure must be in hPa (to match most reanalysis datasets).

        Parameters
        ----------
        input_data : xarray.Dataset
            Dataset containing the forcing fields (ZSFC, QDIAB, EHFD, EMFD_U, EMFD_V)
            on pressure levels, plus surface pressure SP.
            SP is the surface pressure in Pa and must have dimensions (lat, lon)
            ZSFC is the surface height in m and must have dimensions (lat, lon)
            QDIAB is the diabatic heating rate in K/day and must have dimensions (pressure, lat, lon).
            EHFD is the eddy heat flux divergence in K/s and must have dimensions (pressure, lat, lon).
            EMFD_U is the eddy divergence of zonal momentum flux in m/s^2 and must have dimensions (pressure, lat, lon).
            EMFD_V is the eddy divergence of meridional momentum flux in m/s^2 and must have dimensions (pressure, lat, lon).
        """
        input_data = self._preprocess_input_pressure_data(input_data)
        input_data_halflevs = xr.apply_ufunc(lambda sig,y : np.interp(self.sigma_half,sig,y),
                                             input_data.sigma,
                                             input_data[['QDIAB','EHFD','EMFD_U','EMFD_V']],
                                             input_core_dims=(('pressure',),('pressure',)),
                                             output_core_dims=(('sigma_half',),),
                                             vectorize=True)

        input_data_vitp = xr.merge((input_data_halflevs.assign_coords(sigma_half=self.sigma_half),
                                    input_data.ZSFC)
                                    )

        self.initialize_forcings_from_sigma_data(input_data_vitp)

    def _calc_omega(self,i):
        """
        Diagnose perturbation pressure velocity.
        
        Parameters
        ----------
        i : int
            The index of the full sigma level (1 to Nsigma-1).
        """
        p0 = 1e5 * kilogram / meter / second ** 2  # Reference surface pressure in Pa

        # What I'm doing here is to calculate sigmadot at level i. 
        # The sigmadot function does exactly that, but outputs a string that goes in the equation formulation
        # What we simply need to do is to transform this string into an actual expression that acts on the fields
        # This is what 'eval' does.
        # However with the current namespace, the operators 'div' and 'grad' are not imported
        # Thus I simpy replace them with d3.div and d3.grad
        sigmadot_i = eval(self._sigmadot(i,'full').replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
        sigmadotbar_i = self.vars[f"sigmadotbar{i}"]

        # Coefficient used to interpolate from half levels to full levels
        alpha_itp = (self.sigma_half[i] - self.sigma_full[i]) / (self.sigma_half[i] - self.sigma_half[i-1])

        # Calculate basic-state omega
        # All the multiplications by self.vars['one'] are used to bring variables from the zonal basis to the full basis 
        Dlnpsbar_Dt = eval(f"( alpha_itp * ubar{i} + (1-alpha_itp) * ubar{i+1} ) @ d3.grad(lnpsbar)",(self.namespace|{'alpha_itp':alpha_itp}))
        omegabar_i = p0 * np.exp(self.vars["lnpsbar"]) * ( self.sigma_full[i] * Dlnpsbar_Dt + sigmadotbar_i) * self.vars['one']

        # Calculate total omega (basic state + perturbation)
        nonlin = "" if self.linear else f"+ ( alpha_itp * u{i} + (1-alpha_itp) * u{i+1} ) @ grad(lnps)"
        Dlnpstot_Dt = eval(f"{self._dlnps_dt('full')} \
                            +(( alpha_itp * ubar{i} + (1-alpha_itp) * ubar{i+1} ) @ grad(lnpsbar)) * self.vars['one']\
                            + ( alpha_itp * u{i} + (1-alpha_itp) * u{i+1} ) @ grad(lnpsbar)\
                            + ( alpha_itp * ubar{i} + (1-alpha_itp) * ubar{i+1} ) @ grad(lnps)\
                            {nonlin}".replace('div','d3.div').replace('grad','d3.grad')
                          ,(self.namespace|{'alpha_itp':alpha_itp}))
        omegatot_i = p0 * np.exp(self.vars["lnpsbar"] * self.vars['one'] + self.vars["lnps"]) * ( self.sigma_full[i] * Dlnpstot_Dt + sigmadotbar_i * self.vars['one'] + sigmadot_i)
        
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

            # Vorticity, divergence
            self.snapshots.add_task(-d3.div(d3.skew(self.vars[f'u{i}'])) / (1/second), name=f'zeta{i}') 
            self.snapshots.add_task(d3.div(self.vars[f'u{i}']) / (1/second), name=f'div{i}') 

            # Forcings
            self.snapshots.add_task(self.vars[f'Qdiab{i}'] / Kelvin * second, name=f'Qdiab{i}')
            self.snapshots.add_task(self.vars[f'EHFD{i}'] / Kelvin * second, name=f'EHFD{i}')
            self.snapshots.add_task(self.vars[f'EMFD{i}'] / (meter / second**2), name=f'EMFD{i}')

            # Basic-state variables
            self.snapshots.add_task(self.vars[f'ubar{i}'] / (meter / second), name=f'ubar{i}')
            self.snapshots.add_task(self.vars[f'Tbar{i}'] / Kelvin, name=f'Tbar{i}')
            if i < self.Nsigma:  # sigmadotbar is only defined for levels 1 to N-1:
                self.snapshots.add_task(self.vars[f'sigmadotbar{i}'] / (1/second), name=f'sigmadotbar{i}')
            
        # Add geopotential & sigma and pressure velocity to snapshots  
        for i in range(1,self.Nsigma):
            # Geopotential height
            Rd = consts['Rd'] 
            self.snapshots.add_task(eval(self._Phiprime(i),self.namespace) / (meter**2 / second**2), name=f'Phiprime{i}')

            # Sigma velocity
            sigmadot_i = eval(self._sigmadot(i,'full').replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
            self.snapshots.add_task(sigmadot_i / (1/second), name=f'sigmadot{i}') 

            # Pressure velocity
            self.snapshots.add_task(self._calc_omega(i) / (kilogram / meter / second**3), name=f'omega{i}')
            
    def integrate(self, restart=False, restart_id='s1', use_CFL=False, safety_CFL=0.7, timestep=400, stop_sim_time=5*86400):
        """Integrate the problem. 
        Parameters
        ---------------
        restart : bool, optional
            Whether to restart from a previous snapshot. Default is False.
        restart_id : str, optional
            Identifier for the restart snapshot (e.g. 's1', 's2', etc.). Default is 's1'.
        use_CFL : bool, optional
            Whether to use an adaptive timestep based on the CFL condition. Default is False.
        safety_CFL : float, optional
            Fraction of max timestep allowed by the CFL condition to use. Default is 0.7.
        timestep : float, optional
            Initial timestep in seconds. Default is 400s.
        stop_sim_time : float, optional
            Simulation time in seconds until which to integrate. Default is 5 days.
        """
        # Solver
        self.solver = self.problem.build_solver(d3.RK222)
        self.solver.stop_sim_time = stop_sim_time * second
        timestep = timestep * second

        snapshot_id = f'stationarywave_{self.Nsigma}level_T{self.resolution}_{self.case_name}'

        if not restart:
            file_handler_mode = 'overwrite'
        else:
            write, initial_timestep = self.solver.load_state(self.output_dir+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
            for i in range(1,self.Nsigma+1):
                # nondimensionalize u, T perturbations
                self.vars[f'u{i}']['g'] = self.vars[f'u{i}']['g'] * (meter / second)
                self.vars[f'T{i}']['g'] = self.vars[f'T{i}']['g'] * Kelvin
            timestep = min(timestep, initial_timestep)
            file_handler_mode = 'append'
            speed_avg = 0.
            timestep_avg = timestep


        # Save snapshots every 6h of model time
        self.snapshots = self.solver.evaluator.add_file_handler(self.output_dir+snapshot_id, sim_dt=6 * 3600 * second, mode=file_handler_mode)  
        self._configure_snapshots()
        # Save vertical grid
        Path(self.output_dir+snapshot_id).mkdir(parents=True, exist_ok=True)
        np.savetxt(self.output_dir+snapshot_id+'/sigma_full.txt', self.sigma_full)

        # CFL
        CFL = d3.CFL(self.solver, initial_dt=timestep, cadence=20, safety=safety_CFL, min_dt = timestep, max_change = 1.5)
        # Need to add all velocities and sigmadots - not sure if worth it
        for i in range(1,self.Nsigma+1):
            CFL.add_velocity(self.vars[f'ubar{i}'] + self.vars[f'u{i}'])
            # if not self.zonal_basic_state:
            #     CFL.add_velocity(self.vars['one']*self.vars[f'ubar{i}'])
        for i in range(1,self.Nsigma):
            sigmadot_i = eval(self._sigmadot(i,'full').replace('div','d3.div').replace('grad','d3.grad'),self.namespace)
            CFL.add_frequency((self.vars[f'sigmadotbar{i}'] + sigmadot_i)/self.deltasigma_half[i-1])
            # if not self.zonal_basic_state:
            #     CFL.add_frequency(self.vars['one']*self.vars[f'sigmadotbar{i}']/self.deltasigma_half[i-1])

        flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        flow.add_property(self.vars[f'u{1}']@self.vars[f'u{1}'], name='u2')

        # Main loop
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings('error',category=RuntimeWarning)
            try:
                logger.info('Starting main loop')
                while self.solver.proceed:
                    if use_CFL:
                        timestep = CFL.compute_timestep()
                    self.solver.step(timestep)
                    # Enforce zero zonal mean of lnps
                    if self.remove_zonal_mean:
                        m, ell, *_ = self.dist.coeff_layout.local_group_arrays(self.full_basis.domain(self.dist), scales=1)
                        self.vars['lnps']['c'][m == 0] = 0.

                    # Print some statistics
                    if self.solver.iteration % 20 == 0:
                        max_u = np.sqrt(flow.max('u2')) / (meter/second) # Get max wind speed in m/s
                        speed = 20 / (time.time()-t0) # speed in iterations per second
                        speed_avg = speed if self.solver.iteration < 60 else 0.9 * speed_avg + 0.1 * speed # moving average of speed
                        timestep_avg = timestep if self.solver.iteration <= 20 else 0.9 * timestep_avg + 0.1 * timestep # moving average of timestep
                        ETA = (self.solver.stop_sim_time - self.solver.sim_time) / timestep_avg / speed_avg
                        logger.info( f"Iteration={self.solver.iteration:d}, "\
                                    +f"Time={self.solver.sim_time / hour:.1f} h, "\
                                    +f"dt={timestep/second:.0f} s, "\
                                    +f"speed={speed:.2e} it/s, "\
                                    +f"ETA={'unknown' if self.solver.iteration < 60 else str(datetime.timedelta(seconds=int(ETA)))}, "\
                                    +f"max|u|={max_u:.2e} m/s")
                        t0 = time.time()
            except:
                logger.info('Last dt=%e' %(timestep))
                logger.error('Exception raised, triggering end of main loop.')
                raise
            finally:
                self.solver.log_stats()
