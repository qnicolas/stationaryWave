#######################################################################################
######### Shallow water linear stationary wave model on the sphere            #########
######### This integrates the shallow water equations for perturbation height #########
######### and velocity fields in time. A steady-state should be reached after #########
######### a few days of integration                                           #########
######### To run: change SNAPSHOTS_DIR, then                                  #########
######### mpiexec -n {number of cores} python stationarywave_SW.py            #########
#######################################################################################

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/stationarywave_snapshots/"
import warnings

#########################################
######### Simulation Parameters #########
#########################################
# Switch non-constant coefficent terms (U.grad u + u.grad U) from being on the LHS (lhs=True) to the RHS (lhs=False)
# This changes the way the calculation is made. With LHS, the terms go into a matrix that is calculated only once. 
# With RHS, the products are calculated at every time step. 
# Hence, LHS takes longer to build matrices but faster to integrate, while the converse happens for RHS
# For zonally-varying basic states, however, only RHS is available
lhs=True
if lhs:
    snapshot_id = 'stationarywave_SW_lhs'
else:
    snapshot_id = 'stationarywave_SW_rhs'

# Resolution (in spectral space)
Nphi = 128; Ntheta = 64

# Other: dealiasing factor indicates how many grid points to use 
# in physical space (here, 1.5 times the spectral resolution is suited for second-order nonlinearities).
dealias = (3/2, 3/2)
dtype = np.float64

# Simulation units
meter = 1 / 6.37122e6
hour = 1.
second = hour / 3600
day = hour*24

timestep = 400*second
stop_sim_time = 10*day 

# Earth parameters
Omega = 2*np.pi/86400 / second
R0     = 6.37122e6*meter

g = 9.81 *  meter / second**2

# hyperdiffusion & Rayleigh damping 
nu = 40e15*meter**4/second 
epsilon = 1/day

#########################################
############### DOMAIN ##################
#########################################
# see https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_1.html

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)#
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R0, dealias=dealias, dtype=dtype)
zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R0, dealias=dealias, dtype=dtype)

#########################################
########### INITIALIZE FIELDS ###########
#########################################
# Background fields (height & velocity)
H  =  dist.Field(name='H'  , bases=zonal_basis)
U  =  dist.VectorField(coords,name='U'  , bases=zonal_basis)

# Perturbation fields (height & velocity)
h  =  dist.Field(name='h'  , bases=full_basis)
u  =  dist.VectorField(coords,name='u'  , bases=full_basis)

# Bottom topography
hbottom  =  dist.Field(name='hbottom'  , bases=full_basis)

# Define an operator: cross product by zhat times sin(latitude) (for Coriolis force)
zcross = lambda A: d3.MulCosine(d3.skew(A))


# Get coordinate values (phi is longitude between 0 and 2 pi, theta is colatitude between 0 and pi)
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

# define the basic-state wind and height, as well as the topography
# Note that these should be balanced: fk x U +  U.grad(U) = -g*grad(H) and div(H*U)=0
lat0 = np.pi/4
deltalat = np.pi/20
lon0 = 0.
deltalon = np.pi/20

U0 = 20 * (meter/second)
U['g'][0] = np.exp(-(lat[0]-lat0)**2/(2*deltalat**2)) * U0 #lat[0] because it lives on the zonal basis
hbottom['g'] = np.exp(-(lat-lat0)**2/(2*deltalat**2)) * np.exp(-(lon-lon0)**2/(2*deltalon**2)) * 5e2 * meter 

# Balanced height field: solve a quick linear boundary value problem
c = dist.Field(name='c')# a constant
H0 = 1e3 * meter # mean height
problem = d3.LBVP([c, H], namespace=locals())
problem.add_equation("g*lap(H) + c = - div(U@grad(U) + 2*Omega*zcross(U))")
problem.add_equation("ave(H) = H0")
solver = problem.build_solver()
solver.solve()

#########################################
############# SETUP PROBLEM #############
#########################################
problem = d3.IVP([u, h], namespace=locals())

if(lhs):
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u)) + U@grad(u) + u@grad(U) = - g*grad(hbottom)")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h)) + div(U*h) + div(u*H) = -div(U*hbottom) ") #- div(u*hbottom) term ignored but could be included, would have to be on RHS
else:
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u))  = - g*grad(hbottom) -(U@grad(u) + u@grad(U))")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h))  = -(div(U*h) + div(u*H) + div(U*hbottom))")#- div(u*hbottom) term ignored but could be included, would have to be on RHS


# Solver (integrates in time using a Runge Kutta method)
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# Where and when to write output
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode='overwrite')
# What to write : u,h (that's solver.state) and vorticity (-d3.div(d3.skew(u)))
snapshots.add_tasks(solver.state)
snapshots.add_task(-d3.div(d3.skew(u)), name='zeta')
snapshots.add_task(U, name='U')
snapshots.add_task(H, name='H')
snapshots.add_task(hbottom, name='hbottom')

########################################
############ MAIN TIME LOOP ############
########################################

with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop. Final time: %.5e'%stop_sim_time)
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.info('Last dt=%e' %(timestep))
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()