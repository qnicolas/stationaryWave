#######################################################################################
######### Shallow water linear stationary wave model on the sphere            #########
######### This integrates the shallow water equations for perturbation height #########
######### and velocity fields in time. A steady-state should be reached after #########
######### a few days of integration                                           #########
######### To run:                                                             #########
######### mpiexec -n {number of cores} python stationarywave_SW.py            #########
#######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

import os.path as op
ROOT = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir))
DATA_PATH = op.join(ROOT, 'data') + '/'
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

Ntheta = 32; Nphi = 2 * Ntheta 
snapshot_id = 'stationarywave_SW_T%i_ideal_Gill_linear'%Ntheta

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
stop_sim_time = 30*day 

# Earth parameters
Omega = 2*np.pi/86400 / second
R0     = 6.37122e6*meter

g = 9.81 *  meter / second**2

# hyperdiffusion & Rayleigh damping 
nu = 1e17*meter**4/second 
epsilon = 1 / (2 * day)

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

# Heating
Q  =  dist.Field(name='Q'  , bases=full_basis)

# Define an operator: cross product by zhat times sin(latitude) (for Coriolis force)
zcross = lambda A: d3.MulCosine(d3.skew(A))


# Get coordinate values (phi is longitude between 0 and 2 pi, theta is colatitude between 0 and pi)
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

lon0 = np.pi/2.
deltalon = 20 * np.pi / 180. 


# Calculate heating structure corresponding to first baroclinic mode of an idealized atmosphere (see ideal_gill.pdf)
ps = 1e3
gamma = 5e-5
Kn = np.pi**2 / (287. * gamma * ps ** 2)
cn = np.sqrt(1/Kn)
beta = 2 * (Omega * second) / (R0 / meter)
Leq = np.sqrt(cn / (2*beta))

lon_structure = np.cos(np.pi * (lon-lon0)/2/deltalon) * (np.abs((lon-lon0)/deltalon) < 1)
lon_structure = lon_structure - lon_structure.mean()
lat_structure = np.exp(-((R0 / meter) * lat)**2/(4*Leq**2))

# # Imposed heating
QSW = 1 / 86400 / (Kn * gamma * (ps/np.pi))
Q0 = QSW * meter ** 2 / second ** 3 / g
Q['g'] = Q0 * lon_structure * lat_structure

# Mean height field
H0 = cn**2 * meter ** 2 / second ** 2 / g 
H['g'] = H0
    
#########################################
############# SETUP PROBLEM #############
#########################################
problem = d3.IVP([u, h], namespace=locals())

if(lhs):
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u)) + U@grad(u) + u@grad(U) = - g*grad(hbottom)")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h)) + div(U*h) + div(u*H) = Q ")
else:
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u))  = - g*grad(hbottom) -(U@grad(u) + u@grad(U))")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h))  = -(div(U*h) + div(u*H)) + Q")


# Solver (integrates in time using a Runge Kutta method)
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time


# Where and when to write output
snapshots = solver.evaluator.add_file_handler(DATA_PATH + 'output/' + snapshot_id, sim_dt=6*hour, mode='overwrite')
# What to write : u,h (that's solver.state), vorticity (-d3.div(d3.skew(u))), divergence, basic state
snapshots.add_task(u / (meter/second), name='u')
snapshots.add_task(h / meter, name='h')
snapshots.add_task(-d3.div(d3.skew(u)) * second, name='zeta')
snapshots.add_task(d3.div(u) * second, name='div')
snapshots.add_task(U/ (meter / second), name='U')
snapshots.add_task(H/ (meter), name='H')
snapshots.add_task(hbottom/ (meter), name='hbottom')
Q.change_scales(1)
snapshots.add_task(Q / (meter / second), name='Q')


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