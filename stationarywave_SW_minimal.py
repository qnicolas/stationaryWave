import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/stationarywave_snapshots/"
import warnings

#########################################
######### Simulation Parameters #########
#########################################
#Switch from NCC terms to be on the LHS (lhs=True) to the RHS (lhs=False)
lhs=True 
if lhs:
    snapshot_id = 'stationarywave_SW_lhs'
else:
    snapshot_id = 'stationarywave_SW_rhs'

# Resolution
Nphi = 128; Ntheta = 64

# Other
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

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)#
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R0, dealias=dealias, dtype=dtype)
zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R0, dealias=dealias, dtype=dtype)

# cross product by zhat times sin(latitude)
zcross = lambda A: d3.MulCosine(d3.skew(A))

#################################
####### INITIALIZE FIELDS #######
#################################
# Main fields (height & velocity)
h  =  dist.Field(name='h'  , bases=full_basis)
hbottom  =  dist.Field(name='hbottom'  , bases=full_basis)
u  =  dist.VectorField(coords,name='u'  , bases=full_basis)

# Background fields (height & velocity)
H  =  dist.Field(name='H'  , bases=zonal_basis)
U  =  dist.VectorField(coords,name='U'  , bases=zonal_basis)

phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

lat0 = np.pi/4
deltalat = np.pi/20
lon0 = 0.
deltalon = np.pi/20

H0 = 1e3 * meter
U0 = 20 * (meter/second)
U['g'][0] = np.exp(-(lat[0]-lat0)**2/(2*deltalat**2)) * U0 #lat[0] because it lives on the zonal basis
H['g'] = H0
hbottom['g'] = np.exp(-(lat-lat0)**2/(2*deltalat**2)) * np.exp(-(lon-lon0)**2/(2*deltalon**2)) * 2e2 * meter 
  
###############################
######## SETUP PROBLEM ########
###############################

problem = d3.IVP([u, h], namespace=locals())

if(lhs):
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u)) + U@grad(u) + u@grad(U) = - g*grad(hbottom)")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h)) + div(U*h) + div(u*H) = 0")
else:
    problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u))  = - g*grad(hbottom) -(U@grad(u) + u@grad(U))")
    problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h))  = -(div(U*h) + div(u*H))")


# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode='overwrite')
snapshots.add_tasks(solver.state)
snapshots.add_task(-d3.div(d3.skew(u)), name='zeta')

# Main loop
with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop')
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