import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/stationarywave_snapshots/"
import warnings; import sys

#########################################
######### Simulation Parameters #########
#########################################
# Resolution
#Nphi = 128; Ntheta = 64; resolution='T64'
Nphi = 64; Ntheta = 32; resolution='T32'

# Other
dealias = (3/2, 3/2)
dtype = np.float64

# Simulation units
meter = 1 / 6.37122e6
hour = 1.
second = hour / 3600
day = hour*24
Kelvin = 1.

#########################################
###### INTEGRATION TIME & RESTART #######
#########################################
restart=bool(int(sys.argv[1])); restart_id='s1'
use_CFL=restart; safety_CFL = 0.8

linear=False
timestep = 400*second
if not restart:
    stop_sim_time = 10*day 
else:
    stop_sim_time = 100*day

snapshot_id = 'stationarywave_SW_%s_rhs_20'%(resolution)

#########################################
########## PHYSICAL PARAMETERS ##########
#########################################

# Earth parameters
Omega = 2*np.pi/86400 / second
R0     = 6.37122e6*meter

g = 9.81 *  meter / second**2

# hyperdiffusion & Rayleigh damping 
nu = 40e15*meter**4/second 
epsilon = 1/day

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)#
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R0, dealias=dealias, dtype=dtype)
zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R0, dealias=dealias, dtype=dtype)

# cross product by zhat times sin(latitude)
zcross = lambda A: d3.MulCosine(d3.skew(A))

###############################
###### SAVE CURRENT FILE ######
###############################
if dist.comm.rank == 0:
    Path(SNAPSHOTS_DIR+snapshot_id).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), SNAPSHOTS_DIR+snapshot_id+'/'+os.path.basename(__file__))

#################################
####### INITIALIZE FIELDS #######
#################################
# Fields
h  =  dist.Field(name='h'  , bases=full_basis)
hbottom  =  dist.Field(name='hbottom'  , bases=full_basis)
u  =  dist.VectorField(coords,name='u'  , bases=full_basis)

H  =  dist.Field(name='H'  , bases=zonal_basis)
U  =  dist.VectorField(coords,name='U'  , bases=zonal_basis)

phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

phizonal, thetazonal = dist.local_grids(zonal_basis)
latzonal = np.pi / 2 - thetazonal + 0*phizonal
#latzonal=lat

lat0 = np.pi/3*0.9
deltalat = np.pi/20
lon0 = 0.
deltalon = np.pi/20
f0 = 2 * Omega * np.sin(lat0)

H0 = 1e3 * meter
U0 = 20 * (meter/second)
Uprof = np.exp(-(latzonal-lat0)**2/(2*deltalat**2)) * U0

U.change_scales(1)
H.change_scales(1)
hbottom.change_scales(1)
U['g'][0] = Uprof * np.sin(lat0) / np.sin(latzonal) * (latzonal>np.pi/10)
from scipy.special import erf
H['g'] = H0# - R0 * f0 / g * U0 * np.sqrt(np.pi/2) * deltalat * (1 + erf((latzonal - lat0)/(np.sqrt(2)*deltalat)))
hbottom['g'] = np.exp(-(lat-lat0)**2/(2*deltalat**2)) * np.exp(-(lon-lon0)**2/(2*deltalon**2)) * 1e2 * meter 
    
###############################
######## SETUP PROBLEM ########
###############################

problem = d3.IVP([u, h], namespace=locals())
#problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u)) + U@grad(u) + u@grad(U) = - g*grad(hbottom)")
#problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h)) + div(U*h) + div(u*H) = 0")

problem.add_equation("dt(u) + g*grad(h) + 2*Omega*zcross(u) + epsilon*u + nu*lap(lap(u))  = - g*grad(hbottom) -(U@grad(u) + u@grad(U))")
problem.add_equation("dt(h) + epsilon*h + nu*lap(lap(h))  = -(div(U*h) + div(u*H))")


# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1, max_dt = 0.2)
CFL.add_velocity(u)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################

if not restart:
    # Initialize fields that could be nonzero (e.g. with random seeds)
    #for i in range(N):
    #    T[i].fill_random('g', seed=i+1, distribution='normal', scale=1e-4)
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
    file_handler_mode = 'append'


##########################################
######## SETUP SNAPSHOTS & DO RUN ########
##########################################
ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1
sinlat = dist.Field(name='sinlat' , bases=full_basis)
sinlat['g'] = np.cos(theta)

snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode=file_handler_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(U, name='U')
snapshots.add_task(H, name='H')
snapshots.add_task(-d3.div(d3.skew(u)), name='zeta')
snapshots.add_task(2*Omega*sinlat, name='zeta_planetary')
snapshots.add_task(-d3.div(d3.skew(U)), name='Zeta')

# Main loop
with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            if use_CFL:
                timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.info('Last dt=%e' %(timestep))
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()