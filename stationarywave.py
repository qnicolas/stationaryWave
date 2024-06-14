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
Nphi = 128; Ntheta = 64; resolution='T64'
#Nphi = 64; Ntheta = 32; resolution='T32'

# Number of sigma levels
N = 5

# Other
dealias = (2, 2)
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

zonal_basic_state=True
timestep = 200*second
if not restart:
    stop_sim_time = 5*day 
else:
    stop_sim_time = 100*day

snapshot_id = 'stationarywave_%ilevel_%s_topo'%(N,resolution)

#########################################
########## PHYSICAL PARAMETERS ##########
#########################################

# Earth parameters
Omega = 2*np.pi/86400 / second
R0     = 6.37122e6*meter

g = 9.81 *  meter / second**2
cp = 1004. * meter**2 / second**2 / Kelvin
Rd = 287. * meter**2 / second**2 / Kelvin
kappa = Rd/cp

# hyperdiffusion & Rayleigh damping 
nu = 40e15*meter**4/second 
epsilon = np.ones(N)/(15.*day)
epsilon[N-1] = 1/(0.2*day)
epsilon[N-2] = 1/(1.0*day)

# diagnostic parameters
deltasigma = 1/N
sigma = (np.arange(N) + 0.5)*deltasigma

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

###############################
######## SETUP PROBLEM ########
###############################

# Fields
u_names        = ["u%i"%i for i in range(1,N+1)]
T_names        = ["T%i"%i for i in range(1,N+1)]
Qdiab_names    = ["Qdiab%i"%i for i in range(1,N+1)]

ubar_names        = ["ubar%i"%i for i in range(1,N+1)]
Tbar_names        = ["Tbar%i"%i for i in range(1,N+1)]
sigmadotbar_names = ["sigmadotbar%i"%i for i in range(1,N)]

us           = [dist.VectorField(coords, name=name, bases=full_basis) for name in u_names       ]
Ts           = [dist.Field(name=name, bases=full_basis) for name in T_names    ]
Qdiabs       = [dist.Field(name=name, bases=full_basis) for name in Qdiab_names]
lnps         =  dist.Field(name='lnps'  , bases=full_basis)
Phisfc       =  dist.Field(name='Phisfc', bases=full_basis)

if zonal_basic_state:
    basis_basestate = zonal_basis
else:
    basis_basestate = full_basis
ubars        = [dist.VectorField(coords, name=name, bases=basis_basestate) for name in ubar_names       ]
Tbars        = [dist.Field(name=name, bases=basis_basestate) for name in Tbar_names        ]
sigmadotbars = [dist.Field(name=name, bases=basis_basestate) for name in sigmadotbar_names]
lnpsbar      =  dist.Field(name='lnpsbar'  , bases=basis_basestate)

allnames = u_names + T_names + Qdiab_names + ubar_names + Tbar_names + sigmadotbar_names
allvars  = us + Ts + Qdiabs + ubars + Tbars + sigmadotbars  
    
    
###################################################
################ INITIALIZE FIELDS ################
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi
if zonal_basic_state:
    phizonal, thetazonal = dist.local_grids(zonal_basis)
    latzonal = np.pi / 2 - thetazonal + 0*phizonal
else:
    latzonal=lat

lat0 = np.pi/3*0.9
deltalat = np.pi/20
lon0 = 0.
deltalon = np.pi/20
T0 = 290.*Kelvin
Gamma = 7. * Kelvin / (1e3 * meter)
f0 = 2 * Omega * np.sin(lat0)

U0 = 10. * (meter/second)
Uprof = np.exp(-(latzonal-lat0)**2/(2*deltalat**2)) * U0

H0 = 500. * meter
Q0 = 0. * Kelvin / day

# Basic-state wind, temperature, heating
for i in range(N):
    Tbars[i]['g'] = T0 * sigma[i] ** (Rd*Gamma/g) # hydrostatic profile for constant lapse rate
    ubars[i]['g'][0] = Uprof * Tbars[i]['g'] / T0 * np.sin(lat0) / np.sin(latzonal) * (latzonal>np.pi/10)
    Qdiabs[i]['g'] = Q0 * np.exp(-(sigma[i]-0.5)**2/(2*0.1**2)) * np.exp(-(lat)**2/(2*deltalat**2)) * np.exp(-(lon)**2/(2*deltalon**2))
    
for i in range(N-1):
    sigmadotbars[i]['g'] = 0.
from scipy.special import erf
lnpsbar['g'] = -R0 * f0/(Rd*T0) * U0 * np.sqrt(np.pi/2) * deltalat * (1 + erf((latzonal - lat0)/(np.sqrt(2)*deltalat)))
Phisfc['g'] = np.exp(-(lat-lat0)**2/(2*deltalat**2)) * np.exp(-(lon-lon0)**2/(2*deltalon**2)) * H0 * g
    
######################################################################################################   

def dlnps_dt():
    """Function to compute the log surface pressure tendency"""
    sum_us    = "+".join(["u{}"   .format(j) for j in range(1,N+1)])
    sum_ubars = "+".join(["ubar{}".format(j) for j in range(1,N+1)])
    return "(- deltasigma * (div({sum_us}) + ({sum_us})@grad(lnpsbar) + ({sum_ubars})@grad(lnps)))".format(sum_us=sum_us,sum_ubars=sum_ubars)

def sigmadot(i):
    """Function to compute the sigma-vertical-velocity at level i"""
    partialsum_us = "+".join(["u{}".format(j) for j in range(1,i+1)])
    partialsum_ubars = "+".join(["ubar{}".format(j) for j in range(1,i+1)])
    return "(-{i}*deltasigma*{dlnpsdt} - deltasigma * (div({partialsum_us}) + ({partialsum_us})@grad(lnpsbar) + ({partialsum_ubars})@grad(lnps)))".format(i=i, dlnpsdt=dlnps_dt(),partialsum_us=partialsum_us,partialsum_ubars=partialsum_ubars)

def Phiprime(i):
    """Function to compute the geopotential height (perturbation relative to surface geopotential height) at full level i (0 is model top, N is surface)"""
    if i==N:
        return 0.
    else:
        partialsum = "+".join(["T{j}/({j}-0.5)".format(j=j) for j in range(i+1,N+1)])
        return "(Rd * ({partialsum}))".format(partialsum=partialsum)

problem = d3.IVP(us + Ts + [lnps,] , namespace=(locals() | {name:var for name,var in zip(allnames,allvars)}))

# log pressure equation
if zonal_basic_state:
    problem.add_equation("dt(lnps) - {} = 0".format(dlnps_dt()))
else:
    problem.add_equation("dt(lnps) = {}".format(dlnps_dt()))

for i in range(1,N+1):
    # Build terms that involve vertical differentiation/staggering - different treatment for upper and lower boundaries
    if i==1:
        vert_advection_mom_1 = "( sigmadotbar{i}*(u{ip1}-u{i})    )/deltasigma/2".format(i=i, ip1=i+1)
        vert_advection_mom_2 = "( {sigmadoti}*(ubar{ip1}-ubar{i}) )/deltasigma/2".format(i=i, ip1=i+1, sigmadoti=sigmadot(i))
        vert_advection_T_1   = "( sigmadotbar{i}*(T{ip1}-T{i})    )/deltasigma/2".format(i=i, ip1=i+1)
        vert_advection_T_2   = "( {sigmadoti}*(Tbar{ip1}-Tbar{i}) )/deltasigma/2".format(i=i, ip1=i+1, sigmadoti=sigmadot(i))
        expansion = "kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({sigmadoti})/2 + T{i}*(sigmadotbar{i})/2)".format(i=i, ip1=i+1, sigmadoti=sigmadot(i))
    elif i==N:
        vert_advection_mom_1 = "( sigmadotbar{im1}*(u{i}-u{im1})    )/deltasigma/2".format(i=i, im1=i-1)
        vert_advection_mom_2 = "( {sigmadotim1}*(ubar{i}-ubar{im1}) )/deltasigma/2".format(i=i, im1=i-1, sigmadotim1=sigmadot(i-1))
        vert_advection_T_1   = "( sigmadotbar{im1}*(T{i}-T{im1})    )/deltasigma/2".format(i=i, im1=i-1)
        vert_advection_T_2   = "( {sigmadotim1}*(Tbar{i}-Tbar{im1}) )/deltasigma/2".format(i=i, im1=i-1, sigmadotim1=sigmadot(i-1))
        expansion = "kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({sigmadotim1})/2 + T{i}*(sigmadotbar{im1})/2)".format(i=i, im1=i-1, sigmadotim1=sigmadot(i-1))
    else:
        vert_advection_mom_1 = "( sigmadotbar{i}*(u{ip1}-u{i})    + sigmadotbar{im1}*(u{i}-u{im1})    )/deltasigma/2".format(i=i, im1=i-1, ip1=i+1)
        vert_advection_mom_2 = "( {sigmadoti}*(ubar{ip1}-ubar{i}) + {sigmadotim1}*(ubar{i}-ubar{im1}) )/deltasigma/2".format(i=i, im1=i-1, ip1=i+1, sigmadoti=sigmadot(i), sigmadotim1=sigmadot(i-1))
        vert_advection_T_1   = "( sigmadotbar{i}*(T{ip1}-T{i})    + sigmadotbar{im1}*(T{i}-T{im1})    )/deltasigma/2".format(i=i, im1=i-1, ip1=i+1)
        vert_advection_T_2   = "( {sigmadoti}*(Tbar{ip1}-Tbar{i}) + {sigmadotim1}*(Tbar{i}-Tbar{im1}) )/deltasigma/2".format(i=i, im1=i-1, ip1=i+1, sigmadoti=sigmadot(i), sigmadotim1=sigmadot(i-1))
        expansion = "kappa/(deltasigma*({i}-0.5))*(Tbar{i}*({sigmadoti}+{sigmadotim1})/2 + T{i}*(sigmadotbar{i}+sigmadotbar{im1})/2)".format(i=i, im1=i-1, ip1=i+1, sigmadoti=sigmadot(i), sigmadotim1=sigmadot(i-1))
        
    if zonal_basic_state:
        # Momentum equation
        problem.add_equation("dt(u{i}) + grad(({Phiprimeim1}+{Phiprimei})/2) + 2*Omega*zcross(u{i}) + epsilon[{im1}]*u{i} + nu*lap(lap(u{i})) + ( ubar{i}@grad(u{i}) + u{i}@grad(ubar{i}) + {vert_advection_1} + {vert_advection_2} + Rd*Tbar{i}*grad(lnps) + Rd*T{i}*grad(lnpsbar)) = -grad(Phisfc)".format(i=i, im1=i-1, ip1=i+1, vert_advection_1=vert_advection_mom_1, vert_advection_2=vert_advection_mom_2, Phiprimeim1=Phiprime(i-1), Phiprimei=Phiprime(i)))
        # Thermodynamic equation
        problem.add_equation("dt(T{i}) + epsilon[{im1}]*T{i} + nu*lap(lap(T{i})) + (ubar{i}@grad(T{i}) + u{i}@grad(Tbar{i}) + {vert_advection_1} + {vert_advection_2} - {expansion} - kappa*Tbar{i}*({dlnpsdt}+ubar{i}@grad(lnps)+u{i}@grad(lnpsbar)) - kappa*T{i}*ubar{i}@grad(lnpsbar)) = Qdiab{i}".format(i=i, im1=i-1, ip1=i+1, vert_advection_1=vert_advection_T_1, vert_advection_2=vert_advection_T_2, expansion=expansion, dlnpsdt=dlnps_dt()))
        
    else:
        # Momentum equation
        problem.add_equation("dt(u{i}) + grad(({Phiprimeim1}+{Phiprimei})/2) + 2*Omega*zcross(u{i}) + epsilon[{im1}]*u{i} + nu*lap(lap(u{i})) = -grad(Phisfc) - ( ubar{i}@grad(u{i}) + u{i}@grad(ubar{i}) + {vert_advection_1} + {vert_advection_2} + Rd*Tbar{i}*grad(lnps) + Rd*T{i}*grad(lnpsbar))".format(i=i, im1=i-1, ip1=i+1, vert_advection_1=vert_advection_mom_1, vert_advection_2=vert_advection_mom_2, Phiprimeim1=Phiprime(i-1), Phiprimei=Phiprime(i)))
        # Thermodynamic equation
        problem.add_equation("dt(T{i}) + epsilon[{im1}]*T{i} + nu*lap(lap(T{i})) = Qdiab{i} - (ubar{i}@grad(T{i}) + u{i}@grad(Tbar{i}) + {vert_advection_1} + {vert_advection_2} - {expansion} - kappa*Tbar{i}*({dlnpsdt}+ubar{i}@grad(lnps)+u{i}@grad(lnpsbar)) - kappa*T{i}*ubar{i}@grad(lnpsbar))".format(i=i, im1=i-1, ip1=i+1, vert_advection_1=vert_advection_T_1, vert_advection_2=vert_advection_T_2, expansion=expansion, dlnpsdt=dlnps_dt()))
    
# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1, max_dt = 0.2)
CFL.add_velocity(us[0])

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
#ephi = dist.VectorField(coords, bases=full_basis)
#ephi['g'][0] = 1
#etheta = dist.VectorField(coords, bases=full_basis)
#etheta['g'][1] = 1

def dlnps_dt_calc():
    """Function to compute the log surface pressure tendency"""
    sum_us    = "+".join(["u{}"   .format(j) for j in range(1,N+1)])
    sum_ubars = "+".join(["ubar{}".format(j) for j in range(1,N+1)])
    return - deltasigma * (d3.div(sum(us)) + (sum(us))@d3.grad(lnpsbar) + (sum(ubars))@d3.grad(lnps))

def sigmadot_calc(i):
    """Function to compute the sigma-vertical-velocity at level i"""
    partialsum_us = sum(us[:i])
    partialsum_ubars = sum(ubars[:i])
    return (-i*deltasigma*dlnps_dt_calc() - deltasigma * (d3.div(partialsum_us) + (partialsum_us)@d3.grad(lnpsbar) + (partialsum_ubars)@d3.grad(lnps)))


snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode=file_handler_mode)
snapshots.add_tasks(solver.state)
for i in range(1,N+1):
    snapshots.add_task(-d3.div(d3.skew(us[i-1])), name='zeta%i'%i)
for i in range(N):
    snapshots.add_task(ubars[i], name=ubar_names[i])
    snapshots.add_task(Tbars[i], name=Tbar_names[i])
for i in range(N-1):
    snapshots.add_task(sigmadot_calc(i+1), name='sigmadot%i'%(i+1))
    snapshots.add_task(sigmadotbars[i], name=sigmadotbar_names[i])
snapshots.add_task(lnpsbar,name='lnpsbar')

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