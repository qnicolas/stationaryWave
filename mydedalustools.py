import numpy as np
import xarray as xr 
import dedalus.public as d3
import time
import h5py
import matplotlib.pyplot as plt
from stationarywave import StationaryWaveProblem


def thetaphi_to_latlon(ds):
    """Transform native dedalus coordinates (colatitude and longitude in radians)
    to latitude and longitude in degrees"""
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})

def open_h5(name,sim='s1',SNAPSHOTS_DIR = ''):
    """Load output from a dedalus simulation into an xarray dataset.
    Also adds latitude and longitude coordinates in degrees
    args:
        - name : str, simulation output name
        - sim: str, id of the ouput to open (s1, s2, s3... depending on how many restarts you ran)
            default: 's1'
        - SNAPSHOTS_DIR : str, root directory in which snapshot is saved
    returns:
        - xr.DataArray containing simulation output
    """
    ds = thetaphi_to_latlon(xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus'))
    return ds.assign_coords({'day':ds.t/24})
def open_h5s(name,sims,SNAPSHOTS_DIR = ''):
    """Load all output from a dedalus simulation into an xarray dataset.
    Also adds latitude and longitude coordinates in degrees
    args:
        - name : str, simulation output name
        - sims: list of str, usually ['s1', 's2', ... , 's{n}'] where n is the number of restarts you ran
        - SNAPSHOTS_DIR : str, root directory in which snapshot is saved
    returns:
        - xr.DataArray containing simulation output
    """
    return xr.concat([open_h5(name,sim,SNAPSHOTS_DIR) for sim in sims],dim='t')

################################################################
########################  N LEVELS  ############################
################################################################

def concat_levels(ds,N,sigma_full=None):
    """Concatenate sigma-level variables from dedalus output
    stationarywave.py outputs all 3D variables as a list of separate 2D variables,
    one per sigma level (e.g., u1,u2,...,uN).
    
    The purpose of this function is to concatenate these into 3D objects.
    It also works for staggered variables, like sigmadot.
    
    args:
        - ds: xr.DataArray, usually the ouput of open_h5 or open_h5s
        - N: number of half sigma levels
        - sigma_full: optional, if provided, will be used as the sigma coordinate for the output. 
            Should be an array of length N+1, with the first element being 0 and the last being 1.
            If not provided, will use equally spaced sigma levels.
    returns:
        - xr.DataArray with each 3D variables concatenated
    """
    nlen = 1+int(np.log10(N))
    nlen2 = 1+int(np.log10(N-1))
    sigma_varnames = [var[:-nlen] for var in ds.variables if var[-nlen:]==str(N)]
    stagger_varnames = [var[:-nlen2] for var in ds.variables if var[:-nlen2] not in sigma_varnames and var[-nlen2:]==str(N-1)]
    allvvarnames = [var+str(i) for var in sigma_varnames for i in range(1,N+1)] + [var+str(i) for var in stagger_varnames for i in range(1,N)]
    base = ds.drop(allvvarnames)
    
    if sigma_full is None:
        sigma_half = np.arange(N)/N + 1/(2*N)
        sigma_full = np.arange(N-1)/N + 1/N
    else:
        sigma_half = (sigma_full[:-1] + sigma_full[1:]) / 2
        sigma_full = sigma_full[1:-1]
    sigma_vars=[]
    stagger_vars=[]
    for var in sigma_varnames:
        dims = ds[var+str(1)].dims
        if var=='theta':
            rname = 'Theta'
        else:
            rname=var
        sigma_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N+1)],
                                     dim = xr.DataArray(sigma_half,coords={'sigma': sigma_half},dims = ['sigma'])
                                    ).transpose(*dims,'sigma').rename(rname) )
    for var in stagger_varnames:
        dims = ds[var+str(1)].dims
        stagger_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N)],
                                     dim = xr.DataArray(sigma_full,coords={'sigma_stag': sigma_full},dims = ['sigma_stag'])
                                    ).transpose(*dims,'sigma_stag').rename(var) )
        
    return xr.merge((base,*sigma_vars,*stagger_vars))


################################################################
##################  HELMHOLTZ DECOMPOSITION  ###################
################################################################

def calc_helmholtz(u_xr):
    """Calculates a streamfunction and Helmholtz decomposition on dedalus output
    Here the streamfunction psi is defined such that laplacian(psi) = curl(u)
    args:
        - u_xr: xarray.DataArray, that comes from the output of a dedalus simulation
        (opened using open_h5s and concat_levels).
        Note u should not depend on time.
    returns:
        - xarray.Dataset with four variables: u_rot (rotational part of u), 
        u_div (divergent part of u), div (divergence of u), and psi (streamfunction)
    """
    meter = 1 / 6.37122e6 # To perform the numerical calculation, we rescale all lengths by Earth's radius 
    second = 1./3600

    dealias = (3/2,3/2)
    dtype = np.float64
    Nphi = len(u_xr.longitude)
    Ntheta = len(u_xr.latitude)
    
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=6.37122e6 * meter, dealias=dealias, dtype=dtype)
    
    # placeholder for u_rot and u_div
    u_xr = u_xr.transpose('','longitude','latitude','sigma')
    u_rot_xr = 0. * u_xr.rename('u_rot')
    u_div_xr = 0. * u_xr.rename('u_div')
    divu_xr = 0. * u_xr.isel({'':0}).rename('div')
    psi_xr  = 0. * u_xr.isel({'':0}).rename('psi')
        
    for i in range(len(u_xr.sigma)):
        u = dist.VectorField(coords, name='u', bases=basis)
        u_rot = dist.VectorField(coords, name='u_rot', bases=basis)
        u.load_from_global_grid_data(u_xr.isel(sigma=i).data * meter / second)
        
        c = dist.Field(name='c')
        psi = dist.Field(name='psi', bases=basis)
        problem = d3.LBVP([c, psi], namespace=locals())
        problem.add_equation("lap(psi) + c = - div(skew(u))")
        problem.add_equation("ave(psi) = 0")
        solver = problem.build_solver()
        solver.solve()
        
        u_rot = d3.skew(d3.grad(psi)).evaluate()
        divu = d3.div(u).evaluate()
        
        u.change_scales(1)
        u_rot.change_scales(1)
        divu.change_scales(1)
        psi.change_scales(1)
        
        psi_xr[:,:,i] = psi['g']
        u_rot_xr[:,:,:,i] = u_rot['g']
        u_div_xr[:,:,:,i] = u['g']-u_rot['g']
        divu_xr[:,:,i] = divu['g']
        
    return xr.merge((u_rot_xr / (meter/second), 
                     u_div_xr / (meter/second),
                     psi_xr / (meter**2 / second),
                     divu_xr / (1/second))).transpose('','latitude','longitude','sigma')
        
################################################################
################# PRESSURE VELOCITY DIAGNOSTIC #################
################################################################

# A bug in dedalus prevents us from outputting pressure velocity at runtime when using axisymmetric basic states.
# This function allows to calculate it after the fact.

def calc_omega(ds):
    kilogram = 1.
    meter = 1 / 6.37122e6 # To perform the numerical calculation, we rescale all lengths by Earth's radius 
    second = 1./3600

    Ntheta = len(ds.latitude)
    Nsigma = len(ds.sigma)
    
    # Instantiate a stationary wave problem - takes care of instantiating coordinate system and variables,
    # and contains the function needed to calculate the pressure velocity.
    problem_holder = StationaryWaveProblem(Ntheta, Nsigma, True, False, '','')

    ds_new = ds.copy().transpose('','longitude','latitude','sigma','sigma_stag')
    # Broadcast the background fields in longitude
    for var in 'ubar', 'lnpsbar', 'sigmadotbar':
        ds_new[var].data = ds_new[var].isel(longitude=0).broadcast_like(ds_new[var]).data
        
    # placeholder for omega
    omega = 0. * ds.sigmadotbar.transpose('longitude','latitude','sigma_stag').rename('omega')

    # Populate variables in the stationary wave problem with input data
    for i in range(1,Nsigma+1):
        problem_holder.vars[f'u{i}'].load_from_global_grid_data(ds_new.u.isel(sigma=i-1).data * meter / second)
        problem_holder.vars[f'ubar{i}'].load_from_global_grid_data(ds_new.ubar.isel(sigma=i-1).data * meter / second)
        problem_holder.vars[f'lnps'].load_from_global_grid_data(ds_new.lnps.data)
        problem_holder.vars[f'lnpsbar'].load_from_global_grid_data(ds_new.lnpsbar.data)

    for i in range(1,Nsigma):
        problem_holder.vars[f'sigmadotbar{i}'].load_from_global_grid_data(ds_new.sigmadotbar.isel(sigma_stag=i-1).data * 1 / second)

    # Calculate omega at each level
    for i in range(1,Nsigma):
        # output perturbation pressure velocity
        omega_i = problem_holder._calc_omega(i).evaluate()
        omega_i.change_scales(1)
        omega[:,:,i-1] = omega_i['g'] / (kilogram / meter / second**3)
    return omega.transpose('latitude','longitude','sigma_stag')