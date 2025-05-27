import numpy as np
import xarray as xr 
import dedalus.public as d3
import time
import h5py
import matplotlib.pyplot as plt


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

def concat_levels(ds,N):
    """Concatenate sigma-level variables from dedalus output
    stationarywave.py outputs all 3D variables as a list of separate 2D variables,
    one per sigma level (e.g., u1,u2,...,uN).
    
    The purpose of this function is to concatenate these into 3D objects.
    It also works for staggered variables, like sigmadot.
    
    args:
        - ds: xr.DataArray, usually the ouput of open_h5 or open_h5s
        - N: number of sigma levels
    returns:
        - xr.DataArray with each 3D variables concatenated
    """
    nlen = 1+int(np.log10(N))
    nlen2 = 1+int(np.log10(N-1))
    sigma_varnames = [var[:-nlen] for var in ds.variables if var[-nlen:]==str(N)]
    stagger_varnames = [var[:-nlen2] for var in ds.variables if var[:-nlen2] not in sigma_varnames and var[-nlen2:]==str(N-1)]
    allvvarnames = [var+str(i) for var in sigma_varnames for i in range(1,N+1)] + [var+str(i) for var in stagger_varnames for i in range(1,N)]
    base = ds.drop(allvvarnames)
    
    sigma_grid = np.arange(N)/N + 1/(2*N)
    stagger_grid = np.arange(N-1)/N + 1/N
    sigma_vars=[]
    stagger_vars=[]
    for var in sigma_varnames:
        dims = ds[var+str(1)].dims
        if var=='theta':
            rname = 'Theta'
        else:
            rname=var
        sigma_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N+1)],
                                     dim = xr.DataArray(sigma_grid,coords={'sigma': sigma_grid},dims = ['sigma'])
                                    ).transpose(*dims,'sigma').rename(rname) )
    for var in stagger_varnames:
        dims = ds[var+str(1)].dims
        stagger_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N)],
                                     dim = xr.DataArray(stagger_grid,coords={'sigma_stag': stagger_grid},dims = ['sigma_stag'])
                                    ).transpose(*dims,'sigma_stag').rename(var) )
        
    return xr.merge((base,*sigma_vars,*stagger_vars))


################################################################
########################  HELMHOLTZ DECOMPOSITION  ############################
################################################################

def calc_helmholtz(u_xr):
    """Calculates a streamfunction and Helmholtz decomposition on dedalus output
    Here the streamfunction psi is defined such that laplacian(psi) = curl(u)
    args:
        - u_xr: xarray.DataArray, that comes from the output of a dedalus simulation
        (opened using open_h5s and concat_levels).
        Note u should not depend on time, and should be nondimensional.
    returns:
        - xarray.Dataset with four variables: u_rot (rotational part of u), 
        u_div (divergent part of u), div (divergence of u), and psi (streamfunction)
    """
    meter = 1 / 6.37122e6 # To perform the numerical calculation, we rescale all lengths by Earth's radius 
    second = 1.

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
        
    