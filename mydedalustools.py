import numpy as np
import xarray as xr 
import dedalus.public as d3
import time
import h5py
import matplotlib.pyplot as plt


def theta_to_lat(ds):
    return ds.assign_coords({'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'theta':'latitude'})
def thetaphi_to_latlon(ds):
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})
def open_h5(name,sim='s1',SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"):
    ds = thetaphi_to_latlon(xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus'))
    return ds.assign_coords({'day':ds.t/24})
def open_h5s(name,sims,SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"):
    return xr.concat([open_h5(name,sim,SNAPSHOTS_DIR) for sim in sims],dim='t')

################################################################
########################  N LEVELS  ############################
################################################################

def concat_levels(ds,N):
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

