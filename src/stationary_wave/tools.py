import numpy as np
import xarray as xr 
import dedalus.public as d3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Find project root for default data path
import os.path as op
ROOT = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir))
DATA_PATH = op.join(ROOT, 'data', 'output') + '/'

################################################################
###################  OPENING DEDALUS OUTPUT  ###################
################################################################

def thetaphi_to_latlon(ds):
    """Transform native dedalus coordinates (colatitude and longitude in radians)
    to latitude and longitude in degrees"""
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})

def _open_h5(name,sim='s1',data_path = DATA_PATH):
    """Load one output from a dedalus simulation into an xarray dataset.
    Also adds latitude and longitude coordinates in degrees.

    Parameters
    ----------
    name : str
        Simulation output name
    sim : str, optional
        Usually 's{n}' where n is the index of the restart run you want to open. Default is 's1'.
    data_path : str, optional
        Root directory in which snapshots are saved. Default is {ROOT}/data/output/.

    Returns
    -------
    output : xarray.DataArray
        Concatenated output from all specified restarts.
    """
    ds = xr.open_dataset(data_path+f"{name}/{name}_{sim}.h5",engine='dedalus')
    # Add latitude and longitude coordinates + missing longitude point
    return wrap_lon(thetaphi_to_latlon(ds)).isel(longitude=slice(1,None)).assign_coords({'day':ds.t/24})

def open_h5s(name,sims,data_path = DATA_PATH):
    """Load multiple output (resulting from restarts) from a dedalus simulation into an xarray dataset.
    Also adds latitude and longitude coordinates in degrees.
    Also retrieves sigma levels.

    Parameters
    ----------
    name : str
        Simulation output name
    sims : list of str
        Usually ['s1', 's2', ... , 's{n}'] where n is the number of restarts you ran
    data_path : str, optional
        Root directory in which snapshots are saved. Default is {ROOT}/data/output/.

    Returns
    -------
    output : xarray.DataArray
        Concatenated output from all specified restarts.
    """
    ds = xr.concat([_open_h5(name,sim,data_path) for sim in sims],dim='t')
    return ds

def concat_levels(ds):
    """Concatenate sigma-level variables from dedalus output.
    The model outputs all 3D variables as a list of separate 2D variables,
    one per sigma level (e.g., u1,u2,...,uN).
    
    The purpose of this function is to concatenate these into 3D objects.
    It also works for staggered variables, like sigmadot.
    
    Parameters
    ----------
    ds : xarray.DataArray
        Usually the ouput of open_h5 or open_h5s.
    sigma_full : numpy.array
        Full sigma levels of the output. 
        Should be sorted increasingly, with the first element being 0 and the last being 1.

    Returns
    -------
    output : xarray.DataArray
        Input dataset with all 3D variables concatenated.
        Contains a coordinate 'sigma' for half levels and 'sigma_stag' for full levels.
    """
    sigma_full = ds.attrs['sigma_full']
    N = len(sigma_full) - 1

    nlen = 1+int(np.log10(N))
    nlen2 = 1+int(np.log10(N-1))
    sigma_varnames = [var[:-nlen] for var in ds.variables if var[-nlen:]==str(N)]
    stagger_varnames = [var[:-nlen2] for var in ds.variables if var[:-nlen2] not in sigma_varnames and var[-nlen2:]==str(N-1)]
    allvvarnames = [var+str(i) for var in sigma_varnames for i in range(1,N+1)] + [var+str(i) for var in stagger_varnames for i in range(1,N)]
    base = ds.drop(allvvarnames)
    
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

def process_sigma_sim(name,sims,data_path = DATA_PATH,avg = np.array([20,30])):
    """
    Load output from a sigma-level stationary wave simulation, and average some variables in time

    Parameters
    ----------
    name : str
        Simulation output name
    sims : list of str
        Usually ['s1', 's2', ... , 's{n}'] where n is the number of restarts you ran
    data_path : str, optional
        Root directory in which snapshots are saved. Default is {ROOT}/data/output/.
    avg : array-like of two floats, optional
        Start and end time (in days) of the averaging period. Default is [20,30].

    Returns
    -------
    output : xarray.DataArray
        Dataset with aimulation output + time-averaged variables
    """
    sim = open_h5s(name,sims)
    # Read in sigma levels
    sigma_full = np.loadtxt(data_path+f"{name}/sigma_full.txt")
    sim.attrs['sigma_full'] = sigma_full
    sim = concat_levels(sim)

    for var in sim.data_vars:
        test = sim[var].isel(t=0,longitude=slice(1,-1))
        if np.sum(np.isnan(test.data)) == len(test.data.flatten()): # Then this variable is a basic state variable
            sim[var] = sim[var].isel(t=0,longitude=0)

    sim_mean = sim.sel(t=slice(*(avg*24))).mean('t').transpose('','latitude','longitude','sigma','sigma_stag')

    for var in ['u','T','lnps','zeta','sigmadot','Phiprime','div','omega']:
        sim[var+'_mean'] = sim_mean[var]

    return sim.transpose('t','','latitude','longitude','sigma','sigma_stag')



################################################################
##################  HELMHOLTZ DECOMPOSITION  ###################
################################################################

def calc_helmholtz(u_xr):
    """Calculates a streamfunction and Helmholtz decomposition on dedalus output
    Here the streamfunction psi is defined such that laplacian(psi) = curl(u)

    Parameters
    ----------
    u_xr : xarray.DataArray
        Output of a dedalus simulation (opened using open_h5s and concat_levels),
        averaged in time. Note u should not depend on time.

    Returns
    -------
    output : xarray.Dataset
        Dataset with four variables: u_rot (rotational part of u), 
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
#################### MISCELLANEOUS TOOLS #######################
################################################################

def add_wind(ax,u,v,scale=None,key=True,ref=1,unit='m/s',keypos=(0.93,0.95),nm=1):
    """
    Adds a quiver plot to a matplotlib axis

    Parameters
    ----------
    ax : matplotlib axis
        Axis to which the quiver plot will be added.
    u : xarray.DataArray
        Zonal wind component.
    v : xarray.DataArray
        Meridional wind component.
    scale : float, optional
        Scale for the quiver plot. Default is None.
    key : bool, optional
        Whether to add a quiver key. Default is True.
    ref : float, optional
        Reference value for the quiver key. Default is 1.
    unit : str, optional
        Unit for the quiver key label. Default is 'm/s'.
    keypos : tuple of float, optional
        Position of the quiver key in axis coordinates. Default is (0.93, 0.95).
    nm : int, optional
        Step size for downsampling the wind field. Default is 1.
    """
    X = u.latitude.expand_dims({"longitude":u.longitude}).transpose()
    Y = u.longitude.expand_dims({"latitude":u.latitude})
    n=nm;m=nm
    Q = ax.quiver(np.array(Y)[::n,::m],np.array(X)[::n,::m], np.array(u)[::n,::m], np.array(v)[::n,::m],color="k",scale=scale,transform=ccrs.PlateCarree())
    if key:
        ax.quiverkey(Q, *keypos, ref, label='%i %s'%(ref,unit), labelpos='E', coordinates='axes',color='k')

def wrap_lon(ds,lon='lon'):
    """
    Add missing longitude point to the dataset by wrapping it around.
    """
    try: 
        ds[lon]
    except KeyError:
        lon = 'longitude'
    return ds.pad({lon:1}, mode="wrap").assign_coords({lon:ds[lon].pad({lon:1}, mode="reflect", reflect_type="odd")})

def quickplot(field, levels=None, cmap='RdBu_r', ax=None):
    """
    Quick plot of a field on a map using cartopy
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    field = wrap_lon(field)
    field.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=levels, cmap=cmap, extend='both')
    ax.coastlines()
    return ax

def prime(field):
    """
    Remove the zonal mean from a field.
    """
    try:
        return field - field.mean('longitude')
    except (AttributeError,ValueError):
        return field - field.mean('lon')
    
def lon_180_to_360(da,longitude='longitude'):
    """
    Convert longitude from [-180,180] to [0,360]

    Parameters
    ----------
    da : xarray.DataArray
        Input data array with a longitude coordinate.
    longitude : str, optional
        Name of the longitude coordinate. Default is 'longitude'; the code will also check for 'lon'.
    """
    da = da.copy()
    try:
        da[longitude]
    except KeyError:
        longitude = 'lon'
    da.coords[longitude] = da.coords[longitude] % 360
    da = da.sortby(da[longitude])
    return da


def lon_360_to_180(da,longitude='longitude'):
    """
    Convert longitude from [0,360] to [-180,180]

    Parameters
    ----------
    da : xarray.DataArray
        Input data array with a longitude coordinate.
    longitude : str, optional
        Name of the longitude coordinate. Default is 'longitude'; the code will also check for 'lon'.
    """
    da = da.copy()
    try:
        da[longitude]
    except KeyError:
        longitude = 'lon'
    da.coords[longitude] = (da.coords[longitude]+180) % 360 - 180
    da = da.sortby(da[longitude])
    return da