import os
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
#from gaiaxpy import convert,calibrate
#from zero_point import zpt
import healpy as hp
#import minimint
import jax.numpy as jnp





def galactic_to_healpix_nest(lon, lat, nside=32):
    # Convert Galactic (l, b) to Equatorial (RA, Dec)
    coords = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame='galactic')
    ra = coords.icrs.ra.deg
    dec = coords.icrs.dec.deg

    # Convert RA, Dec to theta (colatitude) and phi (longitude) in radians
    theta = np.radians(90.0 - dec)  # colatitude
    phi = np.radians(ra)            # longitude

    # Get HEALPix index (nest ordering)
    healpix_index = hp.ang2pix(nside, theta, phi, nest=True)
    return healpix_index



def distance_prior_components(lon,lat,prefix='Data'):
    """
    This requires the prior_summary.csv from the Bailer Jones
    data products to be in the Data folder.
    """
    healpixels= galactic_to_healpix_nest(lon,lat)
    coeff=pd.read_csv(os.getcwd()+'/Data/prior_summary.csv')
    ls=[]
    alphas=[]
    betas=[]
    for healpix in healpixels:
        coeff_temp=coeff[coeff['healpix']==int(healpix)]
        l=coeff_temp['GGDrlen'].astype(float)
        alpha=coeff_temp['GGDalpha'].astype(float)
        beta=coeff_temp['GGDbeta'].astype(float)

        ls.append(l.values)
        alphas.append(alpha.values)
        betas.append(beta.values)
    l=np.concatenate(ls)
    alpha=np.concatenate(alphas)
    beta=np.concatenate(betas)
    return jnp.array(alpha),jnp.array(l),jnp.array(beta)

