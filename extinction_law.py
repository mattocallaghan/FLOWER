import jax.numpy as jnp
from jax import jit
import jax
#extinction law from https://arxiv.org/pdf/2210.15918

# Coefficient data (converted to JAX-compatible structures)


# Core extinction calculation


# Main function
def get_extinction_function(bands,mean_val=False,teff=False):
    # the first value is the simple version
    R_function_coefficients = ({"FUV": [6.973, 4.68e-10, -1.26e-05, 0.112, 6.520, -10.401, -319.943],
                                            "NUV": [7.293, 2.57e-12, -5.19e-07, 0.0076, -1.022, -3.608, -19.704],
                                            "g"  : [3.248, -8.98e-11, 1.72e-06, -0.0108, -0.556, -0.712, 25.885],
                                            "r"  : [2.363, -5.76e-11, 1.11e-06, -0.00704, -0.273, -0.542, 17.255],
                                            "i"  : [1.791, -3.56e-11, 6.91e-07, -0.00443, -0.417, -0.250, 11.249],
                                            "z"  : [1.398, -3.12e-11, 6e-07, -0.00381, -0.876, 0.253, 9.372],
                                            "y"  : [1.146, -2.67e-11, 5.05e-07, -0.00317, -0.957, 0.407, 7.699],
                                            "J"  : [0.748, -3.99e-12, 7.3e-08, -0.000448, 0.055, -0.176, 1.720],
                                            "H"  : [0.453, 3.85e-13, -9.67e-09, 7.91e-05, -0.366, 0.225, 0.213],
                                            "Ks" : [0.306, 0, 0, 0, 0, 0, 0.306],
                                            "W1" : [0.194, -7.18e-13, 1.11e-08, -5.18e-05, 0.128, -0.038, 0.261],
                                            "W2" : [0.138, 5.4e-12, -9.97e-08, 0.000615, 0.030, 0.062, -1.149],
                                            "W3" : [0.183, 0, 0, 0, 0, 0, 0.183],
                                            "W4" : [0.084, 0, 0, 0, 0, 0, 0.084],
                                            "BP" : [2.998, -1.27e-11, 2.76e-07, -0.00188, -0.673, -0.531, 7.185],
                                            "G"  : [2.364, -1.05e-11, 2.24e-07, -0.00145, -0.681, -0.381, 5.376],
                                            "RP" : [1.737, -7.88e-12, 1.76e-07, -0.00126, -0.630, -0.057, 4.670],
                                            "u'" : [4.500, -1.29e-10, 2.52e-06, -0.0162, 1.741, -2.760, 39.080],
                                            "g'" : [3.452, -9.46e-11, 1.82e-06, -0.0115, 0.221, -1.210, 27.571],
                                            "r'" : [2.400, -5.47e-11, 1.07e-06, -0.00691, -0.080, -0.568, 17.129],
                                            "i'" : [1.799, -3.98e-11, 7.8e-07, -0.00502, -0.675, 0.020, 12.450],
                                            "z'" : [1.299, -3.49e-11, 6.79e-07, -0.00433, -2.027, 1.144, 10.198],
                                            "BP-RP": [1.261, -4.864e-12, 1.006e-07, -6.201e-04, -4.329e-02, -4.741e-01, 2.515],})
    teff_range = ({"FUV": [7000,10000], "NUV": [4000,9000], "g" : [4000,8000],  "r" : [4000,8000], 
                               "i"  : [4000,8000],  "z"  : [4000,9000], "y" : [4000,8000],  "J" : [4000,8000],
                               "H"  : [4000,8000],  "Ks" : [4000,8000], "W1": [4000,8000],  "W2": [4000,8000],
                               "W3" : [4000,4500],  "W4" : [4000,4500], "BP": [4000,10000], "G" : [4000,10000],
                               "RP" : [4000,9000],  "u'" : [4000,9000], "g'": [4000,8000],  "r'": [4000,8000],
                               "i'" : [4000,8000],  "z'" : [4000,8000], "BP-RP" : [4000,9000]})
    Teff_BPRP0_coeff = jnp.array([1187.26193963, -4925.6478914, 8268.06749031, -9038.59055765, 9693.00903561])
    if(mean_val):
        coeff_array = jnp.stack([jnp.array(R_function_coefficients[key][0:1]) for key in bands])
    else:
        coeff_array = jnp.stack([jnp.array(R_function_coefficients[key][1:]) for key in bands])
    teff_array = jnp.stack([jnp.array(teff_range[key]) for key in bands])

    @jit
    def extinction_coefficient_bprp(ebv,bprp):   
        teff = jnp.polyval(Teff_BPRP0_coeff, bprp)
        teff = jnp.clip(teff, jnp.clip(4000,teff_array[:,0],jnp.inf), teff_array[:,1])  # teff should be shape (n_passbands
        # Apply boundary limits
        ebv = jnp.clip(ebv, 0, 0.5)
        a, b, c, d, e, f = coeff_array.T
        return a * teff**3 + b * teff**2 + c * teff + d * ebv**2 + e * ebv + f #output (n_passbands, )

    @jit
    def extinction_coefficient_teff(ebv,teff):   
        teff = jnp.clip(teff, teff_array[:,0], teff_array[:,1])  # teff should be shape (n_passbands
        # Apply boundary limits
        ebv = jnp.clip(ebv, 0, 0.5)
        a, b, c, d, e, f = coeff_array.T
        return a * teff**3 + b * teff**2 + c * teff + d * ebv**2 + e * ebv + f #output (n_passbands, )


    if(teff==True):
        return extinction_coefficient_teff
    else:
        return extinction_coefficient_bprp

# Test the function
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    extinction_coefficient_bprp = get_extinction_function(["BP","RP","G","J","H","Ks","W1","W2","W3","W4","FUV","NUV","g","r","i","z","y","u'","g'","r'","i'","z'","BP-RP"], mean_val=False)
    ebv = jnp.linspace(0, 0.5, 100)
    bprp = jnp.linspace(0.0, 5.0, 100)
    ebv, bprp = jnp.meshgrid(ebv, bprp)
    ebv = ebv.flatten()
    bprp = bprp.flatten()
    outs = jax.vmap(extinction_coefficient_bprp)(ebv, bprp)
    
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(bprp, outs[:, 2].flatten() / 3.1, c=ebv, cmap='viridis')
    plt.colorbar(sc, label='$A_v$ [mag]')
    plt.xlabel('$m_{BP}-m_{RP}$ [mag]')
    plt.ylabel('$K_G$')
    plt.savefig('output/extinction_law/extinction_plot.pdf', format='pdf', dpi=300)
    plt.close()
