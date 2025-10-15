import abc
import jax.numpy as jnp
from jax import random
import jax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from numpyro.infer import MCMC, NUTS
import numpyro
import jax.numpy as jnp
tfb=tfp.bijectors
tfpk = tfp.math.psd_kernels
tfpe=tfp.experimental
import abc
import jax
from datetime import datetime
mcmc = tfp.mcmc
Root = tfd.JointDistributionCoroutine.Root
class HMC_Sampler(abc.ABC):
    """Base class for HMC-based samplers."""
    def __init__(self, config):
        self.config=config
        self.step_size_initial = config.sampling.step_size_initial
        self.kernel = config.sampling.method
        self.warmup = config.sampling.warmup_steps
        self.samples = config.sampling.samples
        self.adapt_step_size = config.sampling.adapt_step_size
        self.adapt_mass_matrix = config.sampling.adapt_mass_matrix
        self.num_chains = config.sampling.num_chains
        self.max_tree_depth = config.sampling.max_tree_depth
        self.acceptance_prob = config.sampling.acceptance_prob
        self.num_warmup_in_loop=config.sampling.num_warmup_in_loop
        self.num_samples_in_loop=config.sampling.num_samples_in_loop
        self.num_loops=config.sampling.num_loops
        self.num_steps_train_loss_model=config.sampling.num_steps_train_loss_model
        self.lr_train_loss_model=config.sampling.lr_train_loss_model
        
        self.deviation_model=None
        self.theta=None
        self.data=None
        self.observation=None
        self.flow=None
        self.cond_dim=None
        self.v_dim=None
        self.mean=None
        self.std=None
        self.log_prob_fn=None
        self.num_obs=None
        self.model=None
        self.unconstraining_bijectors=None
        self.target_log_prob=None
    
    
    def run_model_with_timing(self, rng, i, ebv=0):
        start_time = datetime.now()
        print(f"Started inference for star {i} at:", start_time)
        
        result = self.run_model(rng, i, ebv)
        
        end_time = datetime.now()
        print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = end_time - start_time
        print(f"Total time: {duration}")
        
        return result

    
    def run_model(self, rng, i, ebv=0):
        
        # Introduce the model distribution, this is only to generate initial states for the MCMC

        ###### scale and format the data for the inference procedure 
        ###################################


        photometry=self.photometry
        photometry_covariance=self.photometry_covariances
        parallax=self.parallax
        parallax_error=self.parallax_error
        l=self.l
        b=self.b
        L=self.L
        alpha=self.alpha
        beta=self.beta
        photometry_single = photometry[i]
        photometry_covariance_single = photometry_covariance[i]
        parallax_single = parallax[i]
        parallax_error_single = parallax_error[i]
        l_single = l[i]
        b_single = b[i]
        alpha_single = alpha[i]
        L_single = L[i]
        beta_single = beta[i]

        rng, rng_key = random.split(rng)
        distance_init = 1000 / parallax_single + jax.random.normal(rng, ())
        # Removed print for JAX compatibility
        rng, rng_key = random.split(rng)
        colours_init =(photometry_single[..., 1:] - photometry_single[..., 0]) + jax.random.normal(rng, photometry_single[..., 1:].shape) * 0.1
        rng, rng_key = random.split(rng)
        # Note: colours_init already in physical units (no normalization needed for inference data)
        ebv_init = 0.1 + jax.random.normal(rng, ()) * 0.01
        M_g_init=photometry_single[..., 0] +5-5*jnp.log10(distance_init) + jax.random.normal(rng, *()) * 0.01

        ### add extinction for valdiation set
        bprp=photometry_single[...,1]-photometry_single[...,2]
        extinction_coefficients=(self.extinction_law)(jnp.abs(ebv),bprp)
        photometry_single=photometry_single+extinction_coefficients*ebv
        ###########
        ## set initializations for chains
        ### 

        
    
        

        mc_kernel = NUTS(
                    self.model,
                    step_size=self.step_size_initial,
                    adapt_step_size=self.adapt_step_size,
                    adapt_mass_matrix=self.adapt_mass_matrix,
                    max_tree_depth=self.max_tree_depth,
                                target_accept_prob=self.acceptance_prob ,
                                init_strategy=numpyro.infer.initialization.init_to_value(values=
                                                {'colors':(colours_init),'d': distance_init,'ebv':ebv_init,
                                                 'Mg':M_g_init})
        )   
        

            
        # Use more chains for better parallelization
        effective_num_chains = max(self.num_chains, 1)  # At least 1 chain for parallelization
        mcmc = MCMC(
            mc_kernel,
            num_warmup=self.warmup,
            num_samples=self.samples,
            num_chains=effective_num_chains,
        )
        rng, rng_key = random.split(rng)

        # Timing moved outside for JAX compatibility
        rng, rng_key = random.split(rng)

        mcmc.run(rng_key,
            obs_photometry=photometry_single,
            photometry_covariance=photometry_covariance_single,
            obs_parallax=parallax_single,
            parallax_error=parallax_error_single,
            alpha=alpha_single,
            L=L_single,
            beta=  beta_single,
            l= l_single,
            b= b_single,
            flow_logp=self.flow.flow.log_prob,
            flow_mag=self.flow.distance_flow.log_prob,
            mean=self.mean,
            std=self.std,
            config=self.config,
            extinction_law=self.extinction_law,
            train_stats_mean=self.train_stats_mean,
            train_stats_std=self.train_stats_std
            )
                

        photometry=mcmc.get_samples()['photometry']
        ebv=mcmc.get_samples()['ebv']
        d=mcmc.get_samples()['d']
        Mg= mcmc.get_samples()['Mg']
        observed_photometry=mcmc.get_samples()['observed_photometry']
        all_vars=jnp.concatenate([photometry,observed_photometry,ebv[...,None],d[...,None],Mg[...,None]], axis=-1)

        # Timing operations removed for JAX compatibility
        return all_vars






