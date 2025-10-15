import jax.numpy as jnp
from jax import random
import jax
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb=tfp.bijectors
tfpk = tfp.math.psd_kernels
tfpe=tfp.experimental
from extinction_law import get_extinction_function
from stellar_data_utils import distance_prior_components
import jax

mcmc = tfp.mcmc
Root = tfd.JointDistributionCoroutine.Root
import numpyro
import numpyro.distributions as dist
from inference_models.base_class import HMC_Sampler

########################################################################################
########################################################################################
# Flower
########################################################################################
########################################################################################


class FLower_Embedding_NLPE(HMC_Sampler):
    """HMC sampler for synthetic inference using a normalizing flow."""
    def __init__(self, config):
        super().__init__(config=config)

    def build(self, flow, data):
        assert data is not None
        rng, rng_key = random.split(jax.random.PRNGKey(self.config.seed))

        self.flow = flow
        self.cond_dim = flow.cond_dim#-1
        self.summary_dim= self.config.data.summary_dim

        self.v_dim = self.config.data.vector_dim_inference
        self.mean = flow.mean
        self.std = flow.std
        # Add train_stats for consistent flow normalization
        self.train_stats_mean = flow.train_stats_mean  
        self.train_stats_std = flow.train_stats_std
            # 'ruwe','l','b','parallax_error']+ ['source_id','parallax']
        self.observation = data
        self.extinction_law=get_extinction_function(self.config.sampling.bands)

        ### data - dynamic structure based on config
        # Check if 2MASS data is included
        has_twomass = getattr(self.config.data, 'twomass', False)
        n_bands = 11 if has_twomass else 8  # 11 bands for 2MASS (G,BP,RP,g,r,i,z,y,J,H,Ks), 8 for PS1 only
        
        if has_twomass:
            # 2MASS inference dataset: [ruwe, l, b, parallax_error, source_id, parallax, 11_photometry_bands, 11_photometry_errors]
            # Indices:                 [0,    1, 2, 3,             4,         5,        6:17,             17:28]
            INFERENCE_SUMMARY_DIM = 3  # ruwe, l, b  
            INFERENCE_VECTOR_DIM = 11  # 11 photometric bands
            INFERENCE_COND_DIM = 3     # conditioning dimensions
            
            print(self.observation.shape)
            # Extract photometry: columns 6:17 (after ruwe,l,b,parallax_error,source_id,parallax)
            self.photometry = jnp.array(self.observation[..., 6:17])  # 11 photometric bands
            self.photometry_error = jnp.array(self.observation[..., 17:28])  # 11 photometric errors  
        else:
            # Original inference dataset: [ruwe, l, b, parallax_error, source_id, parallax, 8_photometry_bands, 8_photometry_errors]
            # Indices:                   [0,    1, 2, 3,             4,         5,        6:14,             14:22]
            INFERENCE_SUMMARY_DIM = 3  # ruwe, l, b  
            INFERENCE_VECTOR_DIM = 8   # 8 photometric bands
            INFERENCE_COND_DIM = 3     # conditioning dimensions
            
            print(self.observation.shape)
            # Extract photometry: columns 6:14 (after ruwe,l,b,parallax_error,source_id,parallax)
            self.photometry = jnp.array(self.observation[..., 6:14])  # 8 photometric bands
            self.photometry_error = jnp.array(self.observation[..., 14:22])  # 8 photometric errors  
        
        self.photometry_covariances = jnp.abs(jax.vmap(jnp.diag)(self.photometry_error**2)) #add a small constant to avoid numerical issues

        # Extract parallax data
        self.parallax=jnp.array(self.observation[..., 5])      # parallax at index 5
        self.parallax_error=jnp.array(self.observation[..., 3]) # parallax_error at index 3

        # Extract galactic coordinates  
        self.l= self.observation[..., 1]  # l at index 1
        self.b= self.observation[..., 2]  # b at index 2
        self.alpha,self.L,self.beta=distance_prior_components(self.l.flatten(),self.b.flatten())


        


        def model(obs_photometry,photometry_covariance,obs_parallax,parallax_error,alpha,L,beta,l,b,
                    flow_logp,flow_mag,mean,std,config,extinction_law,train_stats_mean,train_stats_std):
            #flow_logp=copy.deepcopy(self.flow.flow.log_prob)
            #flow_mag=copy.deepcopy(self.flow.distance_flow.log_prob)
            distance_prior=lambda d: beta*jnp.log(d)-(d/L)**alpha
            ### sample distance in jpc
            d = numpyro.sample('d', dist.Uniform((0.5), (50000.0)))
            numpyro.factor('dist_prior', distance_prior(d/1.0)) #prior on distance
            
            # sample a0
            ebv = numpyro.sample('ebv', dist.Uniform(-0.05, 0.5))
            # sample Mg
            Mg = numpyro.sample('Mg', dist.Uniform(5+5-5*jnp.log10(d), 20.5+5-5*jnp.log10(d)))
            Z = 5*jnp.log10((jnp.sin(jnp.abs(jnp.radians(b))) * d))-5
            R=5*jnp.log10(jnp.sqrt(8000**2 + (d*jnp.cos(jnp.radians(b)))**2 - 2*8000*(d*jnp.cos(jnp.radians(b)))*jnp.cos(jnp.radians(l))))-5 #in pc
            Z = (Z - mean[config.data.summary_dim])/std[config.data.summary_dim]
            R = (R - mean[config.data.summary_dim+1])/std[config.data.summary_dim+1]
            Mg=(Mg - mean[config.data.summary_dim+2])/std[config.data.summary_dim+2]
            mu=5*jnp.log10(d)-5
            scaled_distance= (mu - mean[config.data.summary_dim-1]) / std[config.data.summary_dim-1]
            #numpyro.factor('mag_prior', (flow_mag)(Mg[None],scaled_distance[None])) #prior on distance
            # sample the photometry
            photometry=numpyro.sample('phot', dist.Uniform(low=-10*jnp.ones(config.model.flow_dimension), high=10*jnp.ones(config.model.flow_dimension)).to_event(1))
            numpyro.factor('log_prob', (flow_logp)(photometry,jnp.array([Z,R,Mg])))
            photometry = photometry*self.std[self.config.data.summary_dim+self.config.model.cond_dim:]+ self.mean[self.config.data.summary_dim+self.config.model.cond_dim:]
            Mg=Mg*std[config.data.summary_dim+2]+mean[config.data.summary_dim+2]
            photometry=jnp.concatenate([Mg[None],photometry],-1)
            numpyro.deterministic('photometry', photometry)
            photometry=photometry.at[...,0].set(photometry[...,0]+5*jnp.log10(d)-5)
            photometry=photometry.at[...,1:].set(photometry[...,1:]+photometry[...,0])
            numpyro.deterministic('observed_photometry', photometry)
            bprp=photometry[...,1]-photometry[...,2]
            extinction_coefficients=(extinction_law)(jnp.abs(ebv),bprp)
            photometry=photometry+extinction_coefficients*ebv
            parallax=1000.0/d #in mas

            numpyro.sample('obs_x', dist.MultivariateNormal(loc=photometry, covariance_matrix=(photometry_covariance)), obs=obs_photometry)
            numpyro.sample('obs_parallax', dist.Normal(loc=parallax, scale=(parallax_error)), obs=obs_parallax)

        def model(obs_photometry,photometry_covariance,obs_parallax,parallax_error,alpha,L,beta,l,b,
                    flow_logp,flow_mag,mean,std,config,extinction_law,train_stats_mean,train_stats_std):

            distance_prior=lambda d: beta*jnp.log(d)-(d/L)**alpha
            ### sample distance in jpc
            d = numpyro.sample('d', dist.Uniform((0.5), (50000.0)))
            numpyro.factor('dist_prior', distance_prior(d/1.0)) #prior on distance
            
            # sample a0
            ebv = numpyro.sample('ebv', dist.Uniform(-0.05, 0.5))
            # sample absolute magnitude Mg  
            Mg = numpyro.sample('Mg', dist.Uniform(5+5-5*jnp.log10(d), 20.5+5-5*jnp.log10(d)))
            
            # Convert galactic coordinates to galactocentric Z, R (matching loss function exactly)
            R_sun = 8000.0
            Z_sun = 25.0
            l_rad = jnp.radians(l)
            b_rad = jnp.radians(b)
            x_gc = d * jnp.cos(b_rad) * jnp.cos(l_rad)
            y_gc = d * jnp.cos(b_rad) * jnp.sin(l_rad)
            z_gc = d * jnp.sin(b_rad)
            R_gc = jnp.sqrt((R_sun - x_gc)**2 + y_gc**2)
            Z_gc = jnp.abs(z_gc - Z_sun)
            log_Z_physical = jnp.log10(Z_gc + 1e-6)
            log_R_physical = jnp.log10(R_gc + 1e-6)

            # Determine number of colors based on config
            has_twomass = getattr(config.data, 'twomass', False)
            n_colors = 10 if has_twomass else 7
            
            # sample photometry colors 
            colors=numpyro.sample('colors', dist.Uniform(low=-10*jnp.ones(n_colors), high=10*jnp.ones(n_colors)).to_event(1))
            
            # Prepare flow inputs: [M_G, colors] using train_stats normalization
            flow_input_physical = jnp.concatenate([jnp.array([Mg]), colors])
            
            if has_twomass:
                # 2MASS case: train_stats structure [10 colors, M_G, log_R, log_Z] = indices [0:10, 10, 11, 12]
                flow_input_mean = jnp.concatenate([train_stats_mean[10:11], train_stats_mean[0:10]])
                flow_input_std = jnp.concatenate([train_stats_std[10:11], train_stats_std[0:10]]) 
                # Flow conditions: [log_Z, log_R] from indices [12, 11]
                flow_condition_mean = jnp.array([train_stats_mean[12], train_stats_mean[11]])
                flow_condition_std = jnp.array([train_stats_std[12], train_stats_std[11]])
            else:
                # Original case: train_stats structure [7 colors, M_G, log_R, log_Z] = indices [0:7, 7, 8, 9]
                flow_input_mean = jnp.concatenate([train_stats_mean[7:8], train_stats_mean[0:7]])
                flow_input_std = jnp.concatenate([train_stats_std[7:8], train_stats_std[0:7]]) 
                # Flow conditions: [log_Z, log_R] from indices [9, 8]
                flow_condition_mean = jnp.array([train_stats_mean[9], train_stats_mean[8]])
                flow_condition_std = jnp.array([train_stats_std[9], train_stats_std[8]])
            
            flow_input_norm = (flow_input_physical - flow_input_mean) / flow_input_std
            
            # Flow conditions: [log_Z, log_R] using train_stats normalization  
            flow_condition_physical = jnp.array([log_Z_physical, log_R_physical])
            flow_condition_norm = (flow_condition_physical - flow_condition_mean) / flow_condition_std
            
            numpyro.factor('log_prob', (flow_logp)(flow_input_norm, flow_condition_norm))
            # Convert back to physical units for observation model
            photometry = jnp.concatenate([jnp.array([Mg]), colors])
            numpyro.deterministic('photometry', photometry)
            
            # Convert to apparent magnitudes
            photometry=photometry.at[0].set(photometry[0]+5*jnp.log10(d)-5)  # Mg -> apparent G
            photometry=photometry.at[1:].set(photometry[1:]+photometry[0])  # colors -> magnitudes
            numpyro.deterministic('observed_photometry', photometry)
            
            # Apply extinction
            bprp=photometry[1]-photometry[2]  # BP-RP for extinction law
            extinction_coefficients=(extinction_law)(jnp.abs(ebv),bprp)
            photometry=photometry+extinction_coefficients*ebv
            parallax=1000.0/d #in mas

            numpyro.sample('obs_x', dist.MultivariateNormal(loc=photometry, covariance_matrix=jnp.maximum((photometry_covariance),1e-6)), obs=obs_photometry)
            numpyro.sample('obs_parallax', dist.Normal(loc=parallax, scale=(parallax_error)), obs=obs_parallax)
 


        
        self.model=model
        self.model_variational = model

