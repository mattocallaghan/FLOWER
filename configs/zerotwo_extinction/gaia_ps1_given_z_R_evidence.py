
from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
Evidence maximization configuration for p(colours,mag|Z,R) using marginal likelihood
with distance prior and importance sampling.
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    config.training.workspace='output/02_extinction_g_ps1_evidence'

    ##### training/sampling parameters 
    config.training.train=False 
    config.sampling.inference=not config.training.train
    #################################
    config.data.num_tests=30
    config.sampling.sampling_name='nlpe'
    config.sampling.inference_method='flower'
    config.data.num_simulations = 0     #not sbi
    config.data.inference_simulations = 0     #not sbi

    ### model parameters for gaia_ps1 data
    config.model.name = 'nlpe_bnaf_elu'
    config.model.nn_depth_bnaf = 3
    config.model.activation='tanh'
    config.model.flow_dimension=8  # M_G + 7 colors: M_G, BP-G, RP-G, g_PS1-G_Gaia, r_PS1-G_Gaia, i_PS1-G_Gaia, z_PS1-G_Gaia, y_PS1-G_Gaia
    config.model.nn_block_dim=4
    config.model.cond_dim=2  # log(|Z|), log(R)
    config.data.vector_dim=8  # M_G + colors (in physical units)
    config.data.vector_dim_inference=8
    config.data.summary_dim=0  # no summary statistics needed

    config.data.dataset='gaia_ps1'
    config.data.benchmark = False  # We're not using SBIBM data
    config.data.save_csv=False
    config.data.save_path_gaia_data = "Data/gaia_ps1_data_018_evidence.csv"  #if saving then the location
    config.data.inference_dataset='inference_val_gaia_ps1_018'

    #'l','b','parallax']+['Z','R'] + list(filters)
    
    #synthetic data parameters
    config.data.lenz_map_path = '/Users/mattocallaghan/XPNorm/Data/ebv_lhd.hpx.fits'
    config.data.url="http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"
    config.training.validation_split=0.1

    ### Evidence loss configuration
    config.loss = ConfigDict()
    config.loss.use_evidence_max = True
    config.loss.L = 1300.0                    # Distance scale parameter for prior (1.3 kpc)
    config.loss.alpha = 1.5                   # Distance prior exponent (steeper falloff)
    config.loss.beta = 0.8                    # Distance prior coefficient (less volume weighting)
    config.loss.n_distance_samples = 32       # Number of distance samples for importance sampling
    
    ### Contrastive learning for extinction detection
    config.loss.use_contrastive = False        # Enable contrastive learning for zero-extinction star detection
    config.loss.contrastive_weight = 0.3    # Weight for contrastive loss term  
    config.loss.contrastive_samples = 3       # Number of reddened samples per clean star
    config.loss.contrastive_clamp_bound = -70.0  # Clamp bound to prevent CNF mode collapse (literature recommendation)

    #### training
    config.training.batch_size =256*2*2*2 # Moderate batch size for stable training
    config.training.n_iters = 60          # Total number of training iterations.
    config.optim.lr = 1e-2
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-4
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0
    config.training.max_patience=50

    #### Inference
    ## tuning paramaerers
    config.sampling.method = 'nuts'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.step_size_initial=1
    config.sampling.warmup_steps=750
    config.sampling.samples=4000
    config.sampling.adapt_step_size=True
    config.sampling.adapt_mass_matrix=True
    config.sampling.num_chains=1
    config.sampling.max_tree_depth=10
    config.sampling.acceptance_prob=0.9
    config.sampling.covariance_scale=1**2
    config.sampling.max_tree_depth=10
    config.sampling.bands=['G','BP','RP','g','r','i','z','y']  ## what bands to actually do the inference over
      # Total number of training iterations.


    ####sampling, likelihood error values
    
    
    #######
    ### galactic parameters as per 
    ### exponential model from
    ### green et al
    #######
    config.physical_parameters = ConfigDict()
    config.physical_parameters.R_sun=8000.0
    config.physical_parameters.Z_sun = 0.025     # kpc (Height of Sun above Galactic plane)

    
    return config