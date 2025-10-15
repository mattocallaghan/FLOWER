from ml_collections import ConfigDict
import ml_collections

def get_config():
    config = ConfigDict()

    # Global seed.
    config.seed = 42

    # Training settings.
    config.training = ConfigDict()
    config.training.batch_size = 16*2*2
    config.training.n_iters = 200           # Total number of training iterations.
    config.training.train=True   #T/F
    config.training.max_patience=80
    config.training.validation_split=0.1
    config.training.n_jitted_steps = 1        # Number of steps jitted together.
    config.training.log_freq = 20        # Log frequency (must be divisible by n_jitted_steps).
    config.training.snapshot_freq_for_preemption = 20
    config.training.eval_freq = 20
    config.training.snapshot_freq = 1000


    # Evaluation settings.
    config.eval = ConfigDict()
    config.eval.enable_loss = True
    config.eval.bpd_dataset = 'test'            # Use either 'train' or 'test'.
    config.eval.enable_bpd = False #likelihoods not ready yet
    config.eval.enable_sampling = True
    config.eval.num_samples = 50000
    config.eval.batch_size = 5000
    config.eval.begin_ckpt = 0                  # Starting checkpoint index.
    config.eval.end_ckpt = 1000              # Ending checkpoint index.

    # Data settings.
    config.data = ConfigDict()
    config.data.data_path = './Data'            # Path to store/load custom dataset.

    # Model settings.
    config.model = ConfigDict()
    config.model.name = 'bnaf'                    # Options: 've' or 'nf'.
    config.model.activation='arctan'
    # Optimizer settings.
    config.optim = ConfigDict()
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-4
    config.optim.warmup = 1000
    config.optim.grad_clip = 1.0

    # Sampling settings.
    config.sampling = ConfigDict()
    config.sampling.inference=False
    config.sampling.method = 'nuts'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.step_size_initial=0.1
    config.sampling.warmup_steps=0
    config.sampling.samples=1000
    config.sampling.adapt_step_size=False
    config.sampling.adapt_mass_matrix=False
    config.sampling.num_chains=1
    config.sampling.covariance_scale=1.0
    config.sampling.inference_method='synthetic_inference_bnaf'
    config.sampling.max_tree_depth=10
    config.sampling.num_warmup_in_loop=400
    config.sampling.num_samples_in_loop=100
    config.sampling.num_loops=3
    config.sampling.num_steps_train_loss_model=30
    config.sampling.lr_train_loss_model=1e-2
    return config



"""
The snr (signal-to-noise ratio) parameter of LangevinCorrector somewhat behaves like a temperature parameter. 
Larger snr typically results in smoother samples, while smaller snr gives more diverse but lower quality samples.
 Typical values of snr is 0.05 - 0.2, and it requires tuning to strike the sweet spot.
For VE SDEs, we recommend choosing config.model.sigma_max to be
the maximum pairwise distance between data samples in the training dataset.
"""