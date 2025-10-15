# FLOWER: Flow-based Learning Of stellar Wavelength-dependent Extinction Representation

FLOWER is a normalizing flow-based framework for modeling stellar photometry and extinction using evidence maximization. It uses Gaia and Pan-STARRS1 (PS1) photometric data to learn the intrinsic stellar color-magnitude distribution as a function of galactocentric position, marginalizing over distance uncertainties through importance sampling.

## Overview

FLOWER learns the conditional distribution **p(M_G, colors | log|Z|, log R)** where:
- **M_G**: Absolute G magnitude 
- **colors**: 7 colors relative to G band (BP-G, RP-G, g_PS1-G, r_PS1-G, i_PS1-G, z_PS1-G, y_PS1-G)
- **|Z|**:  galactocentric height 
- ** R **: Log galactocentric radius

The framework supports optional 2MASS infrared data (J, H, Ks bands) for enhanced extinction modeling.

## Key Features

### Evidence Maximization Loss Function (`losses.py:60-442`)

The core innovation is the **EvidenceMaxLoss** class that implements evidence maximization using importance sampling to marginalize over distance uncertainties:

```python
class EvidenceMaxLoss:
    """
    Evidence maximization loss using importance sampling for marginal likelihood.
    
    Main loss for FLOWER:
    
    todo: implement selection functions in more complex manner, currently we make 
    selection function cuts so that the selection function is approximately 1 everywhere on the sky
    
    Trains p(colours,mag|Z,R) by marginalizing over distance with a distance prior
    and importance sampling correction. Includes contrastive learning for extinction.
    """
```



### Contrastive Learning for Extinction Detection

The loss includes optional contrastive learning to detect zero-extinction stars:

```python
config.loss.use_contrastive = True        # Currently enabled
config.loss.contrastive_weight = 0.3       # Weight for contrastive term
config.loss.contrastive_samples = 3        # Reddened samples per star
config.loss.contrastive_clamp_bound = -70.0 # Numerical stability bound
```

The contrastive term applies synthetic E(B-V) reddening (0.01-0.3 mag) and encourages the model to assign low probability to artificially reddened stars.

## Architecture

### Normalizing Flow Model (`normalizing_flow.py:624-700`)

FLOWER uses Block Neural Autoregressive Flow (BNAF) with:
- **Architecture**: `nlpe_bnaf_elu` 
- **Input dimension**: 8 (M_G + 7 colors) or 11 (with 2MASS)
- **Condition dimension**: 2 (log|Z|, log R)
- **Depth**: 3 layers
- **Block dimension**: 4
- **Activation**: tanh

### Data Pipeline (`datasets.py`, `stellar_data_utils.py`)

The framework processes Gaia DR3 + PS1 photometric catalogs:

1. **Coordinate transformation**: (l,b,d) � galactocentric (Z,R)
2. **Color computation**: All colors relative to G band 
3. **Normalization**: Z-score standardization of all features
4. **Distance sampling**: Gaussian proposal around geometric distance estimates

## Selection Functions

The data used in the paper was cut so that the cmpleteness is approximately uniform across the sky and that the main selection effects are due to Malmquist bias. Codebase needs to be updated to fully account for the selection effects if less cuts are made





## Configuration System

### Main Config (`configs/zerotwo_extinction/gaia_ps1_given_z_R_evidence.py`)

Key settings for evidence maximization:

```python
# Model architecture
config.model.flow_dimension = 8           # M_G + 7 colors
config.model.cond_dim = 2                 # log|Z|, log R
config.model.nn_depth_bnaf = 3            # BNAF depth
config.model.nn_block_dim = 4             # Block dimension

# Evidence loss parameters  
config.loss.use_evidence_max = True
config.loss.L = 1300.0                    # Distance scale (pc)
config.loss.alpha = 1.5                   # Prior exponent
config.loss.beta = 0.8                    # Prior coefficient
config.loss.n_distance_samples = 32       # Importance samples

# Training
config.training.batch_size = 2048         # Large batches for stability
config.training.n_iters = 60              # Training epochs
config.optim.lr = 1e-2                    # Learning rate
```

## Usage

### Training
```bash
python main_train_eval.py --config=configs/zerotwo_extinction/gaia_ps1_given_z_R_evidence.py --mode=train
```

### Inference  
```bash
python main_train_eval.py --config=configs/zerotwo_extinction/gaia_ps1_given_z_R_evidence.py --mode=inference
```

## Dependencies

Key packages (see `requirements.txt`):
- **JAX ecosystem**: `jax`, `equinox`, `optax`, `flowjax`
- **Astronomy**: `astropy`, `healpy`, `astroquery`
- **ML/Stats**: `tensorflow-probability`, `numpyro`, `blackjax`
- **Data**: `pandas`, `numpy`, `h5py`




