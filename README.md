# FLOWER: 

Implementation of FLOW based Estimation of Reddening (FLOWER) as outlined in O'Callaghan et al 2025.

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
    see https://github.com/gaia-unlimited/gaiaunlimited for selection function implementations
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

## Data Structure
Data should all appear in the Data folder and should match the naming convention outlined in datasets.py and the configs.


FLOWER processes raw Gaia DR3 + Pan-STARRS1 (PS1) observational data with optional 2MASS infrared photometry. **All coordinate transformations, color calculations, and absolute magnitude computations happen during training within the evidence maximization loss function.**

Currently the data does not include the selection function properties other than the distance-dependent effects as the cuts were chosen to make the selection function approximately uniform across the sky. 

FLOWER can be very easily changed to add more colours and generalise to higher dimensions. Future changes will adjust the data and pipeline to reflect an aribitrary number of colours.

### Input Data Columns (Raw Observational Data)

The input data consists of 40 columns (or 46 with 2MASS) of **raw observational measurements**:

#### Standard Mode (40 columns)
The input data structure contains:

```python
# Raw observational data array (indices 0-39):
[source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
 phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, 
 phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error,
 phot_rp_mean_flux, phot_rp_mean_flux_error, ruwe, r_med_geo, r_lo_geo, r_hi_geo,
 g_mean_psf_mag, g_mean_psf_mag_error, r_mean_psf_mag, r_mean_psf_mag_error,
 i_mean_psf_mag, i_mean_psf_mag_error, z_mean_psf_mag, z_mean_psf_mag_error,
 y_mean_psf_mag, y_mean_psf_mag_error, l, b]
```

**Key raw data columns:**
- **Indices 3-4**: `parallax`, `parallax_error` (mas) - Gaia astrometry
- **Indices 9-11**: `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag` - **Apparent** Gaia magnitudes
- **Indices 13,15,17**: Gaia flux errors (G, BP, RP)  
- **Indices 19-21**: `r_med_geo`, `r_lo_geo`, `r_hi_geo` (pc) - Geometric distance estimates
- **Indices 22-31**: PS1 **apparent** magnitudes and errors (g, r, i, z, y bands)
- **Indices 38-39**: `l`, `b` (degrees) - Galactic coordinates

#### 2MASS Extended Mode (46 columns)
When `config.data.twomass = True`, additional infrared data:

```python
# Additional 2MASS columns (indices 32-37, then l,b at 38-39):
[... y_mean_psf_mag, y_mean_psf_mag_error, 
 j_m, j_msigcom, h_m, h_msigcom, ks_m, ks_msigcom, l, b]
```

- **Indices 32,34,36**: `j_m`, `h_m`, `ks_m` - **Apparent** 2MASS magnitudes
- **Indices 33,35,37**: `j_msigcom`, `h_msigcom`, `ks_msigcom` - 2MASS uncertainties

### Processing During Training

**All feature engineering happens within the evidence loss calculation** (`losses.py:283-409`):

#### 1. Color Computation (lines 283-309)
Colors are computed **during loss evaluation** from apparent magnitudes:
```python
# Colors computed on-the-fly during training:
bp_g_obs = phot_bp_mean_mag - phot_g_mean_mag      # BP-G color
rp_g_obs = phot_rp_mean_mag - phot_g_mean_mag      # RP-G color  
g_ps1_g_gaia_obs = g_mean_psf_mag - phot_g_mean_mag # g_PS1-G color
# ... and so on for all 7 (or 10 with 2MASS) colors
```

#### 2. Distance Sampling & Coordinate Transformation (lines 347-367)
For each training sample, the loss function:
```python
# Sample multiple distances for importance sampling
distance_samples = sample_around_geometric_distance(r_med_geo, distance_error)

# For each distance sample, convert to galactocentric coordinates:
x_gc = d * cos(b_rad) * cos(l_rad)
y_gc = d * cos(b_rad) * sin(l_rad)  
z_gc = d * sin(b_rad)

R = sqrt((R_sun - x_gc)^2 + y_gc^2)  # galactocentric radius
Z = abs(z_gc - Z_sun)                 # height above plane


```

#### 3. Absolute Magnitude Calculation (lines 371-373)
```python
# Compute absolute magnitude from apparent mag + distance modulus
mu = 5*log10(d) - 5           # distance modulus  
M_G_sample = phot_g_mean_mag - mu + noise  # absolute G magnitude
```

### Flow Training Features

The normalizing flow is trained on **processed features** computed during loss evaluation:

#### Standard Mode (8 dimensions)
- **Flow inputs**: `[M_G, BP-G, RP-G, g_PS1-G, r_PS1-G, i_PS1-G, z_PS1-G, y_PS1-G]`
- **Flow conditions**: `[log_Z, log_R]`

#### 2MASS Mode (11 dimensions)  
- **Flow inputs**: `[M_G, BP-G, RP-G, g_PS1-G, r_PS1-G, i_PS1-G, z_PS1-G, y_PS1-G, J-G, H-G, Ks-G]`
- **Flow conditions**: `[log_Z, log_R]`

**Key insight**: The flow learns `p(M_G, colors | log_Z, log_R)` where all quantities are **derived** from the raw observational data through the evidence maximization procedure.

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
See https://github.com/gaia-unlimited/gaiaunlimited.



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




