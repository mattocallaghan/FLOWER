
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
from jax import vmap
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, PRNGKeyArray
from flowjax.distributions import AbstractDistribution
import jax
from extinction_law import get_extinction_function
class MaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood).

    This loss can be used to learn either conditional or unconditional distributions.
    """

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute the loss. Key is ignored (for consistency of API)."""
        dist = paramax.unwrap(eqx.combine(params, static))
        log_p = dist.log_prob(x, condition)  # shape (batch,)

        # Standard negative log-likelihood loss
        nll = -log_p.mean()

        lambda_gp=1.0
        # Add gradient penalty if enabled
        if lambda_gp > 0.0 and condition is not None:

            # Define per-sample function to get log p(x|theta) for autodiff
            def per_sample_logp(th, xi):
                return dist.log_prob(xi[None, :], th[None, :])[0]

            # Vectorize grad computation over the batch
            grad_fn = jax.vmap(jax.grad(per_sample_logp), in_axes=(0, 0))
            grad_logp_wrt_condition = grad_fn(condition, x)  # shape (batch, cond_dim)

            # Compute squared L2 norm of gradients
            grad_norm2 = jnp.sum(grad_logp_wrt_condition ** 2, axis=-1)  # shape (batch,)

            # Mean penalty
            grad_penalty = grad_norm2.mean()

            # Add gradient penalty to the loss
            nll = nll + lambda_gp * grad_penalty

        return nll




class EvidenceMaxLoss:
    """
    Evidence maximization loss using importance sampling for marginal likelihood.
    
    Main loss for FLOWER:

    todo: implement selection functions in more complex manner, currently we make 
    selection function cuts so that the selection function is approximately 1 everywhere on the sky

    Trains p(colours,mag|Z,R) by marginalizing over distance with a distance prior
    and importance sampling correction. Includes contrastive learning for extinction.
    """
    
    def __init__(self, 
                 L: float = 1000.0,
                 alpha: float = 2.0, 
                 beta: float = 1.0,
                 n_distance_samples: int = 50,
                 use_contrastive: bool = False,
                 contrastive_weight: float = 1.0,
                 contrastive_samples: int = 5,
                 contrastive_clamp_bound: float = -50.0,
                 twomass: bool = False):
        """
        Args:
            L: Distance scale parameter for prior
            alpha: Distance prior exponent
            beta: Distance prior coefficient  
            n_distance_samples: Number of distance samples for importance sampling
            use_contrastive: Whether to use contrastive learning for extinction
            contrastive_weight: Weight for contrastive loss term
            contrastive_samples: Number of reddened samples per clean star
            contrastive_clamp_bound: Lower bound for clamping to prevent CNF mode collapse
        """
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.n_distance_samples = n_distance_samples
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_samples = contrastive_samples
        self.contrastive_clamp_bound = contrastive_clamp_bound
        self.twomass = twomass
        
        # Setup extinction function for relevant bands if contrastive learning is enabled
        if self.use_contrastive:
            # Default bands: Gaia (G, BP, RP) + PS1 (g, r, i, z, y)
            # Will be updated dynamically based on config in __call__ method
            bands = ["G", "BP", "RP", "g", "r", "i", "z", "y"]
            self.extinction_fn = get_extinction_function(bands, mean_val=False, teff=False)
    
    @eqx.filter_jit 
    def __call__(self,
                 params: AbstractDistribution,
                 static: AbstractDistribution,
                 x: Array,  # [batch, features] - normalized gaia_ps1 data
                 condition: Array | None = None,  # Should be None for this loss
                 key: PRNGKeyArray | None = None,
                 mean: Array | None = None,
                 std: Array | None = None,
                 train_stats_mean: Array | None = None,
                 train_stats_std: Array | None = None) -> Float[Array, ""]:
        """
        Compute evidence maximization loss for gaia_ps1 data.
        
        Everything happens in physical units in the loss function.
        The flow will be trained to work directly on physical units.
        
        Args:
            x: Training batch - normalized gaia_ps1 data (will be denormalized immediately)
            mean, std: Normalization statistics for denormalization to physical units
        """
        if key is None:
            key = jr.PRNGKey(0)
        
        if mean is None or std is None:
            raise ValueError("mean and std must be provided for gaia_ps1 evidence loss")
            
        dist_flow = paramax.unwrap(eqx.combine(params, static))
        batch_size = x.shape[0]
        
        # Convert everything to physical units immediately
        x_physical = x * std + mean
        
        # Extract observables from physical data
        # Data structure depends on whether 2MASS data is included
        # Original: [source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
        #           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, 
        #           phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error,
        #           phot_rp_mean_flux, phot_rp_mean_flux_error, ruwe, r_med_geo, r_lo_geo, r_hi_geo,
        #           g_mean_psf_mag, g_mean_psf_mag_error, r_mean_psf_mag, r_mean_psf_mag_error,
        #           i_mean_psf_mag, i_mean_psf_mag_error, z_mean_psf_mag, z_mean_psf_mag_error,
        #           y_mean_psf_mag, y_mean_psf_mag_error, l, b]
        # With 2MASS: [... y_mean_psf_mag, y_mean_psf_mag_error, j_m, j_msigcom, h_m, h_msigcom, ks_m, ks_msigcom, l, b]
        
        l = x_physical[..., 38]  # galactic longitude
        b = x_physical[..., 39]  # galactic latitude
        r_med_geo = x_physical[..., 19]  # median geometric distance
        r_lo_geo = x_physical[..., 20]   # lower bound geometric distance  
        r_hi_geo = x_physical[..., 21]   # upper bound geometric distance
        
        # Gaia photometry
        phot_g_mean_mag = x_physical[..., 9]
        phot_bp_mean_mag = x_physical[..., 10]
        phot_rp_mean_mag = x_physical[..., 11]
        phot_g_mean_flux_error = x_physical[..., 13]
        phot_bp_mean_flux_error = x_physical[..., 15]
        phot_rp_mean_flux_error = x_physical[..., 17]
        
        # PS1 photometry
        g_mean_psf_mag = x_physical[..., 22]
        g_mean_psf_mag_error = x_physical[..., 23]
        r_mean_psf_mag = x_physical[..., 24]
        r_mean_psf_mag_error = x_physical[..., 25]
        i_mean_psf_mag = x_physical[..., 26]
        i_mean_psf_mag_error = x_physical[..., 27]
        z_mean_psf_mag = x_physical[..., 28]
        z_mean_psf_mag_error = x_physical[..., 29]
        y_mean_psf_mag = x_physical[..., 30]
        y_mean_psf_mag_error = x_physical[..., 31]
        
        # Compute distance proposal statistics
        distance_lower = jnp.abs(r_med_geo - r_lo_geo) * 0.5
        distance_upper = jnp.abs(r_hi_geo - r_med_geo) * 0.5
        distance_error = distance_lower + distance_upper
        
        # Check if 2MASS data is enabled
        has_twomass = self.twomass
        
        # Compute log evidence for each sample in batch
        keys = jr.split(key, batch_size)
        evidence_fn = lambda xi, ki: self._compute_log_evidence_single_gaia(
            xi, ki, dist_flow, mean, std, train_stats_mean, train_stats_std, has_twomass
        )
        log_evidences = vmap(evidence_fn)(x_physical, keys)
        
        # Main evidence loss (negative log evidence for minimization)
        evidence_loss = -log_evidences.mean()
        
        # Add contrastive learning term if enabled
        if self.use_contrastive:
            key, subkey = jr.split(key)
            contrastive_loss = self._compute_contrastive_loss(
                x_physical, subkey, dist_flow, train_stats_mean, train_stats_std
            )
            total_loss = evidence_loss + self.contrastive_weight * contrastive_loss
            
            # Store components for debugging (convert to Python floats to avoid JAX tracer issues)
            try:
                self._last_evidence_loss = float(evidence_loss)
                self._last_contrastive_loss = float(contrastive_loss)
                self._last_total_loss = float(total_loss)
            except:
                # Skip storage if JAX tracers can't be converted
                pass
            return total_loss
        else:
            return evidence_loss
    
    def _sample_distances_gaia(self, key, r_med_geo, distance_error, batch_size):
        """Sample distances from Gaia geometric distance proposal N(d|r_med_geo, σ_d)."""
        keys = jr.split(key, batch_size)
        sample_fn = lambda k, d_mean, d_std: jr.normal(
            k, (self.n_distance_samples,)
        ) * d_std + d_mean
        return vmap(sample_fn)(keys, r_med_geo, distance_error)

    def _sample_distances(self, key, distance_obs, batch_size):
        """Sample distances around observed estimates for importance sampling."""
        # Sample distances in a reasonable range around observed values
        distance_min = jnp.maximum(distance_obs * 0.5, 10.0)  # At least 10 pc
        distance_max = jnp.minimum(distance_obs * 2.0, 5000.0)  # At most 5 kpc
        
        keys = jr.split(key, batch_size)
        sample_fn = lambda k, d_min, d_max: jr.uniform(
            k, (self.n_distance_samples,), minval=d_min, maxval=d_max
        )
        return vmap(sample_fn)(keys, distance_min, distance_max)
    
    def _compute_log_evidence_single_gaia(self, x_single, key, 
                                        dist_flow, mean, std, train_stats_mean=None, train_stats_std=None, has_twomass=False):
        """Compute log evidence for a single gaia_ps1 (or gaia_ps1_2mass) data point with proper importance sampling."""
        
        # Extract data from physical units
        l = x_single[38]  # galactic longitude
        b = x_single[39]  # galactic latitude
        parallax_obs = x_single[3]  # observed parallax (mas)
        parallax_error = x_single[4]  # parallax error (mas)
        r_med_geo = x_single[19]  # median geometric distance
        r_lo_geo = x_single[20]   # lower bound geometric distance  
        r_hi_geo = x_single[21]   # upper bound geometric distance
        
        # Gaia photometry
        phot_g_mean_mag = x_single[9]
        phot_bp_mean_mag = x_single[10]
        phot_rp_mean_mag = x_single[11]
        phot_g_mean_flux_error = x_single[13]
        phot_bp_mean_flux_error = x_single[15]
        phot_rp_mean_flux_error = x_single[17]
        
        # PS1 photometry
        g_mean_psf_mag = x_single[22]
        g_mean_psf_mag_error = x_single[23]
        r_mean_psf_mag = x_single[24]
        r_mean_psf_mag_error = x_single[25]
        i_mean_psf_mag = x_single[26]
        i_mean_psf_mag_error = x_single[27]
        z_mean_psf_mag = x_single[28]
        z_mean_psf_mag_error = x_single[29]
        y_mean_psf_mag = x_single[30]
        y_mean_psf_mag_error = x_single[31]
        
        # Compute distance proposal statistics
        distance_lower = jnp.abs(r_med_geo - r_lo_geo) * 0.5
        distance_upper = jnp.abs(r_hi_geo - r_med_geo) * 0.5
        distance_error = distance_lower + distance_upper
        
        # Convert flux errors to magnitude errors (approximate)
        g_mag_error = 2.5 / jnp.log(10) * phot_g_mean_flux_error / jnp.maximum(phot_g_mean_mag, 1e-6)
        bp_mag_error = 2.5 / jnp.log(10) * phot_bp_mean_flux_error / jnp.maximum(phot_bp_mean_mag, 1e-6)
        rp_mag_error = 2.5 / jnp.log(10) * phot_rp_mean_flux_error / jnp.maximum(phot_rp_mean_mag, 1e-6)
        
        # Prepare observed colors and errors for vectorized computation
        bp_g_obs = phot_bp_mean_mag - phot_g_mean_mag
        rp_g_obs = phot_rp_mean_mag - phot_g_mean_mag
        g_ps1_g_gaia_obs = g_mean_psf_mag - phot_g_mean_mag
        r_ps1_g_gaia_obs = r_mean_psf_mag - phot_g_mean_mag
        i_ps1_g_gaia_obs = i_mean_psf_mag - phot_g_mean_mag
        z_ps1_g_gaia_obs = z_mean_psf_mag - phot_g_mean_mag
        y_ps1_g_gaia_obs = y_mean_psf_mag - phot_g_mean_mag
        
        if has_twomass:
            # Extract 2MASS magnitudes
            j_m = x_single[32]  # j_m
            h_m = x_single[34]  # h_m
            ks_m = x_single[36] # ks_m
            
            # Calculate 2MASS colors relative to G
            j_g_gaia_obs = j_m - phot_g_mean_mag
            h_g_gaia_obs = h_m - phot_g_mean_mag
            ks_g_gaia_obs = ks_m - phot_g_mean_mag
            
            colors_obs = jnp.array([bp_g_obs, rp_g_obs, g_ps1_g_gaia_obs, 
                                   r_ps1_g_gaia_obs, i_ps1_g_gaia_obs, 
                                   z_ps1_g_gaia_obs, y_ps1_g_gaia_obs,
                                   j_g_gaia_obs, h_g_gaia_obs, ks_g_gaia_obs])
        else:
            colors_obs = jnp.array([bp_g_obs, rp_g_obs, g_ps1_g_gaia_obs, 
                                   r_ps1_g_gaia_obs, i_ps1_g_gaia_obs, 
                                   z_ps1_g_gaia_obs, y_ps1_g_gaia_obs])
        
        # Color errors in quadrature
        bp_g_error = jnp.sqrt(bp_mag_error**2 + g_mag_error**2)
        rp_g_error = jnp.sqrt(rp_mag_error**2 + g_mag_error**2)
        g_ps1_g_gaia_error = jnp.sqrt(g_mean_psf_mag_error**2 + g_mag_error**2)
        r_ps1_g_gaia_error = jnp.sqrt(r_mean_psf_mag_error**2 + g_mag_error**2)
        i_ps1_g_gaia_error = jnp.sqrt(i_mean_psf_mag_error**2 + g_mag_error**2)
        z_ps1_g_gaia_error = jnp.sqrt(z_mean_psf_mag_error**2 + g_mag_error**2)
        y_ps1_g_gaia_error = jnp.sqrt(y_mean_psf_mag_error**2 + g_mag_error**2)
        
        if has_twomass:
            # Extract 2MASS magnitude errors
            j_msigcom = x_single[33]  # j magnitude error
            h_msigcom = x_single[35]  # h magnitude error
            ks_msigcom = x_single[37] # ks magnitude error
            
            # Calculate 2MASS color errors in quadrature with G
            j_g_gaia_error = jnp.sqrt(j_msigcom**2 + g_mag_error**2)
            h_g_gaia_error = jnp.sqrt(h_msigcom**2 + g_mag_error**2)
            ks_g_gaia_error = jnp.sqrt(ks_msigcom**2 + g_mag_error**2)
            
            color_errors = jnp.array([bp_g_error, rp_g_error, g_ps1_g_gaia_error,
                                     r_ps1_g_gaia_error, i_ps1_g_gaia_error,
                                     z_ps1_g_gaia_error, y_ps1_g_gaia_error,
                                     j_g_gaia_error, h_g_gaia_error, ks_g_gaia_error])
        else:
            color_errors = jnp.array([bp_g_error, rp_g_error, g_ps1_g_gaia_error,
                                     r_ps1_g_gaia_error, i_ps1_g_gaia_error,
                                     z_ps1_g_gaia_error, y_ps1_g_gaia_error])
        
        # Generate distance samples for this single data point with bounds
        key, subkey = jr.split(key)
        raw_samples = jr.normal(subkey, (self.n_distance_samples,)) * distance_error + r_med_geo
        # Clamp distances to reasonable range: [10 pc, 50000 pc]
        distance_samples = jnp.clip(raw_samples, 1.0, 50000.0)
        
        # Vectorized computation for all distance samples
        def process_distance_sample(d, k):
            # Convert l, b, distance to galactocentric Z, R (in physical units)
            R_sun = 8000.0  # pc
            Z_sun = 25.0    # pc
            
            # Convert to radians
            l_rad = jnp.radians(l)
            b_rad = jnp.radians(b)
            
            # Calculate galactocentric coordinates
            x_gc = d * jnp.cos(b_rad) * jnp.cos(l_rad)
            y_gc = d * jnp.cos(b_rad) * jnp.sin(l_rad)
            z_gc = d * jnp.sin(b_rad)
            
            # Convert to galactocentric coordinates
            R = jnp.sqrt((R_sun - x_gc)**2 + y_gc**2)
            Z = jnp.abs(z_gc - Z_sun)  # |Z| as specified
            
            # Convert Z, R to log scale for normalizing flow conditioning (physical units)
            log_Z = jnp.log10(Z + 1e-6)  # Add small epsilon to avoid log(0)
            log_R = jnp.log10(R + 1e-6)
            
            # Sample absolute magnitude M_G from apparent mag corrected by distance modulus
            k1, k2 = jr.split(k)
            mu = 5*jnp.log10(d) - 5  # distance modulus
            M_G_mean = phot_g_mean_mag - mu  # absolute magnitude proposal mean
            M_G_sample = M_G_mean + jr.normal(k1) * g_mag_error  # sample with noise
            
            # Sample colors with measurement errors
            n_colors = colors_obs.shape[0]  # Get actual number of colors (7 or 10)
            color_noise = jr.normal(k2, (n_colors,))
            colors_sample = colors_obs + color_noise * color_errors
            
            # Flow input: [M_G, colors] - normalize using train_stats if available
            flow_input_physical = jnp.concatenate([jnp.array([M_G_sample]), colors_sample])
            
            # Flow condition: [log_Z, log_R] - normalize using train_stats if available
            flow_condition_physical = jnp.array([log_Z, log_R])
            
            # Normalize flow inputs and conditions if train_stats are available
            if has_twomass:
                # train_stats structure for 2MASS: [10 colors, M_G, log_R, log_Z] = indices [0:10, 10, 11, 12]
                # Flow inputs: [M_G, 10 colors] from indices [10, 0:10]
                flow_input_mean = jnp.concatenate([train_stats_mean[10:11], train_stats_mean[0:10]])
                flow_input_std = jnp.concatenate([train_stats_std[10:11], train_stats_std[0:10]]) 
                flow_input = (flow_input_physical - flow_input_mean) / flow_input_std
                
                # Flow conditions: [log_Z, log_R] from indices [12, 11] 
                flow_condition_mean = jnp.array([train_stats_mean[12], train_stats_mean[11]])  # [log_Z, log_R]
                flow_condition_std = jnp.array([train_stats_std[12], train_stats_std[11]])
                flow_condition = (flow_condition_physical - flow_condition_mean) / flow_condition_std
            else:
                # train_stats structure: [7 colors, M_G, log_R, log_Z] = indices [0:7, 7, 8, 9]
                # Flow inputs: [M_G, 7 colors] from indices [7, 0:7]
                flow_input_mean = jnp.concatenate([train_stats_mean[7:8], train_stats_mean[0:7]])
                flow_input_std = jnp.concatenate([train_stats_std[7:8], train_stats_std[0:7]]) 
                flow_input = (flow_input_physical - flow_input_mean) / flow_input_std
                
                # Flow conditions: [log_Z, log_R] from indices [9, 8] 
                flow_condition_mean = jnp.array([train_stats_mean[9], train_stats_mean[8]])  # [log_Z, log_R]
                flow_condition_std = jnp.array([train_stats_std[9], train_stats_std[8]])
                flow_condition = (flow_condition_physical - flow_condition_mean) / flow_condition_std

            
            # Evaluate flow log probability (normalized inputs)
            log_p_flow = dist_flow.log_prob(flow_input, flow_condition)
            
            # Parallax likelihood: p(parallax_obs | d) - doesn't cancel with any proposal
            parallax_pred = 1000.0 / d  # predicted parallax in mas from distance in pc
            log_likelihood_parallax = -0.5 * ((parallax_obs - parallax_pred) / parallax_error)**2 - jnp.log(parallax_error * jnp.sqrt(2 * jnp.pi))
            
            # Distance prior: p(d) ∝ d^β * exp(-(d/L)^α)
            log_prior_d = self.beta * jnp.log(d) - (d/self.L)**self.alpha
            
            # Distance proposal: N(d|r_med_geo, σ_d)
            log_proposal_d = -0.5 * ((d - r_med_geo) / distance_error)**2 - jnp.log(distance_error * jnp.sqrt(2 * jnp.pi))
            
            # Importance weight: Only distance terms (magnitude and color proposals cancel with their data likelihoods)
            # w = [p(d) × p(parallax_obs|d)] / q(d)
            # Note: p(m_G_obs|M_G,d) cancels with q(M_G|d) and p(colors_obs|colors) cancels with q(colors)
            log_weight = log_prior_d + log_likelihood_parallax - log_proposal_d
            
            return log_weight, log_p_flow
        
        # Generate keys for each distance sample
        keys = jr.split(key, self.n_distance_samples)
        
        # Vectorized computation across all distance samples
        log_weights, log_likelihoods = vmap(process_distance_sample)(distance_samples, keys) 
        
        # Compute log evidence using logsumexp for numerical stability
        # log evidence = log(1/N) + logsumexp(log_weight + log_p_flow)
        # This computes: (1/N) * ∑ᵢ wᵢ × p(M_Gᵢ, colorsᵢ | log|Z|ᵢ, log Rᵢ)
        log_evidence = logsumexp(log_weights + log_likelihoods) - jnp.log(self.n_distance_samples)
        
        return log_evidence
    
    def _apply_reddening_to_colors(self, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                                   g_mean_psf_mag, r_mean_psf_mag, i_mean_psf_mag, 
                                   z_mean_psf_mag, y_mean_psf_mag, ebv, bp_rp_color,
                                   j_m=None, h_m=None, ks_m=None):
        """
        Apply reddening to individual magnitudes and convert back to colors.
        
        Args:
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag: Gaia magnitudes
            g_mean_psf_mag, r_mean_psf_mag, i_mean_psf_mag, z_mean_psf_mag, y_mean_psf_mag: PS1 magnitudes
            j_m, h_m, ks_m: 2MASS magnitudes (optional)
            ebv: E(B-V) extinction value
            bp_rp_color: BP-RP color for stellar effective temperature estimation
            
        Returns:
            colors_reddened: [BP-G, RP-G, g_PS1-G, r_PS1-G, i_PS1-G, z_PS1-G, y_PS1-G, (J-G, H-G, Ks-G)] after reddening
        """
        # Determine if 2MASS data is provided
        has_twomass = j_m is not None and h_m is not None and ks_m is not None
        
        if has_twomass:
            # Update extinction function for 2MASS bands
            bands = ["G", "BP", "RP", "g", "r", "i", "z", "y", "J", "H", "Ks"]
            extinction_fn = get_extinction_function(bands, mean_val=False, teff=False)
            A_values = extinction_fn(ebv, bp_rp_color)  # [A_G, A_BP, A_RP, A_g, A_r, A_i, A_z, A_y, A_J, A_H, A_Ks]
        else:
            # Use existing extinction function for PS1+Gaia only
            A_values = self.extinction_fn(ebv, bp_rp_color)  # [A_G, A_BP, A_RP, A_g, A_r, A_i, A_z, A_y]
        
        # Apply extinction to individual magnitudes
        # Extinction makes stars fainter (adds to magnitude)
        phot_g_reddened = phot_g_mean_mag + A_values[0]      # A_G
        phot_bp_reddened = phot_bp_mean_mag + A_values[1]    # A_BP  
        phot_rp_reddened = phot_rp_mean_mag + A_values[2]    # A_RP
        g_ps1_reddened = g_mean_psf_mag + A_values[3]        # A_g
        r_ps1_reddened = r_mean_psf_mag + A_values[4]        # A_r
        i_ps1_reddened = i_mean_psf_mag + A_values[5]        # A_i
        z_ps1_reddened = z_mean_psf_mag + A_values[6]        # A_z
        y_ps1_reddened = y_mean_psf_mag + A_values[7]        # A_y
        
        # Convert back to colors relative to G band (as expected by flow)
        bp_g_reddened = phot_bp_reddened - phot_g_reddened
        rp_g_reddened = phot_rp_reddened - phot_g_reddened
        g_ps1_g_gaia_reddened = g_ps1_reddened - phot_g_reddened
        r_ps1_g_gaia_reddened = r_ps1_reddened - phot_g_reddened
        i_ps1_g_gaia_reddened = i_ps1_reddened - phot_g_reddened
        z_ps1_g_gaia_reddened = z_ps1_reddened - phot_g_reddened
        y_ps1_g_gaia_reddened = y_ps1_reddened - phot_g_reddened
        
        if has_twomass:
            # Apply extinction to 2MASS magnitudes and convert to colors
            j_reddened = j_m + A_values[8]                   # A_J
            h_reddened = h_m + A_values[9]                   # A_H
            ks_reddened = ks_m + A_values[10]                # A_Ks
            
            j_g_gaia_reddened = j_reddened - phot_g_reddened
            h_g_gaia_reddened = h_reddened - phot_g_reddened
            ks_g_gaia_reddened = ks_reddened - phot_g_reddened
            
            colors_reddened = jnp.array([bp_g_reddened, rp_g_reddened, g_ps1_g_gaia_reddened,
                                        r_ps1_g_gaia_reddened, i_ps1_g_gaia_reddened,
                                        z_ps1_g_gaia_reddened, y_ps1_g_gaia_reddened,
                                        j_g_gaia_reddened, h_g_gaia_reddened, ks_g_gaia_reddened])
        else:
            colors_reddened = jnp.array([bp_g_reddened, rp_g_reddened, g_ps1_g_gaia_reddened,
                                        r_ps1_g_gaia_reddened, i_ps1_g_gaia_reddened,
                                        z_ps1_g_gaia_reddened, y_ps1_g_gaia_reddened])
        
        return colors_reddened
    
    def _compute_contrastive_loss(self, x_physical, key, dist_flow, train_stats_mean, train_stats_std):
        """
        Compute contrastive loss for extinction detection.
        Simple version: sample one E(B-V) per star and apply reddening.
        """
        batch_size = x_physical.shape[0]
        
        # Check if 2MASS data is enabled
        has_twomass = self.twomass
        
        # Sample single E(B-V) for each star
        ebv_samples = jr.uniform(key, (batch_size,), minval=0.01, maxval=0.3)
        
        # Extract basic data (vectorized)
        phot_g_mean_mag = x_physical[:, 9]
        phot_bp_mean_mag = x_physical[:, 10] 
        phot_rp_mean_mag = x_physical[:, 11]
        g_mean_psf_mag = x_physical[:, 22]
        r_mean_psf_mag = x_physical[:, 24]
        i_mean_psf_mag = x_physical[:, 26]
        z_mean_psf_mag = x_physical[:, 28]
        y_mean_psf_mag = x_physical[:, 30]
        r_med_geo = x_physical[:, 19]
        l = x_physical[:, 38]  # galactic longitude
        b = x_physical[:, 39]  # galactic latitude
        
        # Calculate BP-RP color in physical units for extinction function
        bp_rp_color = phot_bp_mean_mag - phot_rp_mean_mag
        
        # Apply extinction using vectorized function
        if has_twomass:
            # Extract 2MASS data
            j_m = x_physical[:, 32]
            h_m = x_physical[:, 34]
            ks_m = x_physical[:, 36]
            
            colors_reddened = vmap(self._apply_reddening_to_colors)(
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                g_mean_psf_mag, r_mean_psf_mag, i_mean_psf_mag, 
                z_mean_psf_mag, y_mean_psf_mag, ebv_samples, bp_rp_color,
                j_m, h_m, ks_m
            )
        else:
            colors_reddened = vmap(self._apply_reddening_to_colors)(
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                g_mean_psf_mag, r_mean_psf_mag, i_mean_psf_mag, 
                z_mean_psf_mag, y_mean_psf_mag, ebv_samples, bp_rp_color
            )
        
        # Calculate absolute magnitudes (clean - intrinsic property)
        mu = 5*jnp.log10(r_med_geo) - 5
        M_G = phot_g_mean_mag - mu
        
        # Calculate galactocentric coordinates (vectorized)
        l_rad = jnp.radians(l)
        b_rad = jnp.radians(b)
        R_sun = 8000.0
        Z_sun = 25.0
        x_gc = r_med_geo * jnp.cos(b_rad) * jnp.cos(l_rad)
        y_gc = r_med_geo * jnp.cos(b_rad) * jnp.sin(l_rad)
        z_gc = r_med_geo * jnp.sin(b_rad)
        R = jnp.sqrt((R_sun - x_gc)**2 + y_gc**2)
        Z = jnp.abs(z_gc - Z_sun)
        log_Z = jnp.log10(Z + 1e-6)
        log_R = jnp.log10(R + 1e-6)
        flow_conditions_physical = jnp.stack([log_Z, log_R], axis=-1)
        
        # Normalize conditions if available
        if train_stats_mean is not None and train_stats_std is not None:
            if has_twomass:
                # Flow conditions: [log_Z, log_R] from indices [12, 11] 
                flow_condition_mean = jnp.array([train_stats_mean[12], train_stats_mean[11]])
                flow_condition_std = jnp.array([train_stats_std[12], train_stats_std[11]])
            else:
                # Flow conditions: [log_Z, log_R] from indices [9, 8] 
                flow_condition_mean = jnp.array([train_stats_mean[9], train_stats_mean[8]])
                flow_condition_std = jnp.array([train_stats_std[9], train_stats_std[8]])
            flow_conditions = (flow_conditions_physical - flow_condition_mean) / flow_condition_std
        else:
            flow_conditions = flow_conditions_physical

            
        # Prepare reddened flow inputs: [M_G, colors_reddened]
        flow_inputs_reddened_physical = jnp.concatenate([M_G[:, None], colors_reddened], axis=-1)
        
        # Check if 2MASS data is enabled
        has_twomass = self.twomass
        
        # Normalize reddened inputs if available  
        if train_stats_mean is not None and train_stats_std is not None:
            if has_twomass:
                # train_stats structure for 2MASS: [10 colors, M_G, log_R, log_Z] = indices [0:10, 10, 11, 12]
                # Flow inputs: [M_G, 10 colors] from indices [10, 0:10]
                flow_input_mean = jnp.concatenate([train_stats_mean[10:11], train_stats_mean[0:10]])
                flow_input_std = jnp.concatenate([train_stats_std[10:11], train_stats_std[0:10]])
                flow_inputs_reddened = (flow_inputs_reddened_physical - flow_input_mean) / flow_input_std
                
                # Flow conditions: [log_Z, log_R] from indices [12, 11] 
                flow_condition_mean = jnp.array([train_stats_mean[12], train_stats_mean[11]])
                flow_condition_std = jnp.array([train_stats_std[12], train_stats_std[11]])
            else:
                # Original structure: [7 colors, M_G, log_R, log_Z] = indices [0:7, 7, 8, 9]
                flow_input_mean = jnp.concatenate([train_stats_mean[7:8], train_stats_mean[0:7]])
                flow_input_std = jnp.concatenate([train_stats_std[7:8], train_stats_std[0:7]])
                flow_inputs_reddened = (flow_inputs_reddened_physical - flow_input_mean) / flow_input_std
                
                # Flow conditions: [log_Z, log_R] from indices [9, 8] 
                flow_condition_mean = jnp.array([train_stats_mean[9], train_stats_mean[8]])
                flow_condition_std = jnp.array([train_stats_std[9], train_stats_std[8]])
            
            flow_conditions = (flow_conditions_physical - flow_condition_mean) / flow_condition_std
        else:
            flow_inputs_reddened = flow_inputs_reddened_physical
            flow_conditions = flow_conditions_physical
            
        # Evaluate flow probabilities for reddened samples
        log_p_reddened = dist_flow.log_prob(flow_inputs_reddened, flow_conditions)
        
        # Clamp individual log probabilities to prevent extreme values
        log_p_reddened_clamped = jnp.maximum(log_p_reddened, self.contrastive_clamp_bound)
        
        # Simple contrastive loss: encourage low probability for reddened samples
        # (No need for clean samples - they're already handled by evidence loss)
        contrastive_loss = log_p_reddened_clamped.mean()  # Want this to be negative (low prob for reddened)
        
        return contrastive_loss

    def _compute_log_evidence_single(self, x_single, key, distance_samples, 
                                   dist_flow, mean, std, mag_error_single, color_noise_std_single):
        """Compute log evidence for a single data point."""
        l, b = x_single[0], x_single[1]
        distance_obs = x_single[2]  # observed distance estimate (pc)
        observed_mag = x_single[5]
        observed_colors = x_single[6:]
        
        # Compute log weights for each distance sample
        log_weights = []
        log_likelihoods = []
        
        for i in range(self.n_distance_samples):
            d = distance_samples[i]
            
            # Compute Z, R from distance and galactic coordinates
            Z = 5*jnp.log10(jnp.sin(jnp.abs(jnp.radians(b))) * d) - 5
            R = 5*jnp.log10(jnp.sqrt(8000**2 + (d*jnp.cos(jnp.radians(b)))**2 
                                   - 2*8000*(d*jnp.cos(jnp.radians(b)))*jnp.cos(jnp.radians(l)))) - 5
            
            # Normalize Z, R 
            if mean is not None and std is not None:
                Z_norm = (Z - mean[3]) / std[3]  # Assuming Z is at index 3
                R_norm = (R - mean[4]) / std[4]  # Assuming R is at index 4
            else:
                Z_norm, R_norm = Z, R
                
            # Sample absolute magnitude from proposal (apparent mag - distance modulus) with noise
            mu = 5*jnp.log10(d) - 5  # distance modulus
            Mg_proposal_mean = observed_mag - mu
            
            # Add noise from magnitude error
            key, subkey = jr.split(key)
            if mag_error_single is not None:
                Mg_noise = jr.normal(subkey) * mag_error_single
                Mg_proposal = Mg_proposal_mean + Mg_noise
            else:
                Mg_proposal = Mg_proposal_mean
            
            # Normalize Mg
            if mean is not None and std is not None:
                Mg_norm = (Mg_proposal - mean[5]) / std[5]  # Assuming Mg is at index 5
            else:
                Mg_norm = Mg_proposal
                
            # Sample colors with proper error propagation
            key, subkey = jr.split(key)
            if color_noise_std_single is not None:
                color_noise = jr.normal(subkey, observed_colors.shape) * color_noise_std_single
                colors_sample = observed_colors + color_noise
            else:
                colors_sample = observed_colors
            
            # Normalize colors
            if mean is not None and std is not None:
                colors_norm = (colors_sample - mean[6:]) / std[6:]
            else:
                colors_norm = colors_sample
                
            # Concatenate features for flow evaluation  
            flow_input = jnp.concatenate([colors_norm])  # Just colors for now
            flow_condition = jnp.array([Z_norm, R_norm, Mg_norm])
            
            # Evaluate flow log probability
            log_p_flow = dist_flow.log_prob(flow_input[None], flow_condition[None])[0]
            log_likelihoods.append(log_p_flow)
            
            # Distance prior log probability
            log_prior_d = self.beta * jnp.log(d) - (d/self.L)**self.alpha
            
            # Proposal density (uniform around observed distance)
            d_min = jnp.maximum(distance_obs * 0.5, 1.0)
            d_max = jnp.minimum(distance_obs * 2.0, 50000.0)  
            log_proposal = -jnp.log(d_max - d_min)  # uniform density
            
            # Importance weight = prior / proposal
            log_weight = log_prior_d - log_proposal
            log_weights.append(log_weight)
        
        # Convert to arrays
        log_weights = jnp.array(log_weights)
        log_likelihoods = jnp.array(log_likelihoods) 
        
        # Compute log evidence using logsumexp for numerical stability
        # log evidence = logsumexp(log_weight + log_likelihood) - logsumexp(log_weight)
        log_numerator = logsumexp(log_weights + log_likelihoods)
        log_denominator = logsumexp(log_weights)
        log_evidence = log_numerator - log_denominator
        
        return log_evidence












