import jax.numpy as jnp
import abc
import jax.random as jr
from flowjax.distributions import Normal,Uniform
import equinox as eqx
import optax
import flowjax
import functools
from tqdm import tqdm
from losses import MaximumLikelihoodLoss,MaximumLikelihoodLoss_nocond,MaximumLikelihoodLoss_embedding,ShannonLossEmbedding,VAELoss,GradientModelLoss,MgFocusedMaximumLikelihoodLoss,EvidenceMaxLoss
from tensorflow.summary import create_file_writer
from utils import get_optimizer
import os
import jax
import pickle
import numpy as np
import tensorflow as tf
from activation_functions import get_activation
from flowjax.flows import masked_autoregressive_flow,block_neural_autoregressive_flow,coupling_flow
from flowjax.bijections import RationalQuadraticSpline
import paramax
from models.nlpe import nlpe_bnaf_elu
from models.embeddings import StatisticEmbedding,StatisticEmbedding_spectra,StatisticDecoder,Discriminator
from models.grad_estimator import GradModel

            
# Constants
_FLOWS = {}

def register_flow(cls=None, *, name=None):
    """A decorator for registering flow classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _FLOWS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _FLOWS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register

def get_flow(name):
    """Retrieve a registered flow class by name."""
    return _FLOWS[name]

class Normalizing_Flow(abc.ABC):
    """Base class for normalizing flow models."""
    def __init__(self, config, *args, **kwargs):
        
        self.config = config
        self.train_bool = config.training.train
        self.optimizer = get_optimizer(config)
        #self.loss_function = G23_Loss()
        # Select loss function based on config
        if hasattr(config, 'loss') and hasattr(config.loss, 'use_evidence_max') and config.loss.use_evidence_max:
            L = getattr(config.loss, 'L', 1000.0)
            alpha = getattr(config.loss, 'alpha', 2.0) 
            beta = getattr(config.loss, 'beta', 1.0)
            n_distance_samples = getattr(config.loss, 'n_distance_samples',10)
            use_contrastive = getattr(config.loss, 'use_contrastive', False)
            contrastive_weight = getattr(config.loss, 'contrastive_weight', 1.0)
            contrastive_samples = getattr(config.loss, 'contrastive_samples', 5)
            contrastive_clamp_bound = getattr(config.loss, 'contrastive_clamp_bound', -50.0)
            twomass = getattr(config.data, 'twomass', False)
            self.loss_function = EvidenceMaxLoss(
                L=L,
                alpha=alpha,
                beta=beta,
                n_distance_samples=n_distance_samples,
                use_contrastive=use_contrastive,
                contrastive_weight=contrastive_weight,
                contrastive_samples=contrastive_samples,
                contrastive_clamp_bound=contrastive_clamp_bound,
                twomass=twomass
            )
        elif hasattr(config, 'loss') and hasattr(config.loss, 'use_mg_focused') and config.loss.use_mg_focused:
            lambda_gp = getattr(config.loss, 'lambda_gp', 1.0)
            lambda_smooth = getattr(config.loss, 'lambda_smooth', 0.1)
            mg_weight_multiplier = getattr(config.loss, 'mg_weight_multiplier', 2.0)
            self.loss_function = MgFocusedMaximumLikelihoodLoss(
                lambda_gp=lambda_gp,
                lambda_smooth=lambda_smooth,
                mg_weight_multiplier=mg_weight_multiplier
            )
        else:
            self.loss_function = MaximumLikelihoodLoss()
        self.mean = None
        self.std = None
        self.train_data = None
        self.eval_data = None




    def train(self):
        """Train the model."""
        self.key, self.subkey = jr.split(self.key)
        self.flow, self.losses = self.fit_to_data(
            key=self.subkey,
            dist=self.flow,
            x=self.train_data,
            eval=self.eval_data,
            learning_rate=self.config.optim.lr,
            max_epochs=self.config.training.n_iters,
            max_patience=self.config.training.max_patience,
            val_prop=self.config.training.validation_split
        )
        self.distance_flow, self.losses = self.fit_to_data_distance(
            key=self.subkey,
            dist=self.distance_flow,
            x=self.train_data,
            eval=self.eval_data,
            learning_rate=self.config.optim.lr,
            max_epochs=self.config.training.n_iters,
            max_patience=self.config.training.max_patience,
            val_prop=self.config.training.validation_split
        )

    def fit_to_data(self, key, dist, x, eval, max_epochs, max_patience, val_prop, learning_rate, return_best=True, show_progress=True):
        """Fit the model to the data."""
        workspace_dir = str(self.config.training.workspace)+'/Tensorboard'
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        summary_writer = create_file_writer(workspace_dir)

        #x=x.reshape(-1,x.shape[-1])
        summary_dim= self.config.data.summary_dim
        cond_dim = self.config.model.cond_dim
        vector_dim = self.config.data.vector_dim
        #x_batch=x[...,summary_dim+cond_dim:summary_dim+cond_dim+vector_dim]
        #condition_batch=x[...,summary_dim:summary_dim+cond_dim]
        #condition_batch=condition_batch.at[...,0:2].set(jnp.sigmoid(condition_batch[...,0:2]))  # Set the first condition to 1.0
        #dist,losses=flowjax.train.fit_to_data(key,dist,x_batch,batch_size=1000,learning_rate=1e-3,condition=condition_batch,show_progress=True)
        #return dist, losses

        optimizer = self.optimizer
        loss_fn = self.loss_function

        


        params, static = eqx.partition(
            dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )



        @eqx.filter_jit
        def step(params, batch, opt_state, key):
            """Perform a single training step."""
            # For EvidenceMaxLoss, use the full batch without conditioning
            if hasattr(self.loss_function, '_compute_log_evidence_single_gaia'):
                # EvidenceMaxLoss case - use full batch and pass mean/std/train_stats
                loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
                    params, static, batch, None, key, self.mean, self.std, self.train_stats_mean, self.train_stats_std
                )
            else:
                # Standard loss case
                x_batch=batch[...,summary_dim+cond_dim:summary_dim+cond_dim+vector_dim]
                condition_batch=batch[...,summary_dim:summary_dim+cond_dim]
                loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params,static,x_batch,condition_batch,key)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss_val
        best_params = params
        opt_state = optimizer.init(params)

        key, subkey = jr.split(key)
        losses = {"train": [], "val": []}

        best_val_loss = float('inf')
        patience_counter = 0


        loop_epoch = tqdm(range(max_epochs), disable=not show_progress, desc="Main Flow Training")
        for epoch in loop_epoch:
            train_losses = []

            for batch_data in x:

                #batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch_data)
                batch=batch_data
                key, subkey = jr.split(key)
                # at some point should change to be allow for jitted steps
                params, opt_state, train_loss = step(params, batch, opt_state, subkey)
                train_losses.append(train_loss)

            # Compute full-batch training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            losses["train"].append(avg_train_loss)

            # Compute validation loss every 20 epochs (or on first/last epoch)
            if epoch % 20 == 0 or epoch == max_epochs - 1:
                # Compute full-batch validation loss
                val_losses = []
                for i, val_batch_data in enumerate(eval):
                    val_batch = jax.tree_util.tree_map(lambda x: jnp.array(x), val_batch_data)
                    # For EvidenceMaxLoss, use the full batch without conditioning
                    if hasattr(self.loss_function, '_compute_log_evidence_single_gaia'):
                        # EvidenceMaxLoss case - use full batch and pass mean/std/train_stats
                        val_loss = loss_fn(params, static, val_batch, None, None, self.mean, self.std, self.train_stats_mean, self.train_stats_std)
                        # Debug logging for first validation batch
                        if i == 0:
                            print(f"[DEBUG] Validation batch {i}: batch_size={val_batch.shape[0]}, val_loss={val_loss:.6f}")
                    else:
                        # Standard loss case
                        x_batch=val_batch[...,summary_dim+cond_dim:summary_dim+cond_dim+vector_dim]
                        condition_batch=val_batch[...,summary_dim:summary_dim+cond_dim]
                        val_loss = loss_fn(params,static,x_batch,condition_batch,None)
                    val_losses.append(val_loss)

                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
                losses["val"].append(avg_val_loss)
                
                # Early stopping check (only on validation epochs)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                    # Convert epoch-based patience to validation steps (validate every 20 epochs)
                    patience_validation_steps = max(3, max_patience // 20)  # At least 3 validation checks
                    if patience_counter >= patience_validation_steps:
                        actual_epochs_waited = patience_counter * 20
                        print(f"Early stopping! No improvement for {actual_epochs_waited} epochs ({patience_counter} validation steps)")
                        break
            else:
                # Use the last computed validation loss for non-validation epochs
                avg_val_loss = losses["val"][-1] if losses["val"] else float('inf')
                losses["val"].append(avg_val_loss)

            # Print losses every epoch for monitoring
            if hasattr(self.loss_function, 'use_contrastive') and self.loss_function.use_contrastive:
                # Log contrastive learning components if available (avoid JAX tracer formatting issues)
                if hasattr(self.loss_function, '_last_evidence_loss'):
                    try:
                        evidence_comp = float(getattr(self.loss_function, '_last_evidence_loss', 0.0))
                        contrastive_comp = float(getattr(self.loss_function, '_last_contrastive_loss', 0.0))
                        print(f"[MAIN] Epoch {epoch:3d}/{max_epochs}: Train Loss = {avg_train_loss:.6f} [Evidence: {evidence_comp:.6f}, Contrastive: {contrastive_comp:.6f}], Val Loss = {avg_val_loss:.6f}")
                    except:
                        print(f"[MAIN] Epoch {epoch:3d}/{max_epochs}: Train Loss = {avg_train_loss:.6f} [Contrastive Enabled], Val Loss = {avg_val_loss:.6f}")
                else:
                    print(f"[MAIN] Epoch {epoch:3d}/{max_epochs}: Train Loss = {avg_train_loss:.6f} [Contrastive Enabled], Val Loss = {avg_val_loss:.6f}")
            else:
                print(f"[MAIN] Epoch {epoch:3d}/{max_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            # Debug logging for evidence loss components (every 20 epochs)
            if hasattr(self.loss_function, '_compute_log_evidence_single_gaia') and (epoch % 20 == 0 or epoch == max_epochs - 1):
                # Enable contrastive debugging for this epoch if using contrastive learning
                if hasattr(self.loss_function, 'use_contrastive') and self.loss_function.use_contrastive:
                    self.loss_function._debug_contrastive = True
                    print(f"[DEBUG] Contrastive learning enabled: weight={self.loss_function.contrastive_weight}, samples={self.loss_function.contrastive_samples}")
                
                # Get a single validation batch for detailed analysis
                if len(eval) > 0:
                    val_batch_sample = next(iter(eval))
                    val_batch_sample = jax.tree_util.tree_map(lambda x: jnp.array(x), val_batch_sample)
                    
                    # Take just the first star for detailed logging
                    x_single_physical = (val_batch_sample[0] * self.std + self.mean)
                    
                    print(f"[DEBUG] Evidence Loss Components Analysis:")
                    print(f"[DEBUG] Star parallax: {x_single_physical[3]:.4f} ± {x_single_physical[4]:.4f} mas")
                    print(f"[DEBUG] Star distance estimate: {x_single_physical[19]:.1f} pc")
                    print(f"[DEBUG] Star G magnitude: {x_single_physical[9]:.3f}")
                    print(f"[DEBUG] Galactic coords: l={x_single_physical[32]:.2f}, b={x_single_physical[33]:.2f}")
                    
                    # Compute detailed evidence components for this star
                    self._debug_evidence_components(
                        params, static, x_single_physical, jax.random.PRNGKey(42)
                    )

            # Log losses only after the full batch
            with summary_writer.as_default():
                tf.summary.scalar("train_loss", avg_train_loss, step=epoch)
                tf.summary.scalar("val_loss", avg_val_loss, step=epoch)
                summary_writer.flush()  # Force write to disk

            # Log model parameters every 10 epochs
            if epoch % 1 == 0:
                with summary_writer.as_default():
                    for i, param in enumerate(jax.tree_util.tree_leaves(params)):
                        tf.summary.histogram(f"parameters/param_{i}", np.array(param), step=epoch)
                    total_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)))
                    tf.summary.scalar("parameters/total_norm", np.array(total_norm), step=epoch)

            # Early stopping now handled inside validation block above


        summary_writer.close()
        dist = eqx.combine(best_params, static) if return_best else eqx.combine(params, static)
        return dist, losses

    def _debug_evidence_components(self, params, static, x_single_physical, key):
        """Debug method to analyze evidence components for a single star."""
        loss_fn = self.loss_function
        
        # Check if 2MASS data is available from config
        twomass_available = getattr(self.config.data, 'twomass', False)
        
        # Get distance sampling parameters from galactic coordinates
        # For both cases, l and b are at the end: indices 38, 39
        l_gal = x_single_physical[38]  # l coordinate (galactic longitude)
        b_gal = x_single_physical[39]  # b coordinate (galactic latitude)
        r_med_geo = x_single_physical[19]  # geometric distance estimate
        parallax_true = x_single_physical[3]  # parallax
        parallax_err = x_single_physical[4]  # parallax error
        
        print(f"[DEBUG] Evidence Components for single star:")
        print(f"[DEBUG] Galactic coords: l={l_gal:.2f}°, b={b_gal:.2f}°")
        print(f"[DEBUG] Geometric distance: {r_med_geo:.1f} pc")
        print(f"[DEBUG] Parallax: {parallax_true:.4f} ± {parallax_err:.4f} mas")
        
        # Sample distances for this star
        L = loss_fn.L
        alpha = loss_fn.alpha
        beta = loss_fn.beta
        n_samples = loss_fn.n_distance_samples
        
        # Generate distance samples from the geometric distance proposal
        key, subkey = jax.random.split(key)
        sigma_d = 0.1 * r_med_geo  # 10% uncertainty
        raw_samples = jax.random.normal(subkey, (n_samples,)) * sigma_d + r_med_geo
        distance_samples = jnp.clip(raw_samples, 10.0, 50000.0)
        
        print(f"[DEBUG] Distance samples: min={distance_samples.min():.1f}, max={distance_samples.max():.1f}, mean={distance_samples.mean():.1f}")
        
        # Compute individual components
        # Distance prior: p(d) ∝ d^β * exp(-(d/L)^α)  
        log_prior_d = beta * jnp.log(distance_samples) - (distance_samples / L) ** alpha
        
        # Parallax likelihood: N(1000/d, σ_π)
        predicted_parallax = 1000.0 / distance_samples
        log_likelihood_parallax = -0.5 * ((parallax_true - predicted_parallax) / parallax_err) ** 2
        
        # Distance proposal likelihood: N(r_med_geo, σ_d)
        log_proposal_d = -0.5 * ((distance_samples - r_med_geo) / sigma_d) ** 2
        
        # Convert distances to Z,R coordinates using galactic transformation
        z_coords = distance_samples * jnp.sin(jnp.radians(b_gal))
        r_coords = jnp.sqrt((distance_samples * jnp.cos(jnp.radians(b_gal)))**2 + 8000.0**2 - 2*distance_samples*jnp.cos(jnp.radians(b_gal))*8000.0*jnp.cos(jnp.radians(l_gal)))
        
        # Flow evaluation at each distance sample
        conditions_physical = jnp.stack([jnp.log10(jnp.abs(z_coords) + 1e-6), jnp.log10(r_coords + 1e-6)], axis=-1)
        
        # Get magnitude/color data using correct column indices
        phot_g_mean_mag = x_single_physical[9]    # phot_g_mean_mag
        phot_bp_mean_mag = x_single_physical[10]  # phot_bp_mean_mag  
        phot_rp_mean_mag = x_single_physical[11]  # phot_rp_mean_mag
        r_med_geo = x_single_physical[19]         # r_med_geo
        g_mean_psf_mag = x_single_physical[22]    # g_mean_psf_mag
        r_mean_psf_mag = x_single_physical[24]    # r_mean_psf_mag
        i_mean_psf_mag = x_single_physical[26]    # i_mean_psf_mag
        z_mean_psf_mag = x_single_physical[28]    # z_mean_psf_mag
        y_mean_psf_mag = x_single_physical[30]    # y_mean_psf_mag
        
        # Calculate absolute magnitude M_G
        M_G = phot_g_mean_mag - 5*jnp.log10(r_med_geo) + 5
        
        # Check if 2MASS data is available (based on config or column count)
        twomass_available = getattr(self.config.data, 'twomass', False)
        
        if twomass_available:
            # For 2MASS data: calculate 10 colors (7 PS1 + 3 2MASS relative to G)
            j_m = x_single_physical[32]               # j_m
            h_m = x_single_physical[34]               # h_m  
            ks_m = x_single_physical[36]              # ks_m
            
            bp_g_obs = phot_bp_mean_mag - phot_g_mean_mag
            rp_g_obs = phot_rp_mean_mag - phot_g_mean_mag
            g_ps1_g_gaia_obs = g_mean_psf_mag - phot_g_mean_mag
            r_ps1_g_gaia_obs = r_mean_psf_mag - phot_g_mean_mag
            i_ps1_g_gaia_obs = i_mean_psf_mag - phot_g_mean_mag
            z_ps1_g_gaia_obs = z_mean_psf_mag - phot_g_mean_mag
            y_ps1_g_gaia_obs = y_mean_psf_mag - phot_g_mean_mag
            j_g_gaia_obs = j_m - phot_g_mean_mag
            h_g_gaia_obs = h_m - phot_g_mean_mag
            ks_g_gaia_obs = ks_m - phot_g_mean_mag
            
            colors_obs = jnp.array([bp_g_obs, rp_g_obs, g_ps1_g_gaia_obs,
                                   r_ps1_g_gaia_obs, i_ps1_g_gaia_obs,
                                   z_ps1_g_gaia_obs, y_ps1_g_gaia_obs,
                                   j_g_gaia_obs, h_g_gaia_obs, ks_g_gaia_obs])
        else:
            # Original gaia_ps1 case: calculate 7 colors relative to G
            bp_g_obs = phot_bp_mean_mag - phot_g_mean_mag
            rp_g_obs = phot_rp_mean_mag - phot_g_mean_mag
            g_ps1_g_gaia_obs = g_mean_psf_mag - phot_g_mean_mag
            r_ps1_g_gaia_obs = r_mean_psf_mag - phot_g_mean_mag
            i_ps1_g_gaia_obs = i_mean_psf_mag - phot_g_mean_mag
            z_ps1_g_gaia_obs = z_mean_psf_mag - phot_g_mean_mag
            y_ps1_g_gaia_obs = y_mean_psf_mag - phot_g_mean_mag
            
            colors_obs = jnp.array([bp_g_obs, rp_g_obs, g_ps1_g_gaia_obs,
                                   r_ps1_g_gaia_obs, i_ps1_g_gaia_obs,
                                   z_ps1_g_gaia_obs, y_ps1_g_gaia_obs])
        
        # Flow input: [M_G, colors] in physical units
        flow_input_physical = jnp.concatenate([jnp.array([M_G]), colors_obs])
        
        print(f"[DEBUG] Flow inputs and conditions:")
        print(f"[DEBUG] M_G + colors: {flow_input_physical}")
        print(f"[DEBUG] Conditions range: log10|Z| [{conditions_physical[:, 0].min():.3f}, {conditions_physical[:, 0].max():.3f}], log10 R [{conditions_physical[:, 1].min():.3f}, {conditions_physical[:, 1].max():.3f}]")
        
        # Evaluate flow for each condition using same normalization as evidence calculation
        flow = eqx.combine(params, static)
        log_p_flows = []
        for i in range(n_samples):
            condition_physical = conditions_physical[i]  # [log_Z, log_R]
            
            # Apply same normalization as in evidence calculation
            if self.train_stats_mean is not None and self.train_stats_std is not None:
                if twomass_available:
                    # train_stats structure for 2MASS: [10 colors, M_G, log_R, log_Z] = indices [0:10, 10, 11, 12]
                    # Flow inputs: [M_G, 10 colors] from indices [10, 0:10]
                    flow_input_mean = jnp.concatenate([self.train_stats_mean[10:11], self.train_stats_mean[0:10]])
                    flow_input_std = jnp.concatenate([self.train_stats_std[10:11], self.train_stats_std[0:10]]) 
                    flow_input = (flow_input_physical - flow_input_mean) / flow_input_std
                    
                    # Flow conditions: [log_Z, log_R] from indices [12, 11] 
                    flow_condition_mean = jnp.array([self.train_stats_mean[12], self.train_stats_mean[11]])  # [log_Z, log_R]
                    flow_condition_std = jnp.array([self.train_stats_std[12], self.train_stats_std[11]])
                    flow_condition = (condition_physical - flow_condition_mean) / flow_condition_std
                else:
                    # train_stats structure: [7 colors, M_G, log_R, log_Z] = indices [0:7, 7, 8, 9]
                    # Flow inputs: [M_G, 7 colors] from indices [7, 0:7]
                    flow_input_mean = jnp.concatenate([self.train_stats_mean[7:8], self.train_stats_mean[0:7]])
                    flow_input_std = jnp.concatenate([self.train_stats_std[7:8], self.train_stats_std[0:7]]) 
                    flow_input = (flow_input_physical - flow_input_mean) / flow_input_std
                    
                    # Flow conditions: [log_Z, log_R] from indices [9, 8] 
                    flow_condition_mean = jnp.array([self.train_stats_mean[9], self.train_stats_mean[8]])  # [log_Z, log_R]
                    flow_condition_std = jnp.array([self.train_stats_std[9], self.train_stats_std[8]])
                    flow_condition = (condition_physical - flow_condition_mean) / flow_condition_std
            else:
                # Fall back to physical units (no normalization)
                flow_input = flow_input_physical
                flow_condition = condition_physical
            
            log_p = flow.log_prob(flow_input[None, :], condition=flow_condition[None])[0]
            log_p_flows.append(log_p)
            if i < 3:  # Log first 3 samples for debugging
                print(f"[DEBUG] Sample {i}: log10|Z|={condition_physical[0]:.3f}, log10 R={condition_physical[1]:.3f}, log_p_flow={log_p:.3f}")
                if self.train_stats_mean is not None:
                    print(f"[DEBUG]   Normalized input: {flow_input}")
                    print(f"[DEBUG]   Normalized condition: {flow_condition}")
        
        log_p_flows = jnp.array(log_p_flows)
        
        print(f"[DEBUG] Component statistics:")
        print(f"[DEBUG]   log_prior_d: min={log_prior_d.min():.2f}, max={log_prior_d.max():.2f}, mean={log_prior_d.mean():.2f}")
        print(f"[DEBUG]   log_likelihood_parallax: min={log_likelihood_parallax.min():.2f}, max={log_likelihood_parallax.max():.2f}, mean={log_likelihood_parallax.mean():.2f}")
        print(f"[DEBUG]   log_proposal_d: min={log_proposal_d.min():.2f}, max={log_proposal_d.max():.2f}, mean={log_proposal_d.mean():.2f}")
        print(f"[DEBUG]   log_p_flow: min={log_p_flows.min():.2f}, max={log_p_flows.max():.2f}, mean={log_p_flows.mean():.2f}")
        
        # Importance weights
        log_weights = log_prior_d + log_likelihood_parallax - log_proposal_d
        print(f"[DEBUG]   log_weights: min={log_weights.min():.2f}, max={log_weights.max():.2f}, mean={log_weights.mean():.2f}")
        
        # Total log evidence contribution
        from jax.scipy.special import logsumexp
        log_evidence = logsumexp(log_weights + log_p_flows) - jnp.log(n_samples)
        print(f"[DEBUG]   log_evidence for this star: {log_evidence:.4f}")
        
        # Check for any problematic values
        if jnp.any(jnp.isnan(log_weights)) or jnp.any(jnp.isnan(log_p_flows)):
            print(f"[DEBUG]   WARNING: NaN values detected!")
        if jnp.any(jnp.isinf(log_weights)) or jnp.any(jnp.isinf(log_p_flows)):
            print(f"[DEBUG]   WARNING: Inf values detected!")
            
        print(f"[DEBUG] End evidence components analysis\n")
        
        # Disable contrastive debugging after this epoch
        if hasattr(self.loss_function, 'use_contrastive') and self.loss_function.use_contrastive:
            self.loss_function._debug_contrastive = False

    def fit_to_data_distance(self, key, dist, x, eval, max_epochs, max_patience, val_prop, learning_rate, return_best=True, show_progress=True):
        """Fit the model to the data."""
        workspace_dir = str(self.config.training.workspace)+'/Tensorboard'
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        summary_writer = create_file_writer(workspace_dir)

        #x=x.reshape(-1,x.shape[-1])
        summary_dim= self.config.data.summary_dim
        cond_dim = self.config.model.cond_dim
        vector_dim = self.config.data.vector_dim
        #x_batch=x[...,summary_dim+cond_dim:summary_dim+cond_dim+vector_dim]
        #condition_batch=x[...,summary_dim:summary_dim+cond_dim]
        #condition_batch=condition_batch.at[...,0:2].set(jnp.sigmoid(condition_batch[...,0:2]))  # Set the first condition to 1.0
        #dist,losses=flowjax.train.fit_to_data(key,dist,x_batch,batch_size=1000,learning_rate=1e-3,condition=condition_batch,show_progress=True)
        #return dist, losses

        optimizer = self.optimizer
        loss_fn = self.loss_function

        


        params, static = eqx.partition(
            dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )



        @eqx.filter_jit
        def step(params, batch, opt_state, key):
            """Perform a single training step."""
            x_batch=batch[...,6:7]
            condition_batch=batch[...,3:4]
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params,static,x_batch,condition_batch,key)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss_val
        best_params = params
        opt_state = optimizer.init(params)

        key, subkey = jr.split(key)
        losses = {"train": [], "val": []}

        best_val_loss = float('inf')
        patience_counter = 0


        loop_epoch = tqdm(range(0), disable=not show_progress, desc="Distance Flow Training (Skipped)")
        for epoch in loop_epoch:
            train_losses = []

            for batch_data in x:

                #batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch_data)
                batch=batch_data
                key, subkey = jr.split(key)
                # at some point should change to be allow for jitted steps
                params, opt_state, train_loss = step(params, batch, opt_state, subkey)
                train_losses.append(train_loss)

            # Compute full-batch training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            losses["train"].append(avg_train_loss)

            # Compute full-batch validation loss
            val_losses = []
            for val_batch_data in eval:
                val_batch = jax.tree_util.tree_map(lambda x: jnp.array(x), val_batch_data)
                x_batch=val_batch[...,6:7]
                condition_batch=val_batch[...,3:4]
                val_loss = loss_fn(params,static,x_batch,condition_batch,None)
                val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            losses["val"].append(avg_val_loss)

            # Print losses every epoch for monitoring
            print(f"[DIST] Epoch {epoch:3d}/{max_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            # Log losses only after the full batch
            with summary_writer.as_default():
                tf.summary.scalar("train_loss", avg_train_loss, step=epoch)
                tf.summary.scalar("val_loss", avg_val_loss, step=epoch)
                summary_writer.flush()  # Force write to disk

            # Log model parameters every 10 epochs
            if epoch % 1 == 0:
                with summary_writer.as_default():
                    for i, param in enumerate(jax.tree_util.tree_leaves(params)):
                        tf.summary.histogram(f"parameters/param_{i}", np.array(param), step=epoch)
                    total_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)))
                    tf.summary.scalar("parameters/total_norm", np.array(total_norm), step=epoch)

            # Early stopping now handled inside validation block above


        summary_writer.close()
        dist = eqx.combine(best_params, static) if return_best else eqx.combine(params, static)
        return dist, losses


    def evaluate_log_likelihood(self):
        """Evaluate the log likelihood on training and evaluation sets.""" 
        batch_size = self.config.training.batch_size 
        def compute_log_likelihood(dataset):
            """Compute log likelihood over a dataset."""
            log_likelihoods = []
            data_iter = iter(dataset)
            
            num_batches = len(dataset) // batch_size
            for _ in range(num_batches):
                try:
                    batch_data = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataset)
                    batch_data = next(data_iter)
                
                batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch_data)
                log_likelihood = -self.loss_function(self.flow, batch)
                log_likelihoods.append(log_likelihood)
            
            avg_log_likelihood = sum(log_likelihoods) / len(log_likelihoods) if log_likelihoods else float('-inf')
            return avg_log_likelihood
        
        # Compute log likelihood for training and evaluation sets
        train_log_likelihood = compute_log_likelihood(self.train_data)
        eval_log_likelihood = compute_log_likelihood(self.eval_data)

        print(f"Train Log Likelihood: {train_log_likelihood:.4f}")
        print(f"Eval Log Likelihood: {eval_log_likelihood:.4f}")

        return {
            'train_log_likelihood': train_log_likelihood,
            'eval_log_likelihood': eval_log_likelihood
        }





@register_flow(name='nlpe_bnaf_elu')
class Block_Neural_Autoregressive(Normalizing_Flow):
    """Block Neural Autoregressive Flow model."""
    def __init__(self, config,key, data=None, eval_data=None, mean=None, std=None):
        super().__init__(config)
        self.key, self.subkey = jr.split(key)
        self.nn_depth = config.model.nn_depth_bnaf
        self.nn_block_dim = config.model.nn_block_dim
        self.cond_dim = config.model.cond_dim
        self.activation = get_activation(config)
        self.file_name = os.path.join(config.data.data_path, f"{config.model.name}_{config.data.dataset}.eqx")
        self.file_name_distance = os.path.join(config.data.data_path, f"{config.model.name}_distance_{config.data.dataset}.eqx")

        self.flow_dimension=config.model.flow_dimension
        
        
        self.flow = nlpe_bnaf_elu(
            key=self.key,
            cond_dim=self.cond_dim,
            base_dist=Normal(jnp.zeros(self.flow_dimension), jnp.ones(self.flow_dimension)),
            invert=True,
            nn_depth=self.nn_depth,
            nn_block_dim=self.nn_block_dim,
            activation=self.activation,
            flow_layers=2
        )

        

        self.distance_flow = nlpe_bnaf_elu(
            key=self.key,
            cond_dim=1,
            base_dist=Normal(jnp.zeros(1), jnp.ones(1)),
            invert=True,
            nn_depth=2,
            nn_block_dim=3,
            activation=self.activation
        )


    def build(self, train_data=None, eval_data=None, mean=None, std=None, train_stats_mean=None, train_stats_std=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.mean, self.std = mean, std
        self.train_stats_mean, self.train_stats_std = train_stats_mean, train_stats_std
        if(self.config.sampling.inference==True):
            assert mean is not None and std is not None, "Mean and std should not be provided for inference"
        

        if self.train_bool:
            assert self.train_data is not None, "Data must be provided for training"
            if os.path.exists(self.file_name):
                print(f"Loading existing model from {self.file_name}")
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
            if os.path.exists(self.file_name):
                print(f"Loading existing model from {self.file_name_distance}")
                self.distance_flow = eqx.tree_deserialise_leaves(self.file_name_distance, self.distance_flow)        
            print(f"Training model...")
            self.train()
            eqx.tree_serialise_leaves(self.file_name, self.flow)
            eqx.tree_serialise_leaves(self.file_name_distance, self.distance_flow)
        else:
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                self.distance_flow = eqx.tree_deserialise_leaves(self.file_name_distance, self.distance_flow)
            except FileNotFoundError:
                print(f"Model file not found: {self.file_name}. Training new model...")
                print(f"Model file not found: {self.file_name_distance}. Training new model...")
                assert self.train_data is not None, "Data must be provided for training"

                self.train()
                eqx.tree_serialise_leaves(self.file_name, self.flow)
                eqx.tree_serialise_leaves(self.file_name_distance, self.distance_flow)



