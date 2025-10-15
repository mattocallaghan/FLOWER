import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Literal
class StatisticEmbedding(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    conv3: eqx.nn.Conv1d
    global_pool: eqx.nn.AdaptiveAvgPool1d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    how: str  # 'MI' or 'vae'


    def __init__(self, *, key, in_channels=1,how='IM',hidden_scale=1 ,z_dim=None, dropout_rate=0.1):
        keys = jax.random.split(key, 11)

        self.conv1 = eqx.nn.Conv1d(in_channels, 4*hidden_scale, kernel_size=3, key=keys[0])
        self.conv2 = eqx.nn.Conv1d(4*hidden_scale,16*hidden_scale, kernel_size=3, key=keys[1])
        self.conv3 = eqx.nn.Conv1d(16*hidden_scale, 32*hidden_scale, kernel_size=3, key=keys[2])
        self.global_pool = eqx.nn.AdaptiveAvgPool1d(target_shape=1)
        self.fc1 = eqx.nn.Linear(32*hidden_scale, 32*hidden_scale, key=keys[5])
        self.how=how
        if(how=='IM'):
            self.fc2 = eqx.nn.Linear(32*hidden_scale, z_dim, key=keys[6])
        else:
            assert how=='vae', "how must be either 'MI' or 'vae'"
            self.fc2 = eqx.nn.Linear(32, z_dim*2, key=keys[6])

    def __call__(self, x, *, key=None, inference=False):
        """
        x: (in_channels, length), e.g. (1, 350)
        key: PRNGKey for dropout (split as needed)
        inference: if True, disables dropout
        """
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))
        x = jax.nn.gelu(self.conv3(x))
        x=self.global_pool(x).squeeze(-1)  # (batch, channels)
        x =jax.nn.gelu(self.fc1(x))
        z_params =(self.fc2(x))
        if(self.how=='vae'):
            mu, logvar = jnp.split(z_params, 2, axis=-1)
            std = jnp.exp(0.5 * logvar)
            if inference:
                z = mu  # Use mean during inference
                return z
            else:
                eps = jax.random.normal(key, shape=std.shape)
                z = mu + eps * std  # Reparameterization trick
                return z,mu,std
        else:
            return z_params  # For MI, return the parameters directly
        




class StatisticEmbedding_spectra(eqx.Module):
    conv1: eqx.nn.Conv1d
    conformer: eqx.Module
    fc_out: eqx.nn.Linear
    downsample:eqx.nn.AdaptiveAvgPool1d
    how: Literal["IM", "vae"]

    def __init__(
        self,
        *,
        key,
        in_channels=1,
        how="IM",
        hidden_scale=1,
        z_dim=32,
        dropout_rate=0.1,
    ):
        keys = jax.random.split(key, 6)

        self.conv1 = eqx.nn.Conv1d(in_channels, 64 * hidden_scale, kernel_size=3, key=keys[0])
        self.conformer = ConformerBlock(
            dim=64 * hidden_scale,
            dropout_rate=dropout_rate,
            key=keys[3],
        )
        self.downsample = eqx.nn.AdaptiveAvgPool1d(target_shape=64)
        self.fc_out = eqx.nn.Linear(64 * hidden_scale, z_dim if how == "IM" else z_dim * 2, key=keys[4])
        self.how = how

    def __call__(self, x, *, key=None, inference=False):
        keys = jax.random.split(key, 2) if key is not None else [None, None]

        # Convolutional feature extractor
        x = jax.nn.gelu(self.conv1(x))   # (C1, L)
        x = self.downsample(x)
        # Prepare for conformer: (C, L) → (L, C)
        x = x.T
        # Apply global attention modeling
        x = self.conformer(x, key=keys[0], inference=inference)
        x = x.T  # Back to (C, L)
        # Aggregate: mean over time
        x = jnp.mean(x, axis=1)  # (channels,)
        # Final projection
        z_params = self.fc_out(x)

        if self.how == "vae":
            mu, logvar = jnp.split(z_params, 2, axis=-1)
            std = jnp.exp(0.5 * logvar)
            if inference:
                return mu
            eps = jax.random.normal(keys[1], shape=std.shape)
            z = mu + eps * std
            return z, mu, std
        else:
            return z_params

class Swish(eqx.Module):
    def __call__(self, x,key=None):
        return jax.nn.swish(x)



class ConformerBlock(eqx.Module):
    ff1: eqx.nn.Sequential
    ff2: eqx.nn.Sequential
    mha: eqx.nn.MultiheadAttention
    conv_pw1: eqx.nn.Conv1d
    conv_dw: eqx.nn.Conv1d
    conv_pw2: eqx.nn.Conv1d
    ln: eqx.nn.LayerNorm
    dropout_rate: float

    def __init__(self, dim, ff_mult=2, num_heads=2, dropout_rate=0.1, *, key):
        k_ff1, k_ff2,key = jax.random.split(key, 3)

        self.ff1 = eqx.nn.Sequential([
            eqx.nn.Linear(dim, ff_mult * dim, key=k_ff1),
            Swish(),
            eqx.nn.Linear(ff_mult * dim, dim, key=k_ff2),
        ])
        k_ff1, k_mha, k_conv1, k_conv2, k_conv3, k_ff2,key = jax.random.split(key, 7)
        self.ff2 = eqx.nn.Sequential([
            eqx.nn.Linear(dim, ff_mult * dim, key=k_ff1),
            Swish(),
            eqx.nn.Linear(ff_mult * dim, dim, key=k_ff2),
        ])

        self.mha = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=dim,
            dropout_p=dropout_rate,
            inference=True,  # Turn off dropout by default
            key=k_mha
        )

        self.conv_pw1 = eqx.nn.Conv1d(dim, dim, kernel_size=1, key=k_conv1)
        self.conv_dw = eqx.nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1, key=k_conv2)
        self.conv_pw2 = eqx.nn.Conv1d(dim, dim, kernel_size=1, key=k_conv3)

        self.ln = eqx.nn.LayerNorm((dim,))
        self.dropout_rate = dropout_rate

    def __call__(self, x, *, key=None, inference=False):
        """
        x: (seq_len, dim)
        key: PRNGKey
        inference: disables dropout if True
        """
        dropout = lambda y,k: y if inference else eqx.nn.Dropout(self.dropout_rate)(y, key=k)
        keys = jax.random.split(key, 4) if key is not None else [None] * 4

        # Feedforward module (1st half)
        x_ff1 =  dropout((jax.vmap(self.ff1)(x)),keys[0])
        x = x + 0.5 * x_ff1  # residual
        # Multi-head attention: expects (seq_len, dim), no batch
        x_attn = dropout(self.mha(x, x, x, key=keys[1]), keys[1])
        x = x + x_attn  # residual

        # Conv path: Conv1d expects (channels, length), so transpose
        x_conv = x.T  # (dim, seq_len)
        x_conv = self.conv_pw1(x_conv)
        x_conv = self.conv_dw(x_conv)
        x_conv = jax.nn.swish(x_conv)
        x_conv = self.conv_pw2(x_conv)
        x = (x_conv + dropout(x_conv, keys[2])).T  # residual
        # Feedforward module (2nd half)
        x_ff2 = dropout(jax.vmap(self.ff2)(x), keys[3])
        x = x + 0.5 * x_ff2  # residual

        return jax.vmap(self.ln)(x)




class StatisticDecoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    deconv1: eqx.nn.ConvTranspose1d
    deconv2: eqx.nn.ConvTranspose1d
    deconv3: eqx.nn.ConvTranspose1d
    output_shape: int  # Final output length (e.g., 336)

    def __init__(self, *, key, latent_dim=5, out_channels=1, output_shape=200):
        keys = jax.random.split(key, 4)
        self.output_shape = output_shape

        # Linear to project latent vector to a suitable hidden shape
        self.fc1 = eqx.nn.Linear(latent_dim, 256, key=keys[0])
        self.fc2 = eqx.nn.Linear(256, 32 * (output_shape // 8), key=keys[0])

        
        # Deconv layers (reversed structure of encoder convs)
        self.deconv1 = eqx.nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, key=keys[1])
        self.deconv2 = eqx.nn.ConvTranspose1d(16, 4, kernel_size=4, stride=2, padding=1, key=keys[2])
        self.deconv3 = eqx.nn.ConvTranspose1d(4, out_channels, kernel_size=4, stride=2, padding=1, key=keys[3])

    def __call__(self,  z, theta, *, key,inference=False):
        """
        z: (latent_dim,) latent vector
        Returns: (out_channels, output_shape) reconstructed time-series
        """
        x = self.fc1(z)  # (5 * output_shape // 8,)
        x = jax.nn.gelu(x)
        x= self.fc2(x)  # (5, output_shape // 8)
        x = jax.nn.gelu(x)
        x = x.reshape(32, self.output_shape // 8)  # (channels, length)

        x = jax.nn.gelu(self.deconv1(x))  # (5, output_shape // 4)
        x = jax.nn.gelu(self.deconv2(x))  # (5, output_shape // 2)

        x = self.deconv3(x)               # (1, output_shape)

        return x

class Discriminator(eqx.Module):
    # original weights
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    
    #fc5: eqx.nn.Linear
    #fc6: eqx.nn.Linear
    #dropout: eqx.nn.Dropout



    def __init__(self, *, key, z_dim, theta_dim, hidden_dim=100, dropout_rate=0.1):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        input_dim = z_dim + theta_dim
        self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.fc3 = eqx.nn.Linear(hidden_dim, 1, key=k4)

        #self.fc5 = eqx.nn.Linear(z_dim, hidden_dim, key=k3)
        #self.fc6 = eqx.nn.Linear(hidden_dim, 2, key=k4)
        #self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, z, theta, *, key,inference=False):
        x = jnp.concatenate([z, theta], axis=-1)
        key1, key2 = (jax.random.split(key) if key is not None else (None, None))

        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        logit = self.fc3(x).squeeze(-1)
        return logit