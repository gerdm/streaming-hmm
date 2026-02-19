"""
Streaming Hidden Markov Model (HMM) implementation with Gaussian observations.

This module implements an online Bayesian HMM for regime detection and prediction,
using beam search to maintain top-K particles.
"""

from typing import Callable, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import chex
import einops


@chex.dataclass
class Cfg:
    """Configuration for the streaming HMM."""
    var_obs: float
    num_particles: int
    num_regimes: int


@chex.dataclass
class ParticleState:
    """State representation for particle filtering in streaming HMM."""
    means: jax.Array
    variances: jax.Array
    regime: jax.Array
    log_weight: jax.Array
    timestep: jax.Array

    @staticmethod
    def init(
        key: jax.random.PRNGKey,
        mean: jax.Array,
        cov: jax.Array,
        n_particles: int,
        n_regimes: int,
        n_steps: int
    ) -> "ParticleState":
        """
        Initialize particle state.
        
        Args:
            key: JAX random key
            mean: Initial mean estimate
            cov: Initial covariance estimate
            n_particles: Number of particles
            n_regimes: Number of regimes
            n_steps: Number of time steps
            
        Returns:
            Initialized ParticleState
        """
        key_mean, key_regimes = jax.random.split(key)

        means = jax.random.normal(key_mean, (n_particles, n_regimes, 1)) * jnp.sqrt(cov)
        variances = einops.repeat(cov, "i -> s k i", s=n_particles, k=n_regimes)
        log_weights = jnp.full(n_particles, -jnp.log(n_particles))
        timestep = jnp.zeros(n_particles)

        regimes = jnp.zeros((n_particles, n_steps)).astype(int)
        regimes_init = jax.random.choice(key_regimes, n_regimes, (n_particles,)).astype(int)
        regimes = regimes.at[:, 0].set(regimes_init)
        
        return ParticleState(
            means=means, 
            variances=variances,
            regime=regimes,
            log_weight=log_weights,
            timestep=timestep,
        )


def flatten_fn(fn: Callable) -> Callable:
    """
    Decorator to flatten particle dimensions.
    
    Args:
        fn: Function to wrap
        
    Returns:
        Wrapped function that flattens particles
    """
    def flatten_particles(tree):
        """
        Given any pytree with leaf shapes (particles options ...),
        we stack the leaf to be (particle options) ...
        """
        einops_expr = "particles options ... -> (particles options) ..."
        res = jax.tree.map(lambda x: einops.rearrange(x, einops_expr), tree)
        return res
    
    return lambda *x: flatten_particles(fn(*x))


@jax.vmap
def build_weights(log_weights: jax.Array) -> jax.Array:
    """
    Convert log weights to normalized weights.
    
    Args:
        log_weights: Log-space weights
        
    Returns:
        Normalized weights in probability space
    """
    log_weights_norm = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(log_weights_norm)


def update_gauss(
    y: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    cfg: Cfg
) -> Tuple[jax.Array, jax.Array]:
    """
    Perform Kalman filter update for Gaussian observation model.
    
    Args:
        y: Observation
        mean: Current mean estimate
        var: Current variance estimate
        cfg: Configuration
        
    Returns:
        Tuple of (updated mean, updated variance)
    """
    err = y - mean
    kt = var / (var + cfg.var_obs)
    mean_update = mean + kt * err
    var_update = (1 - kt) * var
    
    return mean_update, var_update


@flatten_fn
@partial(jax.vmap, in_axes=(None, None, 0, None, None))
@partial(jax.vmap, in_axes=(None, 0, None, None, None))
def update_conditional_posterior(
    y: jax.Array,
    regime: jax.Array,
    bel: ParticleState,
    cfg: Cfg,
    log_transition_matrix: jax.Array
) -> Tuple[ParticleState, jax.Array, jax.Array]:
    """
    Update model parameters for a regime using posterior inference.
    
    Args:
        y: Current observation
        regime: Regime index
        bel: Current particle state
        cfg: Configuration
        log_transition_matrix: Log transition probabilities
        
    Returns:
        Tuple of (updated belief, log predictive probability, log transition probability)
    """
    yhat = mean = bel.means[regime]
    var = bel.variances[regime]

    mean_update, var_update = update_gauss(y, mean, var, cfg)

    pred_sttdev = jnp.sqrt(var + cfg.var_obs)
    log_pp = jax.scipy.stats.norm.logpdf(y, yhat, pred_sttdev).squeeze()

    regime_curr = bel.regime[bel.timestep.astype(int)]
    log_p_transition = log_transition_matrix[regime_curr, regime]

    timestep_new = bel.timestep + 1
    bel = bel.replace(
        means=bel.means.at[regime].set(mean_update),
        variances=bel.variances.at[regime].set(var_update),
        regime=bel.regime.at[timestep_new.astype(int)].set(regime),
        timestep=timestep_new,
    )

    return bel, log_pp, log_p_transition


@jax.vmap
def update_log_weights(
    bel: ParticleState,
    log_pp: jax.Array,
    log_p_transition: jax.Array
) -> ParticleState:
    """
    Update beliefs for all regimes by updating log weights.
    
    Args:
        bel: Current particle state
        log_pp: Log predictive probabilities
        log_p_transition: Log transition probabilities
        
    Returns:
        Updated particle state with new log weights
    """
    log_weight = log_pp + log_p_transition + bel.log_weight

    return bel.replace(
        log_weight=log_weight
    )


def beam_search(bel: ParticleState, K: int) -> ParticleState:
    """
    Maintain top K particles with highest weight and normalize.
    
    Args:
        bel: Current particle state
        K: Number of particles to keep
        
    Returns:
        Pruned and normalized particle state
    """
    log_weights = bel.log_weight
    indices = jnp.argsort(log_weights, descending=True)[:K]
    
    bel = jax.tree.map(lambda x: x[indices], bel)

    log_weights = log_weights[indices]
    log_weights = log_weights - jax.nn.logsumexp(log_weights)
        
    return bel.replace(
        log_weight=log_weights
    )


def forecast(
    bel: ParticleState,
    cfg: Cfg,
    transition_matrix: jax.Array
) -> Dict[str, jax.Array]:
    """
    Compute posterior predictive distribution.
    
    Args:
        bel: Current particle state
        cfg: Configuration
        transition_matrix: Transition probability matrix
        
    Returns:
        Dictionary containing:
            - mean: Posterior predictive mean
            - stdev_obs: Posterior predictive standard deviation
            - stdev_param: Posterior standard deviation over parameters
    """
    mean = bel.means
    variances = bel.variances
    weights = bel.log_weight
    weights = jnp.exp(weights - jax.nn.logsumexp(weights))
    timestep = bel.timestep.astype(int)[0]
    regime = bel.regime[:, timestep] 

    p_transition = transition_matrix.at[regime].get()

    yhat = jnp.einsum("s,sk,sk...->", weights, p_transition, mean)
    mean2 = jnp.einsum("s,sk,sk...->", weights, p_transition, mean ** 2 + variances)
    yhat2 = mean2 + cfg.var_obs

    # posterior predictive standard deviation
    yhat_std = jnp.sqrt(yhat2 - yhat ** 2)
    # posterior standard deviation
    mean_std = jnp.sqrt(mean2 - yhat ** 2)

    out = {
        "mean": yhat,
        "stdev_obs": yhat_std,
        "stdev_param": mean_std,
    }
    
    return out


def step(
    bel: ParticleState,
    y: jax.Array,
    cfg: Cfg,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]]:
    """
    Perform one step of streaming HMM inference.
    
    Args:
        bel: Current particle state
        y: Current observation
        cfg: Configuration
        transition_matrix: Transition probability matrix
        log_transition_matrix: Log transition probability matrix
        
    Returns:
        Tuple of (updated belief, (log_weight, means, variances, forecast))
    """
    regimes = jnp.arange(cfg.num_regimes)
    
    # Forecast estimate before seeing next observation
    fcst = forecast(bel, cfg, transition_matrix)

    # Update mean and variance of each particle under all regimes [S -> SxK]
    bel_update, log_pp, log_p_transition = update_conditional_posterior(
        y, regimes, bel, cfg, log_transition_matrix
    )

    # Update log-weight of all particles [SxK -> SxK]
    bel_update = update_log_weights(bel_update, log_pp, log_p_transition)

    # Choose top K particles according to log-weights [SxK -> S]
    bel_update = beam_search(bel_update, cfg.num_particles) 

    return bel_update, (bel_update.log_weight, bel_update.means, bel_update.variances, fcst)


def run_streaming_hmm(
    obs: jax.Array,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    key: jax.random.PRNGKey,
    n_particles: int = 5,
    var_obs: float = 1.0,
    mean_init: jax.Array = None,
    var_init: jax.Array = None,
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]], jax.Array]:
    """
    Run the streaming HMM algorithm on a sequence of observations.
    
    Args:
        obs: Observation sequence (n_steps,)
        transition_matrix: Transition probability matrix (n_regimes, n_regimes)
        log_transition_matrix: Log transition probability matrix (n_regimes, n_regimes)
        key: JAX random key
        n_particles: Number of particles for beam search
        var_obs: Observation variance
        mean_init: Initial mean estimate (if None, defaults to [0.0])
        var_init: Initial variance estimate (if None, defaults to [1.0])
        
    Returns:
        Tuple of:
            - bel_final: Final particle state
            - history: Tuple of (log_weights, means, variances, forecasts) over time
            - hist_weights: Normalized weights over time
    """
    n_steps = len(obs)
    num_regimes = transition_matrix.shape[0]
    
    # Set defaults
    if mean_init is None:
        mean_init = jnp.array([0.0])
    if var_init is None:
        var_init = jnp.array([1.0])
    
    # Initialize particle state
    bel_init = ParticleState.init(key, mean_init, var_init, n_particles, num_regimes, n_steps)
    
    # Create configuration
    cfg = Cfg(var_obs=var_obs, num_particles=n_particles, num_regimes=num_regimes)
    
    # Define step function with fixed parameters
    _step = partial(step, cfg=cfg, transition_matrix=transition_matrix, 
                    log_transition_matrix=log_transition_matrix)
    
    # Run the HMM
    bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast) = jax.lax.scan(_step, bel_init, obs)
    
    # Build normalized weights
    hist_weights = build_weights(hist_lw)
    
    return bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast), hist_weights
