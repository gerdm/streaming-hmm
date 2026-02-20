"""
Rao-Blackwellised Particle Filter (RBPF) for Gaussian HMM.

This module implements an online Rao-Blackwellised particle filter for regime
detection in Gaussian HMMs. Unlike the beam search approach in streaming_hmm.py,
RBPF uses:
- Regime sampling from the transition matrix (one regime per particle)
- Effective Sample Size (ESS) based resampling
- Rao-Blackwellisation: closed-form updates for continuous parameters

The API follows streaming_hmm.py for consistency.
"""

from typing import Any, Callable, Dict, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
import chex
import einops


# ============================================================
# Data Structures (from streaming_hmm)
# ============================================================

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


# ============================================================
# Particle Initialization
# ============================================================

def init_gauss_particles(
    key: jax.random.PRNGKey,
    mean_init: jax.Array,
    var_init: jax.Array,
    n_particles: int,
    n_regimes: int,
    n_steps: int,
) -> ParticleState:
    """
    Initialize particles for Gaussian observation model.

    Args:
        key: JAX random key.
        mean_init: Prior mean (D,).
        var_init: Prior variance (D,).
        n_particles: Number of particles.
        n_regimes: Number of regimes.
        n_steps: Number of time steps.

    Returns:
        Initialized ParticleState.
    """
    key_mean, key_regimes = jax.random.split(key)
    D = mean_init.shape[0]

    means = (
        mean_init
        + jax.random.normal(key_mean, (n_particles, n_regimes, D)) * jnp.sqrt(var_init)
    )
    variances = einops.repeat(var_init, "i -> s k i", s=n_particles, k=n_regimes)
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


# ============================================================
# Generic Utilities
# ============================================================

@jax.vmap
def build_weights(log_weights: jax.Array) -> jax.Array:
    """Convert log weights to normalized probability weights."""
    log_weights_norm = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(log_weights_norm)


# ============================================================
# RBPF Update Functions
# ============================================================

def gauss_rbpf_update(
    y: jax.Array,
    regime: jax.Array,
    bel: ParticleState,
    cfg: Cfg,
) -> Tuple[ParticleState, jax.Array]:
    """
    Bayesian update for 1-D Gaussian observation model (RBPF version).

    Args:
        y: Scalar observation.
        regime: Regime index (scalar int).
        bel: Current particle state (single particle, all regimes).
        cfg: Configuration (contains var_obs).

    Returns:
        (bel_updated, log_pp) where bel_updated has modified means/variances
        for the given regime.
    """
    mean = bel.means[regime]
    var = bel.variances[regime]

    yhat = mean
    pred_var = var + cfg.var_obs
    pred_std = jnp.sqrt(pred_var)
    log_pp = jax.scipy.stats.norm.logpdf(y, yhat, pred_std).squeeze()

    err = y - mean
    kt = var / pred_var
    mean_update = mean + kt * err
    var_update = (1 - kt) * var

    bel = bel.replace(
        means=bel.means.at[regime].set(mean_update),
        variances=bel.variances.at[regime].set(var_update),
    )
    return bel, log_pp


def multinomial_resampling(
    key: jax.random.PRNGKey,
    log_weights: jax.Array,
    n_particles: int,
) -> jax.Array:
    """Multinomial resampling based on log-weights."""
    indices = jax.random.categorical(key, log_weights, shape=(n_particles,))
    return indices


# ============================================================
# RBPF Step Function
# ============================================================

def rbpf_step(
    bel: ParticleState,
    obs_input: jax.Array,
    cfg: Cfg,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    key: jax.random.PRNGKey,
    ess_threshold: float = 0.5,
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """
    One step of RBPF inference with ESS-based resampling.

    Args:
        bel: Current particle state.
        obs_input: Current observation (scalar).
        cfg: Configuration.
        transition_matrix: Transition probability matrix (K, K).
        log_transition_matrix: Log transition matrix (K, K).
        key: JAX random key.
        ess_threshold: ESS threshold as fraction of n_particles (default 0.5).

    Returns:
        (bel_updated, (log_weights, mean_est, forecast, ess))
        - log_weights: Log importance weights (S,)
        - mean_est: Rao-Blackwellised regime mean estimates (K,)
        - forecast: Scalar one-step-ahead prediction accounting for regime transitions
        - ess: Effective Sample Size (scalar)
    """
    key_propagate, key_resample = jax.random.split(key)
    keys = jax.random.split(key_propagate, cfg.num_particles)

    # Regime sampling and updating (vmapped per particle)
    @jax.vmap
    def step_particle(key_i, bel_i):
        # Get current regime
        regime_curr = bel_i.regime[bel_i.timestep.astype(int)]
        log_p_transition = log_transition_matrix[regime_curr]

        # Sample next regime
        regime_next = jax.random.categorical(key_i, log_p_transition)

        # Update belief under sampled regime
        bel_update, log_pp = gauss_rbpf_update(obs_input, regime_next, bel_i, cfg)

        # Update regime and timestep
        timestep_new = bel_i.timestep + 1
        bel_update = bel_update.replace(
            regime=bel_update.regime.at[timestep_new.astype(int)].set(regime_next),
            timestep=timestep_new,
        )

        return bel_update, log_pp

    bel_updated, log_pp = step_particle(keys, bel)
    log_weights = log_pp + bel.log_weight

    # Compute effective sample size
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    ess = 1.0 / jnp.sum(weights ** 2)

    # Resample if ESS is low
    def resample_fn(_):
        indices = multinomial_resampling(key_resample, log_weights, cfg.num_particles)
        bel_resampled = jax.tree.map(lambda x: x[indices], bel_updated)
        bel_resampled = bel_resampled.replace(
            log_weight=jnp.full(cfg.num_particles, -jnp.log(cfg.num_particles))
        )
        return bel_resampled

    def no_resample_fn(_):
        return bel_updated.replace(log_weight=log_weights)

    bel_final = jax.lax.cond(
        ess < cfg.num_particles * ess_threshold,
        resample_fn,
        no_resample_fn,
        None,
    )

    # Compute Rao-Blackwellised mean estimate and one-step-ahead forecast
    weights_final = jnp.exp(bel_final.log_weight - jax.nn.logsumexp(bel_final.log_weight))
    
    # Mean estimate: Rao-Blackwellised estimate marginalizing over regime
    mean_est = jnp.einsum("s,sk...->k...", weights_final, bel_final.means)
    
    # One-step-ahead forecast accounting for regime transitions
    # For each particle with current regime z_s, marginalize over next regime k:
    # forecast = sum_s w_s * sum_k P(k | z_s) * mean_s_k
    
    # Get current regime for each particle using vmap over particles
    def get_current_regime(particle_idx):
        ts = bel_final.timestep[particle_idx].astype(int)
        ts_clamped = jnp.maximum(ts - 1, 0)
        return bel_final.regime[particle_idx, ts_clamped]
    
    regime_current = jax.vmap(get_current_regime)(jnp.arange(cfg.num_particles))  # (S,)
    
    # Get transition probabilities from current regime to all next regimes
    trans_probs = transition_matrix[regime_current]  # (S, K)
    
    # Compute forecast: weight particles and their regime transitions
    # Explicitly contract over s, k, and d dimensions
    # Shape: (S,) x (S, K) x (S, K, D) -> scalar
    forecast = jnp.einsum("s,sk,skd->", weights_final, trans_probs, bel_final.means)

    return bel_final, (bel_final.log_weight, mean_est, forecast, ess)


# ============================================================
# Main Run Function
# ============================================================

def run_rbpf_streaming_hmm(
    obs: jax.Array,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    key: jax.random.PRNGKey,
    n_particles: int = 100,
    var_obs: float = 1.0,
    mean_init: jax.Array = None,
    var_init: jax.Array = None,
    n_steps: Optional[int] = None,
    ess_threshold: float = 0.5,
    bel_init: Optional[ParticleState] = None,
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Run the RBPF streaming HMM algorithm on a sequence of observations.

    Args:
        obs: Observation array (T,) of scalar observations.
        transition_matrix: Transition probability matrix (K, K).
        log_transition_matrix: Log transition matrix (K, K).
        key: JAX random key.
        n_particles: Number of particles.
        var_obs: Observation variance.
        mean_init: Initial mean estimate (default: [0.0]).
        var_init: Initial variance estimate (default: [1.0]).
        n_steps: Number of time steps (inferred from obs if None).
        ess_threshold: ESS threshold as fraction of n_particles (default 0.5).
        bel_init: Pre-initialized particle state (optional).

    Returns:
        (bel_final, (hist_lw, hist_means, hist_forecast, hist_ess), hist_weights)
        - hist_lw: Log weights history (T, S)
        - hist_means: Rao-Blackwellised mean estimates (T, K)
        - hist_forecast: One-step-ahead forecasts (T,)
        - hist_ess: Effective Sample Size history (T,)
    """
    n_steps = obs.shape[0]
    num_regimes = transition_matrix.shape[0]

    if mean_init is None:
        mean_init = jnp.array([0.0])
    if var_init is None:
        var_init = jnp.array([1.0])

    if bel_init is None:
        key_init, _ = jax.random.split(key)
        bel_init = init_gauss_particles(
            key_init, mean_init, var_init, n_particles, num_regimes, n_steps
        )

    cfg = Cfg(var_obs=var_obs, num_particles=n_particles, num_regimes=num_regimes)

    # Generate random keys for each timestep
    keys = jax.random.split(key, n_steps)

    _step = partial(
        rbpf_step,
        cfg=cfg,
        transition_matrix=transition_matrix,
        log_transition_matrix=log_transition_matrix,
        ess_threshold=ess_threshold,
    )

    bel_final, (hist_lw, hist_means, hist_forecast, hist_ess) = jax.lax.scan(
        lambda bel, x: _step(bel, x[0], key=x[1]),
        bel_init,
        (obs, keys),
    )

    hist_weights = build_weights(hist_lw)

    return bel_final, (hist_lw, hist_means, hist_forecast, hist_ess), hist_weights
