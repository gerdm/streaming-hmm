"""
Streaming Hidden Markov Model (HMM) implementation.

This module implements an online Bayesian HMM for regime detection and prediction,
using beam search to maintain top-K particles. The implementation is modular,
supporting different observation models (Gaussian, AR, GP) through pluggable
update and forecast functions.
"""

from typing import Any, Callable, Dict, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
import chex
import einops


# ============================================================
# Core Data Structures
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

    @staticmethod
    def init(
        key: jax.random.PRNGKey,
        mean: jax.Array,
        cov: jax.Array,
        n_particles: int,
        n_regimes: int,
        n_steps: int,
    ) -> "ParticleState":
        """
        Initialize particle state for the Gaussian observation model.
        Kept for backward compatibility; prefer init_gauss_particles.
        """
        return init_gauss_particles(key, mean, cov, n_particles, n_regimes, n_steps)


@chex.dataclass
class GPParticleState:
    """
    State representation for GP-based particle filtering in streaming HMM.
    
    Each particle maintains FIFO buffers per regime, storing (X, y) pairs
    for Gaussian Process regression.
    
    Note: buffer_size is inferred from X_buffers.shape[2].
    """
    X_buffers: jax.Array      # (S, K, buffer_size, dim_in)
    y_buffers: jax.Array      # (S, K, buffer_size)
    counters: jax.Array       # (S, K, buffer_size) - which slots are filled
    num_obs: jax.Array        # (S, K) - number of observations per regime per particle
    regime: jax.Array         # (S, n_steps)
    log_weight: jax.Array     # (S,)
    timestep: jax.Array       # (S,)


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
        mean_init: Prior mean (D,). Used to center random initialization.
        var_init: Prior variance (D,) — diagonal.
        n_particles: Number of particles.
        n_regimes: Number of regimes.
        n_steps: Number of time steps.

    Returns:
        Initialized ParticleState with
            means     (S, K, D)
            variances (S, K, D)
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


def init_ar_particles(
    key: jax.random.PRNGKey,
    mean_init: jax.Array,
    cov_init: jax.Array,
    n_particles: int,
    n_regimes: int,
    n_steps: int,
) -> ParticleState:
    """
    Initialize particles for an AR observation model.

    Args:
        key: JAX random key.
        mean_init: Prior mean for AR coefficients (D,).
        cov_init: Prior covariance for AR coefficients (D, D).
        n_particles: Number of particles.
        n_regimes: Number of regimes.
        n_steps: Number of time steps.

    Returns:
        Initialized ParticleState with
            means     (S, K, D)
            variances (S, K, D, D)
    """
    key_mean, key_regimes = jax.random.split(key)

    means = jax.random.multivariate_normal(
        key_mean, mean_init, cov_init, (n_particles, n_regimes)
    )
    variances = einops.repeat(cov_init, "d1 d2 -> s k d1 d2", s=n_particles, k=n_regimes)
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

def flatten_fn(fn: Callable) -> Callable:
    """Decorator to flatten (particles, options, ...) -> (particles*options, ...)."""
    def flatten_particles(tree: Any) -> Any:
        einops_expr = "particles options ... -> (particles options) ..."
        return jax.tree.map(lambda x: einops.rearrange(x, einops_expr), tree)
    return lambda *x: flatten_particles(fn(*x))


@jax.vmap
def build_weights(log_weights: jax.Array) -> jax.Array:
    """Convert log weights to normalized probability weights."""
    log_weights_norm = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(log_weights_norm)


@jax.vmap
def update_log_weights(
    bel: ParticleState,
    log_pp: jax.Array,
    log_p_transition: jax.Array,
) -> ParticleState:
    """Update particle log-weights."""
    log_weight = log_pp + log_p_transition + bel.log_weight
    return bel.replace(log_weight=log_weight)


def beam_search(bel: ParticleState, K: int) -> ParticleState:
    """Retain top-K particles by weight and renormalize."""
    log_weights = bel.log_weight
    indices = jnp.argsort(log_weights, descending=True)[:K]
    bel = jax.tree.map(lambda x: x[indices], bel)
    log_weights = log_weights[indices]
    log_weights = log_weights - jax.nn.logsumexp(log_weights)
    return bel.replace(log_weight=log_weights)


# ============================================================
# Update Conditional Posterior (Factory)
# ============================================================

def make_update_conditional_posterior(
    update_fn: Callable,
) -> Callable:
    """
    Factory that wraps a model-specific *update_fn* into a fully-
    vmapped ``update_conditional_posterior`` function.

    The *update_fn* should have the signature::

        (obs_input, regime, bel, cfg) -> (bel_updated, log_pp)

    where *bel_updated* has modified ``means`` / ``variances`` but
    untouched ``regime`` and ``timestep`` fields (those are handled here).

    Args:
        update_fn: Model-specific parameter update function.

    Returns:
        A function with signature::

            (obs_input, regimes, bel, cfg, log_transition_matrix)
            -> (bel, log_pp, log_p_transition)

        that is vmapped over (particles x regimes) and flattened.
    """
    @flatten_fn
    @partial(jax.vmap, in_axes=(None, None, 0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def _update_conditional_posterior(
        obs_input: Any,
        regime: jax.Array,
        bel: ParticleState,
        cfg: Cfg,
        log_transition_matrix: jax.Array,
    ) -> Tuple[ParticleState, jax.Array, jax.Array]:
        # Model-specific parameter update + log predictive probability
        bel_update, log_pp = update_fn(obs_input, regime, bel, cfg)

        # Transition probability from current regime to proposed regime
        regime_curr = bel.regime[bel.timestep.astype(int)]
        log_p_transition = log_transition_matrix[regime_curr, regime]

        # Book-keeping: record proposed regime and advance timestep
        timestep_new = bel.timestep + 1
        bel_update = bel_update.replace(
            regime=bel_update.regime.at[timestep_new.astype(int)].set(regime),
            timestep=timestep_new,
        )

        return bel_update, log_pp, log_p_transition

    return _update_conditional_posterior


# ============================================================
# Gaussian Observation Model
# ============================================================

def gauss_update_fn(
    obs_input: jax.Array,
    regime: jax.Array,
    bel: ParticleState,
    cfg: Cfg,
) -> Tuple[ParticleState, jax.Array]:
    """
    Bayesian update for a 1-D Gaussian observation model
    (conjugate Gaussian-Gaussian).

    Args:
        obs_input: Scalar observation *y*.
        regime: Regime index (scalar int).
        bel: Current particle state (single particle, all regimes).
        cfg: Configuration (contains ``var_obs``).

    Returns:
        ``(bel_updated, log_pp)`` where *bel_updated* has modified
        ``means`` and ``variances`` for the given *regime*.
    """
    y = obs_input
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


def gauss_forecast_fn(
    bel: ParticleState,
    cfg: Cfg,
    transition_matrix: jax.Array,
    obs_input: Any,
) -> Dict[str, jax.Array]:
    """
    Posterior predictive distribution for the Gaussian observation model.

    Args:
        bel: Current particle state.
        cfg: Configuration.
        transition_matrix: Transition probability matrix (K, K).
        obs_input: (unused for the Gaussian model).

    Returns:
        Dict with keys ``mean``, ``stdev_obs``, ``stdev_param``.
    """
    mean = bel.means
    variances = bel.variances
    weights = jnp.exp(bel.log_weight - jax.nn.logsumexp(bel.log_weight))
    timestep = bel.timestep.astype(int)[0]
    regime = bel.regime[:, timestep]

    p_transition = transition_matrix.at[regime].get()

    yhat = jnp.einsum("s,sk,sk...->", weights, p_transition, mean)
    mean2 = jnp.einsum("s,sk,sk...->", weights, p_transition, mean ** 2 + variances)
    yhat2 = mean2 + cfg.var_obs

    yhat_std = jnp.sqrt(yhat2 - yhat ** 2)
    mean_std = jnp.sqrt(mean2 - yhat ** 2)

    return {"mean": yhat, "stdev_obs": yhat_std, "stdev_param": mean_std}


# ============================================================
# AR Observation Model
# ============================================================

def ar_update_fn(
    obs_input: Tuple[jax.Array, jax.Array],
    regime: jax.Array,
    bel: ParticleState,
    cfg: Cfg,
) -> Tuple[ParticleState, jax.Array]:
    """
    Bayesian update for an AR(p) observation model (Bayesian linear
    regression with known variance).

    Args:
        obs_input: ``(y, x)`` where *y* is the scalar observation and
            *x* is the feature vector ``[y_{t-1}, ..., y_{t-p}]`` of shape (D,).
        regime: Regime index (scalar int).
        bel: Current particle state.
        cfg: Configuration (``var_obs`` is the known noise variance).

    Returns:
        ``(bel_updated, log_pp)``.
    """
    y, x = obs_input
    mean = bel.means[regime]       # (D,)
    cov = bel.variances[regime]    # (D, D)

    # Predictive distribution
    yhat = x @ mean                              # scalar
    pred_var = x @ cov @ x + cfg.var_obs          # scalar
    pred_std = jnp.sqrt(pred_var)
    log_pp = jax.scipy.stats.norm.logpdf(y, yhat, pred_std).squeeze()

    # Kalman update (Bayesian linear regression)
    K = cov @ x / pred_var                        # (D,)
    err = y - yhat
    mean_update = mean + K * err                  # (D,)
    cov_update = cov - jnp.outer(K, x) @ cov     # (D, D)

    bel = bel.replace(
        means=bel.means.at[regime].set(mean_update),
        variances=bel.variances.at[regime].set(cov_update),
    )
    return bel, log_pp


def ar_forecast_fn(
    bel: ParticleState,
    cfg: Cfg,
    transition_matrix: jax.Array,
    obs_input: Tuple[jax.Array, jax.Array],
) -> Dict[str, jax.Array]:
    """
    Posterior predictive distribution for the AR observation model.

    Args:
        bel: Current particle state.
        cfg: Configuration.
        transition_matrix: Transition probability matrix (K, K).
        obs_input: ``(y, x)`` — only *x* (feature vector) is used.

    Returns:
        Dict with keys ``mean``, ``stdev_obs``, ``stdev_param``.
    """
    _, x = obs_input
    mean = bel.means         # (S, K, D)
    cov = bel.variances      # (S, K, D, D)
    weights = jnp.exp(bel.log_weight - jax.nn.logsumexp(bel.log_weight))
    timestep = bel.timestep.astype(int)[0]
    regime = bel.regime[:, timestep]

    p_transition = transition_matrix.at[regime].get()  # (S, K)

    # Predicted mean per (particle, regime): x^T m_{s,k}
    yhats = jnp.einsum("d,skd->sk", x, mean)          # (S, K)

    # Overall predicted mean
    yhat = jnp.einsum("s,sk,sk->", weights, p_transition, yhats)

    # Parameter variance component: x^T Sigma_{s,k} x
    param_var = jnp.einsum("d,skde,e->sk", x, cov, x)  # (S, K)

    # Second moment
    mean2 = jnp.einsum("s,sk,sk->", weights, p_transition, yhats ** 2 + param_var)
    yhat2 = mean2 + cfg.var_obs

    yhat_std = jnp.sqrt(jnp.maximum(yhat2 - yhat ** 2, 0.0))
    mean_std = jnp.sqrt(jnp.maximum(mean2 - yhat ** 2, 0.0))

    return {"mean": yhat, "stdev_obs": yhat_std, "stdev_param": mean_std}


# ============================================================
# Gaussian Process (GP) Observation Model 
# ============================================================

def gaussian_kernel(sigma2: float) -> Callable:
    """
    Gaussian (RBF) kernel for GP regression.

    Args:
        sigma2: Length scale parameter.

    Returns:
        A kernel function K(X1, X2) -> (n1, n2) array.
    """
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def kernel(u: jax.Array, v: jax.Array) -> jax.Array:
        k = jnp.exp(-(u - v) ** 2 / (2 * sigma2))
        return k.squeeze()

    return kernel


def periodic_kernel(length_scale: float, period: float, amplitude: float = 1.0) -> Callable:
    """
    Periodic (exponential sine squared) kernel for modeling oscillatory patterns.

    The kernel is:
        k(x, y) = amplitude^2 * exp(-2*sin^2(pi*|x-y|/period) / length_scale^2)

    Args:
        length_scale: Controls smoothness of oscillations.
        period: The period of the oscillation.
        amplitude: Kernel amplitude (default: 1.0).

    Returns:
        A kernel function K(X1, X2) -> (n1, n2) array.
    """
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def kernel(u: jax.Array, v: jax.Array) -> jax.Array:
        dist = jnp.abs(u - v)
        sin_dist = jnp.sin(jnp.pi * dist / period)
        k = amplitude ** 2 * jnp.exp(-2 * sin_dist ** 2 / (length_scale ** 2))
        return k.squeeze()

    return kernel


def init_gp_particles(
    key: jax.random.PRNGKey,
    dim_in: int,
    buffer_size: int,
    n_particles: int,
    n_regimes: int,
    n_steps: int,
) -> GPParticleState:
    """
    Initialize particles for a GP observation model.

    Args:
        key: JAX random key.
        dim_in: Dimensionality of input features (typically 1 for time).
        buffer_size: Size of FIFO buffer per regime.
        n_particles: Number of particles.
        n_regimes: Number of regimes.
        n_steps: Number of time steps.

    Returns:
        Initialized GPParticleState with empty buffers.
    """
    key_regimes, _ = jax.random.split(key)

    X_buffers = jnp.zeros((n_particles, n_regimes, buffer_size, dim_in))
    y_buffers = jnp.zeros((n_particles, n_regimes, buffer_size))
    counters = jnp.zeros((n_particles, n_regimes, buffer_size))
    num_obs = jnp.zeros((n_particles, n_regimes), dtype=jnp.int32)
    log_weights = jnp.full(n_particles, -jnp.log(n_particles))
    timestep = jnp.zeros(n_particles)

    regimes = jnp.zeros((n_particles, n_steps)).astype(int)
    # regimes_init = jax.random.choice(key_regimes, n_regimes, (n_particles,)).astype(int)
    # regimes = regimes.at[:, 0].set(regimes_init)

    return GPParticleState(
        X_buffers=X_buffers,
        y_buffers=y_buffers,
        counters=counters,
        num_obs=num_obs,
        regime=regimes,
        log_weight=log_weights,
        timestep=timestep,
    )


def _gp_mask_matrix(counter: jax.Array, A: jax.Array) -> jax.Array:
    """
    Set rows and columns of matrix A to zero where counter == 0.
    """
    mask = jnp.where(counter == 0, size=len(counter), fill_value=jnp.nan)[0]
    mask = jnp.nan_to_num(mask, nan=mask[0]).astype(int)
    A_masked = A.at[mask].set(0.0).at[:, mask].set(0.0)
    return A_masked


def _gp_build_kernel_matrices(
    X_train: jax.Array,
    x_test: jax.Array,
    counter: jax.Array,
    kernel: Callable,
    obs_variance: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Build kernel matrices for GP predictive distribution.
    """
    is_full_buffer = jnp.all(counter == 1)

    var_train = kernel(X_train, X_train)
    var_train_diag = jnp.diag(var_train) + obs_variance
    var_train = var_train.at[jnp.diag_indices_from(var_train)].set(var_train_diag)

    var_train = jax.lax.cond(
        is_full_buffer,
        lambda: var_train,
        lambda: _gp_mask_matrix(counter, var_train)
    )

    cov_test_train = kernel(x_test, X_train)
    var_test = kernel(x_test, x_test)

    return cov_test_train, var_train, var_test


def _gp_predictive(
    X_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    counter: jax.Array,
    kernel: Callable,
    obs_variance: float,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute GP predictive mean and variance.
    """
    x_test = jnp.atleast_2d(x_test)
    cov_test_train, var_train, var_test = _gp_build_kernel_matrices(
        X_train, x_test, counter, kernel, obs_variance
    )
    K = jnp.linalg.lstsq(var_train, cov_test_train.T)[0].T
    is_empty = jnp.all(counter == 0.0)

    # Posterior predictive mean
    mu_pred = jax.lax.cond(
        is_empty,
        lambda: jnp.zeros(len(x_test)),
        lambda: K @ y_train
    )

    # Posterior predictive variance
    cov_pred = jax.lax.cond(
        is_empty,
        lambda: var_test,
        lambda: var_test - jnp.einsum("ij,jk,lk->il", K, var_train, K, precision="highest")
    )

    return mu_pred, cov_pred


def make_gp_update_fn(
    kernel: Callable,
) -> Callable:
    """
    Factory for creating a GP update function.

    Args:
        kernel: A kernel function K(X1, X2) -> (n1, n2) array.

    Returns:
        An update function compatible with make_update_conditional_posterior.
    """
    def gp_update_fn(
        obs_input: Tuple[jax.Array, jax.Array],
        regime: jax.Array,
        bel: GPParticleState,
        cfg: Cfg,
    ) -> Tuple[GPParticleState, jax.Array]:
        """
        Bayesian update for GP observation model.

        Args:
            obs_input: (y, x) where y is the observation and x is the input (e.g., time).
            regime: Regime index.
            bel: Current GP particle state.
            cfg: Configuration.

        Returns:
            (bel_updated, log_pp).
        """
        y, x = obs_input
        x = jnp.atleast_1d(x)

        # Get current buffer for this regime
        X_train = bel.X_buffers[regime]
        y_train = bel.y_buffers[regime]
        counter = bel.counters[regime]
        n_obs = bel.num_obs[regime]
        buffer_size = X_train.shape[0]

        # Compute log predictive probability
        mu_pred, cov_pred = _gp_predictive(
            X_train, y_train, x, counter, kernel, cfg.var_obs
        )
        print("Reload with debug prints")
        pred_var = jnp.maximum(cov_pred[0, 0], 1e-6)
        pred_std = jnp.sqrt(pred_var)
        log_pp = jax.scipy.stats.norm.logpdf(y, mu_pred[0], pred_std)

        # Update buffer (FIFO)
        ix_buffer = n_obs % buffer_size
        X_train_new = X_train.at[ix_buffer].set(x)
        y_train_new = y_train.at[ix_buffer].set(y)
        counter_new = counter.at[ix_buffer].set(1.0)
        n_obs_new = n_obs + 1

        # Update belief
        bel = bel.replace(
            X_buffers=bel.X_buffers.at[regime].set(X_train_new),
            y_buffers=bel.y_buffers.at[regime].set(y_train_new),
            counters=bel.counters.at[regime].set(counter_new),
            num_obs=bel.num_obs.at[regime].set(n_obs_new),
        )
        return bel, log_pp

    return gp_update_fn


def make_gp_forecast_fn(
    kernel: Callable,
    n_ahead: int = 1,
    dt: float = 1.0,
) -> Callable:
    """
    Factory for creating a GP forecast function.

    Args:
        kernel: A kernel function K(X1, X2) -> (n1, n2) array.
        n_ahead: Number of steps ahead to forecast (default: 1).
        dt: Time step between forecast points (default: 1.0).

    Returns:
        A forecast function compatible with run_streaming_hmm.
    """
    def gp_forecast_fn(
        bel: GPParticleState,
        cfg: Cfg,
        transition_matrix: jax.Array,
        obs_input: Tuple[jax.Array, jax.Array],
    ) -> Dict[str, jax.Array]:
        """
        Posterior predictive for GP observation model with multi-step-ahead forecasts.
        
        For h-step ahead forecasts, uses h-step transition probabilities Π^h.
        
        Approximation: Does not marginalize over intermediate observations y(t+1:t+h-1),
        which is computationally intractable (would require K^h regime paths and
        integration over continuous intermediate observations).
        """
        _, x = obs_input
        x = jnp.atleast_1d(x)

        # Create test points for multi-step-ahead prediction
        x_test = x + jnp.arange(n_ahead).reshape(-1, 1) * dt

        weights = jnp.exp(bel.log_weight - jax.nn.logsumexp(bel.log_weight))
        timestep = bel.timestep.astype(int)[0]
        regime = bel.regime[:, timestep]

        # Compute h-step transition probabilities: Π^h for h=1,2,...,n_ahead
        # Shape: (n_ahead, n_regimes, n_regimes)
        transition_powers = jnp.stack([
            jnp.linalg.matrix_power(transition_matrix, h) 
            for h in range(1, n_ahead + 1)
        ])

        # Compute predictive for each particle and regime
        def particle_regime_pred(s_k):
            s, k = s_k
            X_train = bel.X_buffers[s, k]
            y_train = bel.y_buffers[s, k]
            counter = bel.counters[s, k]

            mu, cov = _gp_predictive(
                X_train, y_train, x_test, counter, kernel, cfg.var_obs
            )
            return mu, jnp.diag(cov)

        # Grid over particles and regimes
        n_particles = bel.log_weight.shape[0]
        n_regimes = cfg.num_regimes
        particle_idxs = jnp.repeat(jnp.arange(n_particles), n_regimes)
        regime_idxs = jnp.tile(jnp.arange(n_regimes), n_particles)

        mus, vars_ = jax.vmap(particle_regime_pred)((particle_idxs, regime_idxs))
        mus = mus.reshape(n_particles, n_regimes, n_ahead)
        vars_ = vars_.reshape(n_particles, n_regimes, n_ahead)

        # Extract h-step transition probabilities for each particle's current regime
        # p_transition_h[s, h, k] = P(s_{t+h}=k | s_t=regime[s])
        p_transition_h = transition_powers[:, regime, :]  # (n_ahead, n_particles, n_regimes)
        p_transition_h = jnp.transpose(p_transition_h, (1, 2, 0))  # (n_particles, n_regimes, n_ahead)

        # Weighted average using h-step transition probabilities
        yhat = jnp.einsum("s,skh,skh->h", weights, p_transition_h, mus)
        mean2 = jnp.einsum("s,skh,skh->h", weights, p_transition_h, mus ** 2 + vars_)
        yhat2 = mean2 + cfg.var_obs

        yhat_std = jnp.sqrt(jnp.maximum(yhat2 - yhat ** 2, 0.0))
        mean_std = jnp.sqrt(jnp.maximum(mean2 - yhat ** 2, 0.0))

        return {
            "mean": yhat,
            "stdev_obs": yhat_std,
            "stdev_param": mean_std,
            "x_test": x_test.squeeze(),
        }

    return gp_forecast_fn


def make_gp_update_conditional_posterior(
    kernel: Callable,
) -> Callable:
    """
    Factory that creates a fully-vmapped update_conditional_posterior
    for the GP observation model.

    The GP model requires special handling because it uses GPParticleState
    instead of ParticleState.

    Args:
        kernel: A kernel function K(X1, X2) -> (n1, n2) array.

    Returns:
        A function with signature::

            (obs_input, regimes, bel, cfg, log_transition_matrix)
            -> (bel, log_pp, log_p_transition)
    """
    gp_update_fn = make_gp_update_fn(kernel)

    @flatten_fn
    @partial(jax.vmap, in_axes=(None, None, 0, None, None))
    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def _update_conditional_posterior(
        obs_input: Any,
        regime: jax.Array,
        bel: GPParticleState,
        cfg: Cfg,
        log_transition_matrix: jax.Array,
    ) -> Tuple[GPParticleState, jax.Array, jax.Array]:
        # Model-specific parameter update + log predictive probability
        bel_update, log_pp = gp_update_fn(obs_input, regime, bel, cfg)

        # Transition probability from current regime to proposed regime
        regime_curr = bel.regime[bel.timestep.astype(int)]
        log_p_transition = log_transition_matrix[regime_curr, regime]

        # Book-keeping: record proposed regime and advance timestep
        timestep_new = bel.timestep + 1
        bel_update = bel_update.replace(
            regime=bel_update.regime.at[timestep_new.astype(int)].set(regime),
            timestep=timestep_new,
        )

        return bel_update, log_pp, log_p_transition

    return _update_conditional_posterior


# ============================================================
# Generic Step & Run
# ============================================================

def step(
    bel: ParticleState,
    obs_input: Any,
    cfg: Cfg,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    update_conditional_posterior_fn: Callable,
    forecast_fn: Callable,
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]]:
    """
    One step of streaming HMM inference.

    Args:
        bel: Current particle state.
        obs_input: Current observation (model-specific pytree).
        cfg: Configuration.
        transition_matrix: Transition probability matrix (K, K).
        log_transition_matrix: Log transition probability matrix (K, K).
        update_conditional_posterior_fn: Vmapped update function
            (produced by :func:`make_update_conditional_posterior`).
        forecast_fn: Forecast function.

    Returns:
        ``(bel_updated, (log_weights, means, variances, forecast_dict))``
    """
    regimes = jnp.arange(cfg.num_regimes)

    # Forecast *before* seeing current observation
    fcst = forecast_fn(bel, cfg, transition_matrix, obs_input)

    # Update under all regimes [S -> S*K]
    bel_update, log_pp, log_p_transition = update_conditional_posterior_fn(
        obs_input, regimes, bel, cfg, log_transition_matrix
    )

    # Update log-weights [S*K]
    bel_update = update_log_weights(bel_update, log_pp, log_p_transition)

    # Beam search: keep top-K [S*K -> S]
    bel_update = beam_search(bel_update, cfg.num_particles)

    return bel_update, (bel_update.log_weight, bel_update.means, bel_update.variances, fcst)


def run_streaming_hmm(
    obs: Any,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    key: jax.random.PRNGKey = None,
    n_particles: int = 5,
    var_obs: float = 1.0,
    mean_init: jax.Array = None,
    var_init: jax.Array = None,
    update_fn: Callable = None,
    forecast_fn: Callable = None,
    bel_init: ParticleState = None,
) -> Tuple[ParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]], jax.Array]:
    """
    Run the streaming HMM algorithm on a sequence of observations.

    This function is backward-compatible: when called without
    *update_fn*, *forecast_fn*, and *bel_init*, it defaults to the
    Gaussian observation model and creates the initial state from
    *key*, *mean_init*, and *var_init*.

    For custom observation models, pass *update_fn*, *forecast_fn*,
    and *bel_init* explicitly.

    Args:
        obs: Observation inputs — a scalar array ``(T,)`` for
            Gaussian, or any pytree with leading axis *T* for custom
            models.
        transition_matrix: Transition probability matrix ``(K, K)``.
        log_transition_matrix: Log transition matrix ``(K, K)``.
        key: JAX random key (required when *bel_init* is ``None``).
        n_particles: Number of particles for beam search.
        var_obs: Observation variance.
        mean_init: Initial mean estimate (Gaussian mode only).
        var_init: Initial variance estimate (Gaussian mode only).
        update_fn: Model-specific update function; defaults to
            :func:`gauss_update_fn`.
        forecast_fn: Model-specific forecast function; defaults to
            :func:`gauss_forecast_fn`.
        bel_init: Pre-initialized particle state. When provided,
            *key* / *mean_init* / *var_init* are ignored.

    Returns:
        ``(bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast), hist_weights)``
    """
    # Default to Gaussian model
    if update_fn is None:
        update_fn = gauss_update_fn
    if forecast_fn is None:
        forecast_fn = gauss_forecast_fn

    num_regimes = transition_matrix.shape[0]

    # Create initial state when not explicitly provided
    if bel_init is None:
        n_steps = jax.tree.leaves(obs)[0].shape[0]
        if mean_init is None:
            mean_init = jnp.array([0.0])
        if var_init is None:
            var_init = jnp.array([1.0])
        bel_init = init_gauss_particles(
            key, mean_init, var_init, n_particles, num_regimes, n_steps
        )
    else:
        n_particles = bel_init.log_weight.shape[0]

    cfg = Cfg(var_obs=var_obs, num_particles=n_particles, num_regimes=num_regimes)

    # Build the vmapped update function
    update_cond_post = make_update_conditional_posterior(update_fn)

    _step = partial(
        step,
        cfg=cfg,
        transition_matrix=transition_matrix,
        log_transition_matrix=log_transition_matrix,
        update_conditional_posterior_fn=update_cond_post,
        forecast_fn=forecast_fn,
    )

    bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast) = jax.lax.scan(
        _step, bel_init, obs
    )
    hist_weights = build_weights(hist_lw)

    return bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast), hist_weights


def gp_step(
    bel: GPParticleState,
    obs_input: Any,
    cfg: Cfg,
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    update_conditional_posterior_fn: Callable,
    forecast_fn: Callable,
) -> Tuple[GPParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]]:
    """
    One step of streaming HMM inference for GP observation model.
    # TODO: Remove dependence on transition_matrix. We only need log_transition_matrix.

    Args:
        bel: Current GP particle state.
        obs_input: Current observation (y, x).
        cfg: Configuration.
        transition_matrix: Transition probability matrix (K, K).
        log_transition_matrix: Log transition probability matrix (K, K).
        update_conditional_posterior_fn: Vmapped update function for GP.
        forecast_fn: Forecast function.

    Returns:
        ``(bel_updated, (log_weights, y_buffers, X_buffers, forecast_dict))``
    """
    regimes = jnp.arange(cfg.num_regimes)

    # Forecast *before* seeing current observation
    fcst = forecast_fn(bel, cfg, transition_matrix, obs_input)

    # Update under all regimes [S -> S*K]
    bel_update, log_pp, log_p_transition = update_conditional_posterior_fn(
        obs_input, regimes, bel, cfg, log_transition_matrix
    )

    # Update log-weights [S*K]
    bel_update = update_log_weights(bel_update, log_pp, log_p_transition)
    log_weight_update = bel_update.log_weight

    # Beam search: keep top-K [S*K -> S]
    bel_update = beam_search(bel_update, cfg.num_particles)

    return bel_update, (bel_update.log_weight, bel_update.y_buffers, bel_update.X_buffers, fcst, log_weight_update)


def run_gp_streaming_hmm(
    obs: Tuple[jax.Array, jax.Array],
    transition_matrix: jax.Array,
    log_transition_matrix: jax.Array,
    kernel: Callable,
    key: jax.random.PRNGKey,
    n_particles: int = 5,
    var_obs: float = 1.0,
    buffer_size: int = 50,
    n_ahead: int = 1,
    dt: float = 1.0,
    bel_init: Optional[GPParticleState] = None,
) -> Tuple[GPParticleState, Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]], jax.Array]:
    """
    Run the streaming HMM algorithm with GP observation model.

    Args:
        obs: Observation inputs (y, X) where y is (T,) and X is (T, dim_in).
        transition_matrix: Transition probability matrix (K, K).
        log_transition_matrix: Log transition matrix (K, K).
        kernel: GP kernel function.
        key: JAX random key.
        n_particles: Number of particles for beam search.
        var_obs: Observation variance.
        buffer_size: Size of FIFO buffer per regime.
        n_ahead: Number of steps ahead to forecast.
        dt: Time step between forecast points (default: 1.0).
        bel_init: Pre-initialized GP particle state.

    Returns:
        ``(bel_final, (hist_lw, hist_y_buffers, hist_X_buffers, hist_forecast), hist_weights)``
    """
    num_regimes = transition_matrix.shape[0]
    y, X = obs
    n_steps = y.shape[0]
    dim_in = X.shape[-1] if X.ndim > 1 else 1
    X = X.reshape(n_steps, dim_in)

    # Create initial state when not explicitly provided
    if bel_init is None:
        bel_init = init_gp_particles(
            key, dim_in, buffer_size, n_particles, num_regimes, n_steps
        )
    else:
        n_particles = bel_init.log_weight.shape[0]

    cfg = Cfg(var_obs=var_obs, num_particles=n_particles, num_regimes=num_regimes)

    # Build the vmapped update function
    update_cond_post = make_gp_update_conditional_posterior(kernel)
    forecast_fn = make_gp_forecast_fn(kernel, n_ahead=n_ahead, dt=dt)

    _step = partial(
        gp_step,
        cfg=cfg,
        transition_matrix=transition_matrix,
        log_transition_matrix=log_transition_matrix,
        update_conditional_posterior_fn=update_cond_post,
        forecast_fn=forecast_fn,
    )

    bel_final, (hist_lw, hist_y_buffers, hist_X_buffers, hist_forecast, hist_logpp) = jax.lax.scan(
        _step, bel_init, (y, X)
    )
    hist_weights = build_weights(hist_lw)

    return bel_final, (hist_lw, hist_y_buffers, hist_X_buffers, hist_forecast, hist_logpp), hist_weights
