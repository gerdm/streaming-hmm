# Streaming HMM: Predictive-First Online Regime Inference

Tutorial post: https://grdm.io/posts/hidden-markov-model

This repository contains the code for the paper **"A Predictive View on Streaming Hidden Markov Models"**.

The project implements an online Hidden Markov Model (HMM) framework designed for **streaming prediction and regime tracking** under strict compute budgets.

---

## Paper summary

### Motivation
Classical online HMM methods typically optimize latent-state posterior recovery or likelihood. In streaming settings, this can be expensive because the number of latent regime paths grows exponentially over time.

This work takes a **predictive-first** view:
- the main target is the one-step-ahead predictive distribution,
- the model is updated recursively as new data arrive,
- and computation is controlled by a fixed path budget.

### Main idea
At each time step, every retained path branches into all possible next regimes. Instead of keeping all branches, the method keeps only the best `S` hypotheses (beam width).

The key theoretical contribution is that this is not only a heuristic: under a constrained support size (`S` paths), the retained mixture is the **forward-KL optimal approximation** to the full posterior predictive mixture.

In short:
- full predictive mixture is intractable,
- truncate to `S` paths,
- keep top-`S` posterior path weights and renormalize,
- obtain a recursive deterministic streaming algorithm.

### Experimental findings (from the paper)
Under matched computational budgets, the Streaming HMM:
- gives competitive or better prequential prediction than Online EM and RBPF,
- is stable with small beam sizes,
- and remains computationally efficient.

The repository also demonstrates a GP-based regime-switching variant with multi-step forecasting.

---

## Repository structure

- `streaming_hmm.py`  
  Core implementation of Streaming HMM:
  - generic beam-search filtering scaffold,
  - Gaussian observation model,
  - AR observation model,
  - GP observation model with FIFO buffers and multi-step forecasts.

- `rbpf_hmm.py`  
  Rao-Blackwellized Particle Filter baseline with ESS-triggered resampling.

- `gaussian_process.py`  
  Standalone GP regression utilities used for experimentation.

- `online-hmm.ipynb`  
  Original notebook-style derivation/implementation for online Gaussian HMM.

- `streaming-hmm-demo.ipynb`  
  Compact demo of the modular streaming implementation.

- `gauss1d-hmm.ipynb`  
  Comparison study: Online EM vs Streaming HMM vs RBPF.

- `ar2-hmm-demo.ipynb`  
  Streaming AR(2)-HMM regime inference demo.

- `gp-hmm-demo.ipynb`  
  GP-HMM demo with one-step and multi-step predictive analysis.

---

## Method overview

For each retained path, the algorithm maintains:
1. path weight,
2. current regime sequence (implicitly by history),
3. regime-specific belief states (e.g., Gaussian conjugate state, AR posterior, or GP buffer state).

At each new observation:
1. **Forecast** using current retained mixture.
2. **Branch** each path over all possible next regimes.
3. **Update** regime-specific predictive state and evaluate log predictive density.
4. **Reweight** by transition probability and predictive likelihood.
5. **Prune** back to top-`S` paths (beam search) and renormalize.

This gives an online recursion with bounded memory/compute controlled by `S`.

---

## Installation

Create an environment and install core dependencies:

```bash
pip install jax jaxlib chex einops distrax matplotlib seaborn numpy
```

If you use notebooks:

```bash
pip install notebook ipykernel
```

---

## Quick start

### 1) Gaussian Streaming HMM

```python
import jax
import jax.numpy as jnp
from streaming_hmm import run_streaming_hmm

# Synthetic setup
key = jax.random.PRNGKey(0)
obs = jax.random.normal(key, (500,))

transition_matrix = jnp.array([
    [0.99, 0.01],
    [0.01, 0.99],
])
log_transition_matrix = jnp.log(transition_matrix)

bel_final, (hist_lw, hist_mean, hist_var, hist_forecast), hist_weights = run_streaming_hmm(
    obs,
    transition_matrix=transition_matrix,
    log_transition_matrix=log_transition_matrix,
    key=key,
    n_particles=5,
    var_obs=1.0,
)
```

### 2) AR observation model
Use `init_ar_particles`, `ar_update_fn`, and `ar_forecast_fn` with `run_streaming_hmm`.

### 3) GP observation model
Use `run_gp_streaming_hmm` with a kernel from `gaussian_kernel(...)` or `periodic_kernel(...)`.

---

## Key API entry points

- `run_streaming_hmm(...)`  
  Generic streaming HMM runner (Gaussian by default; can plug custom update/forecast functions).

- `run_gp_streaming_hmm(...)`  
  GP-specific streaming runner with multi-step forecast support.

- `run_rbpf_streaming_hmm(...)`  
  RBPF baseline for comparison.

---

## Notes on interpretation

- `n_particles` is used as beam width `S` in Streaming HMM (deterministic top-`S` truncation).
- In RBPF, particles are stochastic and may require larger counts for stable results.
- Multi-step GP forecasts include regime-transition uncertainty via powers of the transition matrix; they use an approximation that does not condition on unknown intermediate future observations.

---

## Citation

If you use this code, please cite:

```bibtex
@article{duranmartin2026predictive,
  title={A Predictive View on Streaming Hidden Markov Models},
  author={Duran-Martin, Gerardo},
  year={2026}
}
```

(Replace with the final bibliographic record used by the paper release.)
