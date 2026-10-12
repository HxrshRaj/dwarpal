"""Tiny MLP trajectory forecaster in JAX (hand-written Adam, jit + grad)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i), jnp.zeros(o)) for k, i, o in zip(ks, sizes[:-1], sizes[1:])]


def forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def huber(pred, y, delta=2.0):
    e = jnp.abs(pred - y)
    return jnp.mean(jnp.where(e < delta, 0.5 * e**2, delta * (e - 0.5 * delta)))


def train(Xtr, ytr, epochs=300, lr=3e-3, hidden=32, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    params = init(jax.random.PRNGKey(seed), [Xtr.shape[1], hidden, hidden, 2])
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    X, Y = jnp.asarray((Xtr - mu) / sd), jnp.asarray(ytr)
    loss_fn = lambda p: huber(forward(p, X), Y)

    @jax.jit
    def step(p, m, v, t):
        loss, g = jax.value_and_grad(loss_fn)(p)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b**2, v, g)
        p = jax.tree_util.tree_map(
            lambda w, a, b: w - lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8), p, m, v
        )
        return p, m, v, loss

    for t in range(1, epochs + 1):
        params, m, v, loss = step(params, m, v, t)
    return lambda Xn: np.asarray(forward(params, jnp.asarray((Xn - mu) / sd)))
