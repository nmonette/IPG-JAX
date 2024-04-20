import jax.numpy as jnp
import jax
from functools import partial

def projection_simplex_truncated(x: jnp.ndarray, eps: float) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * eps - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((idx < x.shape[-1]) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * eps).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
        
    # print(jnp.expand_dims(totals, -1).shape)
        
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, eps, 1)

if __name__ == "__main__":
    rng = jax.random.key(0)
    x = jax.random.uniform(rng, (4,4,2,4,4,2,4,4,2,4,4,2,4))
    projected = projection_simplex_truncated(x, 0.1)
    print(projected[4,4,2,4,4,2,4,4,2,4,4,2, :])
    print(projected[4,4,2,4,4,2,4,4,2,4,4,2, :].sum())
    
    # x = jnp.array([1/3, 2/3, 0])
    # x = jnp.array([1/3, 0, 2/3])
    # print(projection_simplex_truncated(x, 0.1).sum())