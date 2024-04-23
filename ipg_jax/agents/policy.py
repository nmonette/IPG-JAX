import jax
import jax.numpy as jnp
from flax.struct import dataclass
from ..utils import projection_simplex_truncated as proj

class DirectPolicy:
    """
    We only need one of these, as it doesn't store the 
    parameters of the policy. 
    """
    def __init__(self, param_dims, lr, eps):
        self.param_dims = param_dims
        self.lr = lr
        self.eps = eps

    def init_params(self, rng, _=None):
        # return jnp.full(self.param_dims, 1 / self.param_dims[-1])
        initializer = jax.nn.initializers.orthogonal()
        return proj(initializer(rng, self.param_dims), self.eps)
    
    def step(self, params, grad):
        return proj(params + self.lr * grad, self.eps)
    
    def get_actions(self, rng, state, params):
        logits = jnp.log(params[tuple(state)])
        action = jax.random.categorical(rng, logits)

        return action, logits[action]
        
    def save_model(self, path):
        # use orbax for this
        pass