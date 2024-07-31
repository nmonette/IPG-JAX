import jax
import jax.numpy as jnp
from flax.struct import dataclass

from ..utils import projection_simplex_truncated as proj

@dataclass
class TrainState:
    team_params: float
    adv_params: float

    @staticmethod
    def update_team(policy, train_state, grad, idx):
        val = policy.step(train_state.team_params[idx], grad)
        return train_state.replace(
            team_params = train_state.team_params.at[idx].set(val)
        )
    
    @staticmethod
    def update_team_agent(policy, train_state, grad, idx):
        val = policy.step(train_state.team_params[idx], grad)
        return train_state.replace(
            team_params = train_state.team_params.at[idx].set(val)
        )
    
    @staticmethod
    def update_adv(policy, train_state, grad):
        return train_state.replace(
            adv_params = policy.step(train_state.adv_params, grad)
        )
    
    @staticmethod
    def copy_team(old, new):
        return new.replace(
            team_params=old.team_params,
        )

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
        if len(self.param_dims) > 1:
            initializer = jax.nn.initializers.orthogonal()
        else:
            initializer = lambda rng, _: jax.random.uniform(rng, (self.param_dims[0],))
        return proj(initializer(rng, self.param_dims), self.eps)
    
    def step(self, params, grad):
        return proj(params - self.lr * grad, self.eps)
    
    def get_actions(self, rng, state, params):
        if len(state.shape) > 0:
            logits = jnp.log(params[tuple(state)])
        else:
            logits = jnp.log(params[state])
        action = jax.random.categorical(rng, logits)

        return action, logits[action]
    
    @staticmethod
    def team_diff(current, old):
        return jnp.linalg.norm(current[:-1] - old[:-1])
    
    @staticmethod
    def tree_change_at_idx(params, new_params, idx):
        return params.at[idx].set(new_params)

    def get_agent_params(self, params, idx):
        return params[idx]