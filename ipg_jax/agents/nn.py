from functools import partial 

import jax
import jax.numpy as jnp
from ..utils import projection_simplex_truncated as proj

from flax.linen.activation import selu
import flax.linen as nn
from flax.struct import dataclass, field
import optax

from typing import Union

@dataclass
class TrainState:
    team_params: float
    adv_params: float

    team_optimizer: "Optimizer" = field(pytree_node=False)
    adv_optimizer: "Optimizer" = field(pytree_node=False)

    team_opt_states: float
    adv_opt_state: float

    @staticmethod
    def update_team(policy, train_state, grad, idx):
        new_params, new_opt_state = jax.vmap(policy.step2, in_axes=(0, 0, None, 0))(train_state.team_params, grad, train_state.team_optimizer, train_state.team_opt_states)
        return train_state.replace(
            team_params=new_params,
            team_opt_states=new_opt_state
        )
    
    @staticmethod
    def update_adv(policy, train_state, grad):
        new_params, new_opt_state = policy.step2(train_state.adv_params, grad, train_state.adv_optimizer, train_state.adv_opt_state)
        return train_state.replace(
            adv_params=new_params,
            adv_opt_state=new_opt_state
        )
    
    @staticmethod
    def copy_team(old, new):
        return new.replace(
            team_params=old.team_params,
            team_opt_states=old.team_opt_states
        )

class SELUPolicy(nn.Module):
    eps: float
    arch: list[int]
    state_space: Union[list[int], str]

    def setup(self):
        self.layers = [nn.Dense(s) for s in self.arch]

    def encode_state(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        if self.state_space != "c":
            obs = jnp.concatenate(
                [
                    jax.nn.one_hot(obs_, num_classes=int(self.state_space[idx]))
                    for idx, obs_ in enumerate(jnp.split(x, 1, axis=1))
                ],
                axis=-1,
            ).reshape(x.shape[0], -1)
        else:
            obs = x
        return obs

    def __call__(self, x):
        x = self.encode_state(x)
        for i in range(len(self.layers) - 1):
            x = selu(self.layers[i](x))
        val = proj(self.layers[-1](x), self.eps)
        if val.shape[0] == 1:
            val = val.flatten()
        return val    
    
    def get_actions(self, rng, state, params):
        out = self.apply(params, state)
        logits = jnp.log(out)
        action = jax.random.categorical(rng, logits)

        return action, logits[action]
    
    @staticmethod
    def team_diff(current, old):
        current_weights = jax.tree_map(lambda x: x[:-1], current)
        current_leaves = jnp.concatenate([i.flatten() for i in jax.tree_util.tree_leaves(current_weights)])

        old_weights = jax.tree_map(lambda x: x[:-1], old)
        old_leaves = jnp.concatenate([i.flatten() for i in jax.tree_util.tree_leaves(old_weights)])

        return jnp.linalg.norm(current_leaves - old_leaves)
    
    @staticmethod
    def tree_change_at_idx(params, new_params, idx):
        map_fn = lambda leaf, new_leaf, idx: leaf.at[idx].set(new_leaf)
        
        fn = partial(map_fn, idx=idx)
        return jax.tree_map(fn, params, new_params)
    
    def step(self, params, grad, optimizer, optimizer_state, idx):
        updates, new_optimizer_state = optimizer.update(grad, jax.tree_map(lambda x: x[idx], optimizer_state))
        idx_params = self.get_agent_params(params, idx)
        params = self.tree_change_at_idx(params, optax.apply_updates(idx_params, updates), idx)
        optimizer_state = self.tree_change_at_idx(optimizer_state, new_optimizer_state, idx)
        return params, optimizer_state
    
    def step2(self, params, grad, optimizer, optimizer_state):
        updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, new_optimizer_state
        
    def get_agent_params(self, params, idx):
        """
        Gets a p articular agent's parameters
        (agent idx).
        """
        return jax.tree_map(lambda x: x[idx], params)