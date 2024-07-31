"""
Based on https://github.com/EmptyJackson/groove/blob/main/environments/rollout.py, 
adapted for Multi-Agent RL and Direct Parameterization.
"""
from typing import Optional

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .gridworld import AdvMultiGrid
from .rps import AdvRPS
from ..agents import DirectPolicy, SELUPolicy

@dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    log_probs: jnp.ndarray

class RolloutWrapper:
    def __init__(
        self,
        adv_policy: DirectPolicy | SELUPolicy,
        team_policy: DirectPolicy | SELUPolicy,
        env_name: str = "MultiGrid-TeamAdv-v0",
        train_rollout_len: Optional[int] = None,
        eval_rollout_len: Optional[int] = None,
        env_kwargs: dict = {},
        return_info: bool = False,
        num_agents: int = 3, 
        gamma: float = 0.9,
        state_action_space = None
    ):
        """
        env_name (str): Name of environment to use.
        train_rollout_len (int): Number of steps to rollout during training.
        eval_rollout_len (int): Number of steps to rollout during evaluation.
        env_kwargs (dict): Static keyword arguments to pass to environment, same for all agents.
        return_info (bool): Return rollout information.
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        # Define the RL environment & network forward function
        if env_name == "MultiGrid-TeamAdv-v0":
            self.env = AdvMultiGrid(**env_kwargs) 
        else:
            self.env = AdvRPS(**env_kwargs)
            
        self.adv_policy = adv_policy
        self.team_policy = team_policy
        self.num_agents = num_agents
        self.gamma = gamma
        self.train_rollout_len = train_rollout_len
        self.eval_rollout_len = eval_rollout_len
        self.return_info = return_info
        self.state_action_space = state_action_space

    # --- ENVIRONMENT RESET ---
    def batch_reset(self, rng, num_workers):
        """Reset a single environment for multiple workers, returning initial states and observations."""
        rng = jax.random.split(rng, num_workers)
        batch_reset_fn = jax.vmap(self.env.reset)
        return batch_reset_fn(rng)
    
    # --- ENVIRONMENT ROLLOUT ---
    def batch_rollout(
        self, rng, adv_params, team_params, init_obs, init_state, eval=False, adv=False, adv_idx = -1
    ):
        """Evaluate an agent on a single environment over a batch of workers."""
        rng = jax.random.split(rng, init_obs.shape[0])
        if adv:
            return jax.vmap(self.adv_rollout, in_axes=(0, None, None, 0, 0, None, None))(
            rng, adv_params, team_params, init_obs, init_state, adv_idx, eval
        )
        else:
            return jax.vmap(self.single_rollout, in_axes=(0, None, None, 0, 0, None))(
                rng, adv_params, team_params, init_obs, init_state, eval
            )

    def single_rollout(
        self, rng, adv_params, team_params, init_obs, init_state, eval=False
    ):
        """Rollout an episode."""
        def policy_step(state_input, _):
            rng, obs, state, adv_params, team_params, valid_mask = state_input

            action = jnp.empty((self.num_agents, ))
            log_probs = jnp.empty((self.num_agents, ))

            rng, team_rng, adv_rng = jax.random.split(rng, 3)
            team_rng = jax.random.split(team_rng, self.num_agents - 1)
            team_action, team_lp = jax.vmap(self.team_policy.get_actions)(team_rng, obs[:-1], team_params)

            action = action.at[:-1].set(team_action)
            log_probs = log_probs.at[:-1].set(team_lp)

            adv_action, adv_lp = self.adv_policy.get_actions(adv_rng, obs[-1], adv_params)

            action = action.at[-1].set(adv_action)
            log_probs = log_probs.at[-1].set(adv_lp)

            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = self.env.step(
                _rng, state, action
            )
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                rng,
                next_obs,
                next_state,
                adv_params, team_params,
                new_valid_mask,
            ]
            transition = Transition(obs, action, reward, next_obs, done, log_probs)
            if self.return_info:
                return carry, (transition, info)
            return carry, transition

        # Scan over episode step loop
        carry_out, rollout = jax.lax.scan(
            policy_step,
            [
                rng,
                init_obs,
                init_state,
                adv_params, team_params,
                jnp.int32(1.0),
            ],
            (),
            self.eval_rollout_len if eval else self.train_rollout_len,
        )
        if self.return_info:
            rollout, info = rollout
        end_obs, end_state, cum_return = carry_out[1], carry_out[2], carry_out[5]
        if self.return_info:
            return rollout, end_obs, end_state, cum_return, info
        return rollout, end_obs, end_state, cum_return
    
    def render_rollout(
        self, rng, adv_params, team_params, init_obs, init_state, eval=False
    ):
        """Rollout an episode."""
        def policy_step(state_input, _):
            rng, obs, state, adv_params, team_params, valid_mask = state_input

            action = jnp.empty((self.num_agents, ))
            log_probs = jnp.empty((self.num_agents, ))

            rng, team_rng, adv_rng = jax.random.split(rng, 3)
            team_rng = jax.random.split(team_rng, self.num_agents - 1)
            team_action, team_lp = jax.vmap(self.team_policy.get_actions)(team_rng, obs[:-1], team_params)

            action = action.at[:-1].set(team_action)
            log_probs = log_probs.at[:-1].set(team_lp)

            adv_action, adv_lp = self.adv_policy.get_actions(adv_rng, obs[-1], adv_params)

            action = action.at[-1].set(adv_action)
            log_probs = log_probs.at[-1].set(adv_lp)

            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = self.env.step(
                _rng, state, action
            )
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                rng,
                next_obs,
                next_state,
                adv_params, team_params,
                new_valid_mask,
            ]
            transition = Transition(obs, action, reward, next_obs, done, log_probs)
            if self.return_info:
                return carry, (transition, info)
            return carry, next_state

        # Scan over episode step loop
        _, states = jax.lax.scan(
            policy_step,
            [
                rng,
                init_obs,
                init_state,
                adv_params, team_params,
                jnp.int32(1.0),
            ],
            (),
            self.eval_rollout_len if eval else self.train_rollout_len,
        )
        return states

    def adv_rollout(
        self, rng, adv_params, team_params, init_obs, init_state, adv_idx, eval=False
    ):
        """Rollout an episode."""
        def policy_step(state_input, _):
            rng, obs, state, adv_params, team_params, valid_mask, lambda_, gamma = state_input

            action = jnp.empty((self.num_agents, ))
            log_probs = jnp.empty((self.num_agents, ))

            rng, team_rng, adv_rng = jax.random.split(rng, 3)
            team_rng = jax.random.split(team_rng, self.num_agents - 1)
            team_action, team_lp = jax.vmap(self.team_policy.get_actions)(team_rng, obs[:-1], team_params)

            action = action.at[:-1].set(team_action)
            log_probs = log_probs.at[:-1].set(team_lp)

            adv_action, adv_lp = self.adv_policy.get_actions(adv_rng, obs[-1], adv_params)

            action = action.at[-1].set(adv_action)
            log_probs = log_probs.at[-1].set(adv_lp)
            
            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = self.env.step(
                _rng, state, action
            )
            new_valid_mask = valid_mask * (1 - done)

            state_action = tuple(jnp.pad(obs[adv_idx], (0, 1)).at[adv_idx].set(action[adv_idx]))
            lambda_ = lambda_.at[state_action].set(lambda_[state_action] + gamma)
            carry = [
                rng,
                next_obs,
                next_state,
                adv_params, team_params,
                new_valid_mask,
                lambda_,
                gamma * self.gamma
            ]
            transition = Transition(obs, action, reward, next_obs, done, log_probs)
            if self.return_info:
                return carry, (transition, info)
            return carry, transition
        # Scan over episode step loop
        carry_out, rollout = jax.lax.scan(
            policy_step,
            [
                rng,
                init_obs,
                init_state,
                adv_params, team_params,
                jnp.int32(1.0),
                jnp.zeros_like(adv_params) if self.state_action_space is None else jnp.zeros(self.state_action_space),
                jnp.float32(1.0),
            ],
            (),
            self.eval_rollout_len if eval else self.train_rollout_len,
        )
        if self.return_info:
            rollout, info = rollout
        end_obs, end_state, cum_return, lambda_ = carry_out[1], carry_out[2], carry_out[5], carry_out[6]
        if self.return_info:
            return rollout, end_obs, end_state, cum_return, lambda_, info
        return rollout, end_obs, end_state, cum_return, lambda_


    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape