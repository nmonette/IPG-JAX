from chex._src.pytypes import PRNGKey
from flax.struct import dataclass

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

@dataclass
class EnvState:
    matrices: jnp.ndarray
    current: int
    done: int = 0
    time: int = 0

class AdvRPS(environment.Environment):
    def __init__(self, num_states, num_agents, num_actions, num_timesteps = 5):
        super().__init__()
        self._num_states = num_states
        self.num_agents = num_agents
        self._num_actions = num_actions
        self.num_timesteps = num_timesteps

    def init_state(self, rng):
        rng, gen_rng, init_rng = jax.random.split(rng, 3)
        matrices = jax.random.uniform(gen_rng, tuple([self._num_states] + [self._num_actions for _ in range(self.num_agents)]), minval=-1)
        init_matrix = jax.random.choice(init_rng, self._num_states)
        return EnvState(
            matrices=matrices,
            current=init_matrix,
            done = jnp.int32(0),
            time = jnp.int32(1),
        )
    
    def step_env(self, rng, state, action, *args, **kwargs):
        team_reward = state.matrices[state.current][tuple(action)]
        reward = jnp.full((self.num_agents, ), team_reward).at[-1].set(-team_reward.squeeze())

        rng, _rng = jax.random.split(rng)
        matrix = jax.random.choice(_rng, self._num_states)

        state = state.replace(
            current = matrix,
            done = jnp.bitwise_or(jnp.int32(state.time + 1 == self.num_timesteps), state.done),
            time = state.time + 1
        )
        return (
            jax.lax.stop_gradient(jnp.full(self.num_agents, state.current).reshape(-1, 1)),
            jax.lax.stop_gradient(state),
            reward,
            state.done,
            {}
        )
        
    def reset_env(self, rng, *args, **kwargs):
        state = self.init_state(rng)
        return jnp.full(self.num_agents, state.current).reshape(-1, 1), state
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "RPS-TeamAdv-v0"
    
    def action_space(
        self, params = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self._num_actions)
    
    def observation_space(self, params):
        return spaces.Discrete(self._num_states)
    
registered_envs = ["RPS-TeamAdv-v0"]