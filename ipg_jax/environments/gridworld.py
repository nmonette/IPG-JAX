from flax.struct import dataclass

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

# LEFT RIGHT UP DOWN

@dataclass
class Obj:
    idx: int
    pos: jnp.ndarray
    active: int = 1
    adv: int = 0

@dataclass
class EnvState:
    agent: Obj
    goal1: Obj
    goal2: Obj
    time: int
    done: int = 0

class AdvMultiGrid(environment.Environment):
    def __init__(self, dim, max_time, partial = True):
        super().__init__()
        self.dim = dim
        self.walls = jnp.pad(jnp.full((dim - 1, dim - 1), False, jnp.int32), ((1, 1), (1, 1)), constant_values=True)
        self.max_time = max_time
        self.possible_coords = jnp.array([(i,j) for i in range(dim) for j in range(dim)], dtype=jnp.int32)

        self.num_agents = 3
        self.partial = partial

    def init_state(self, rng):
        rng, goal_rng, agent_rng = jax.random.split(rng, 3)
        goals = jax.random.choice(goal_rng, jnp.arange(self.dim * self.dim),shape=(2, ), replace=False)
        goal1 = self.possible_coords[goals[0]]
        goal2 = self.possible_coords[goals[1]]

        probs = 1 / (self.dim * self.dim - 2) * jnp.array(jnp.logical_and(jnp.arange(self.dim * self.dim) != goals[0], jnp.arange(self.dim * self.dim) != goals[1]), dtype=jnp.float32)

        agents = jax.random.choice(agent_rng, self.possible_coords, shape=(3,), p=probs)
        return EnvState(
            jax.vmap(lambda i,p: Obj(i, p, jnp.array(1,jnp.int32), jnp.array(i == 2, jnp.int32)), in_axes=(0, 0))(jnp.arange(3), agents),
            Obj(0, goal1, jnp.array(1, jnp.int32)),
            Obj(1, goal2, jnp.array(1, jnp.int32)),
            jnp.array(0, jnp.int32),
            jnp.array(0, jnp.int32)
        )

    def _handle_actions(self, pos, action):
        step = jnp.int32(action == 0) * jnp.array([0, -1], jnp.int32) \
             + jnp.int32(action == 1) * jnp.array([0, 1], jnp.int32) \
             + jnp.int32(action == 2) * jnp.array([1, 0], jnp.int32) \
             + jnp.int32(action == 3) * jnp.array([-1, 0], jnp.int32)
        
        pos = pos + step

        return jnp.clip(pos, 0, self.dim - 1)
    
    def _handle_rewards(self, state, agent):
        val1 =jnp.int32(jnp.array_equal(agent.pos, state.goal1.pos)) * state.goal1.active
        val2 = jnp.int32(jnp.array_equal(agent.pos, state.goal2.pos)) * state.goal2.active
        val = val1 + val2
        reward = ((1 - agent.adv) * jnp.full((self.num_agents, ), val, jnp.int32).at[-1].set(-val) + agent.adv * jnp.full((self.num_agents, ), -val, jnp.int32).at[-1].set(val))
        return reward, val1, val2

    def get_obs(self, state):

        def extract_agent_obs():
            return jnp.concatenate((jnp.append(state.agent.pos, state.agent.active.reshape(self.num_agents, -1), 1).flatten(), state.goal1.pos, state.goal1.active.reshape(1 ), state.goal2.pos, state.goal2.active.reshape(1)))
        
        def partial_extract_agent_obs():
            goals = jnp.concatenate((state.goal1.pos, state.goal1.active.reshape(1 ), state.goal2.pos, state.goal2.active.reshape(1)))
            goals = jnp.repeat(goals.reshape((-1, 6)), jnp.array((3, ), jnp.int32), axis=0, total_repeat_length=3)
            return jnp.append(state.agent.pos, goals, 1)
        
        if self.partial:
            return jnp.int32(partial_extract_agent_obs())
        else:
            return jnp.int32(jnp.repeat(extract_agent_obs().reshape(-1, 15), jnp.array((3, ), jnp.int32), axis=0, total_repeat_length=3))
    
    def step_env(self, _, state: EnvState, action: jnp.ndarray, *args, **kwargs):
        pos = jax.vmap(self._handle_actions)(state.agent.pos, action)
        reward, term1, term2 = jax.vmap(self._handle_rewards, in_axes=(None, 0))(state, state.agent)



        agent = state.agent.replace(pos=pos, active=jnp.int32(jnp.logical_not(term1 + term2)))
        goal1 = state.goal1.replace(active=jnp.int32(jnp.all(jnp.logical_and(state.goal1.active, jnp.logical_not(term1)))))
        goal2 = state.goal2.replace(active=jnp.int32(jnp.all(jnp.logical_and(state.goal2.active, jnp.logical_not(term2)))))
        done = jnp.int32(jnp.logical_or(state.time == self.max_time, jnp.logical_or(jnp.sum(state.agent.active[:-1]) == 0, jnp.int32(state.agent.active[-1]) == 0), ))
        state = state.replace(time = state.time + 1, done=done, agent=agent, goal1=goal1, goal2=goal2)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.sum(axis=0),
            done,
            {}
        )
        
    def reset_env(self, rng, *args, **kwargs):
        state = self.init_state(rng)
        return self.get_obs(state), state
    
    @property
    def name(self) -> str:
        """Environment name."""
        return "MultiGrid-TeamAdv-v0"
    
    def action_space(
        self, params = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params) -> spaces.Dict:
        """Observation space of the environment."""
        if self.partial:
            return spaces.Dict({
                0: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)],
                1: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)],
                2: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)],
            })
        else:
            return spaces.Dict({
                0: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)],
                1: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)],
                2: [spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2), spaces.Discrete(self.dim), spaces.Discrete(self.dim), spaces.Discrete(2)]
            })
        

registered_envs = ["MultiGrid-TeamAdv-v0"]
        