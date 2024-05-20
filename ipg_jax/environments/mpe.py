from jaxmarl import make
import jax.numpy as jnp

class AdvMPE:
    def __init__(self, **kwargs):
        self.env = make("MPE_simple_adversary_v3")

    def step(self, rng, state, actions):
        action = {
            "agent_0": actions[0],
            "agent_1": actions[1],
            "adversary_0": actions[2]
        }

        obs, state, reward, done, infos = self.env.step(rng, state, action)

        reward  = jnp.stack([reward["agent_0"], reward["agent_1"], reward["adversary_0"]])
        done = done["__all__"]
        
        return obs, state, reward, done, infos
    
    def reset(self, rng):
        return self.env.reset(rng)

        