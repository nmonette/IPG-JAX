"""
Visualizer largely based on https://github.com/RobertTLange/gymnax/blob/main/gymnax/visualize/visualizer.py.
"""

from typing import Optional
import gym
import jax
import jax.numpy as jnp
from matplotlib import animation
import matplotlib.pyplot as plt
import gymnax

from gridworld import AdvMultiGrid

class GridVisualizer(object):
    """Visualizer for Gymnax environments."""

    def __init__(self, env_kwargs, state_seq_arg, reward_seq_arg=None):

        self.env = AdvMultiGrid(**env_kwargs)
        self.state_seq = state_seq_arg
        self.reward_seq = reward_seq_arg
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.interval = 200 
        self.artists = []

    def animate(
        self,
        save_fname: Optional[str] = "test.gif",
        view: bool = False,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)."""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)
        # Simply view it 3 times
        if view:
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    def init(self):
        """Plot placeholder points."""
        self.ax.set_facecolor('black')

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(100, 0)
        plt.axis("off")
        self.ax.add_artist(plt.Rectangle((0, 0), 100, 100, facecolor='black',
                           transform=self.ax.transAxes, zorder=-1))
        artists = []
        def plot_object(obj, goal=False):
            def anchor_fn(x,y):
                x = x * (100 / self.env.dim) + (50 / self.env.dim)
                y = y * (100 / self.env.dim) + (50 / self.env.dim)
                return x,y
            
            def rect_anchor_fn(x,y):
                x = x * (100 / self.env.dim)
                y = y * (100 / self.env.dim)
                return x,y

            if goal:
                artist = self.ax.add_artist(plt.Rectangle(rect_anchor_fn(obj.pos[0], obj.pos[1]), 100 / self.env.dim, 100 / self.env.dim, color="lime"))
            else:
                def plot_adv():
                    return plt.Circle(anchor_fn(obj.pos[0], obj.pos[1]), 50 / self.env.dim - 2, color="red")
                def plot_team(): 
                    return plt.Circle(anchor_fn(obj.pos[0], obj.pos[1]), 50 / self.env.dim - 2, color="blue")
                
                artist = self.ax.add_artist(jax.lax.cond(obj.adv, plot_adv, plot_team))
            
            return artist

        for idx in range(self.env.num_agents):
            agent = jax.tree_util.tree_map(lambda a: a[idx], self.state_seq[0].agent)
            artists.append(plot_object(agent))

        artists.append(plot_object(self.state_seq[0].goal1, True))
        artists.append(plot_object(self.state_seq[0].goal2, True))
        xlines = jnp.linspace(0, 100, self.env.dim + 1)
        ylines = jnp.linspace(0, 100, self.env.dim + 1)

        for x in xlines:
            self.ax.axvline(x, 0, 100, c="grey", lw=2)
        for y in ylines:
            self.ax.axhline(y, 0, 100, c="grey", lw=2)
        
        self.artists = artists

        self.goal1_removed = False
        self.goal2_removed = False
        self.agents_removed = [False for _ in range(self.env.num_agents)]
        self.finished = False


    def update(self, frame):
        frame = self.state_seq[frame]
        if frame.done or self.finished:
            self.finished = True
            return

        artists = []
        def plot_object(obj, goal=False):
            def anchor_fn(x,y):
                x = x * (100 / self.env.dim) + (50 / self.env.dim)
                y = y * (100 / self.env.dim) + (50 / self.env.dim)
                return x,y
            
            def rect_anchor_fn(x,y):
                x = x * (100 / self.env.dim)
                y = y * (100 / self.env.dim)
                return x,y

            if goal:
                artist = rect_anchor_fn(obj.pos[0], obj.pos[1])
            else:
                def plot_adv():
                    return anchor_fn(obj.pos[0], obj.pos[1])
                def plot_team(): 
                    return anchor_fn(obj.pos[0], obj.pos[1])
                
                artist = jax.lax.cond(obj.adv, plot_adv, plot_team)
            
            return artist

        for idx in range(self.env.num_agents):
            agent = jax.tree_util.tree_map(lambda a: a[idx], frame.agent)
            if agent.active:
                artists.append(plot_object(agent))
            elif not agent.active and not self.agents_removed[idx]:
                self.artists[idx].remove()
                artists.append((0,0))
                self.agents_removed[idx] = True
            else:
                artists.append((0,0))

        if frame.goal1.active:
            artists.append(plot_object(frame.goal1, True))
        elif jnp.bitwise_not(frame.goal1.active) and not self.goal1_removed:
            self.artists[idx+1].remove()
            artists.append((0,0))
            self.goal1_removed = True
        else:
            artists.append((0,0))

        if frame.goal2.active:
            artists.append(plot_object(frame.goal2, True))
        elif jnp.bitwise_not(frame.goal2.active) and not self.goal2_removed:
            self.artists[idx+2].remove()
            artists.append((0,0))
            self.goal2_removed = True
        else:
            artists.append((0,0))

        for i in range(len(artists)):
            if i < len(artists) - 2:
                self.artists[i].set_center(artists[i])
            else:
                self.artists[i].set_xy(artists[i])


if __name__ == "__main__":
    # rng = jax.random.PRNGKey(0)
    # env, env_params = gymnax.make("Pong-misc")

    # state_seq, reward_seq = [], []
    # rng, rng_reset = jax.random.split(rng)
    # obs, env_state = env.reset(rng_reset, env_params)
    # while True:
    #     state_seq.append(env_state)
    #     rng, rng_act, rng_step = jax.random.split(rng, 3)
    #     action = env.action_space(env_params).sample(rng_act)
    #     next_obs, next_env_state, reward, done, info = env.step(
    #         rng_step, env_state, action, env_params
    #     )
    #     reward_seq.append(reward)
    #     if done:
    #         break
    #     else:
    #         obs = next_obs
    #         env_state = next_env_state

    # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    # vis = Visualizer(env, env_params, state_seq, cum_rewards)
    # vis.animate("anim.gif")
    with jax.disable_jit(True):
        env = AdvMultiGrid(**{"dim": 5, "max_time":12},)
        rng = jax.random.split(jax.random.key(0), 100)
        state_seq = [(lambda r: env.reset(r)[1])(_rng) for _rng in rng]
        gv = GridVisualizer({"dim": 5, "max_time":12}, state_seq, None)
        gv.animate(view=True)