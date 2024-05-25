import jax
import jax.numpy as jnp

from ..br import make_adv_br
from ..reinforce import make_reinforce

from functools import partial

def compute_nash_gap(rng, args, adv_policy, team_policy, rollout, train_state, adv_state_space):

    def avg_return(rng, train_state):
        
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
        data, _, _, _ = rollout.batch_rollout(rollout_rng, train_state.adv_params, train_state.team_params, init_obs, init_state)

        gamma = jnp.cumprod(jnp.full((data.reward.shape[1], ), args.gamma)) / args.gamma

        agent_return = lambda r, g: jnp.dot(r, g)
        episode_return = jax.vmap(agent_return, in_axes=(-1, None))
        
        return jax.vmap(episode_return, in_axes=(0, None))(jnp.float32(data.reward), gamma).mean(axis=0)
    
    rng, _rng = jax.random.split(rng)
    base = avg_return(_rng, train_state)
    gap = jnp.empty((rollout.num_agents,))

    adv_br = make_adv_br(args, rollout, adv_state_space, adv_policy) 
    reinforce = make_reinforce(args, rollout)

    # Run agent's train loop 
    rng, _rng = jax.random.split(rng)
    adv_train_state = adv_br(_rng, train_state)
    gap = gap.at[-1].set(avg_return(rng, adv_train_state)[-1] - base[-1])

    def gap_fn(rng, train_state, agent_idx):
    
        # Run agent's train loop 
        rng, _rng = jax.random.split(rng)
        train_state = adv_br(_rng, train_state, agent_idx)

        def update_fn(carry, _):
            rng, train_state = carry

            train_state = train_state.update_team_agent(team_policy, train_state, team_policy.get_agent_params(reinforce(train_state.team_params, train_state.adv_params, rng, agent_idx), agent_idx), agent_idx)

            return (rng, train_state), train_state

        rng, _rng = jax.random.split(rng)
        _, all_states = jax.lax.scan(update_fn, (_rng, train_state), None, args.br_length)
        # TODO: try avg-iterate for the team
        ## DOING it::
        avg_params = train_state.replace(
            team_params = jax.tree_map(lambda x, y:  x.at[agent_idx].set(y[:, agent_idx].mean(axis=0)), train_state.team_params, all_states.team_params)
        )
        return avg_return(rng, avg_params)[agent_idx] - base[agent_idx]

    return jax.vmap(gap_fn, in_axes=(0, None, 0))(jax.random.split(rng, rollout.num_agents), train_state, jnp.arange(rollout.num_agents))
