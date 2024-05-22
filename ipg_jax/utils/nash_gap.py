import jax
import jax.numpy as jnp

from functools import partial

def compute_nash_gap(rng, args, policy, agent_params, rollout):

    def avg_return(rng, params):
        
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
        data, _, _, _ = rollout.batch_rollout(rollout_rng, params, init_obs, init_state, adv=False)

        gamma = jnp.cumprod(jnp.full((data.reward.shape[1], ), args.gamma)) / args.gamma

        agent_return = lambda r, g: jnp.dot(r, g)
        episode_return = jax.vmap(agent_return, in_axes=(-1, None))
        
        return jax.vmap(episode_return, in_axes=(0, None))(jnp.float32(data.reward), gamma).mean(axis=0)
    
    rng, _rng = jax.random.split(rng)
    base = avg_return(_rng, agent_params)

    # Calculate adversarial best response
    def adv_br(carry, _):
        rng, agent_params = carry

        # Calculate Gradients
        @partial(jax.grad, has_aux=True)
        def loss(agent_params, rng):
            # Collect rollouts
            rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
            init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
            data, _, _, _, lambda_ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=True)

            lambda_ = lambda_.mean(axis=0)

            def inner_loss(data):

                fn = lambda r, l, lp: (r - args.nu * l) * lp 

                idx = jnp.concatenate((data.obs[:, -1], data.action[:, -1].reshape(data.action.shape[0], -1)), axis=-1)
                idx_fn = lambda idx: lambda_[tuple(idx)]
                lambdas = jax.vmap(idx_fn)(idx)

                loss = jax.vmap(fn)(jnp.float32(data.reward[:, -1]), lambdas, jnp.cumsum(data.log_probs[:, -1]))

                disc = jnp.cumprod(jnp.ones_like(loss) * args.gamma) / args.gamma
                return jnp.dot(loss, disc), jnp.dot(data.reward[:, -1], disc)
                    
            grad, val =  jax.vmap(inner_loss)(data)
            
            return grad.mean(), val.mean()

        rng, _rng = jax.random.split(rng)
        grad, val = loss(agent_params, _rng)
        grad = grad[-1]
        agent_params = agent_params.at[-1].set(policy.step(agent_params[-1], grad))

        return (rng, agent_params), (agent_params[-1], val)

    # Run adversary's train loop 
    rng, _rng = jax.random.split(rng)
    _, (adv_params_new, val)  = jax.lax.scan(adv_br, (_rng, agent_params), None, args.br_length)

    # Best Iterate of Adversary BR 
    idx = jnp.argmax(val) - 1
    adv_params = agent_params.at[-1].set(adv_params_new[jnp.where(idx > 0, idx, 0)])

    rng, _rng = jax.random.split(rng)
    adv_gap = (avg_return(_rng, adv_params)[-1] - base[-1])
    
    # Update team
    def train_single(rng, agent_params, idx):
        def train_team(carry, _):
            rng, agent_params = carry

            @jax.grad
            def outer_loss(agent_params, rng):
                rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
                init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
                data, _, _, _ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=False)
                
                def episode_loss(log_probs, rewards):

                    def inner_returns(carry, i):
                        returns = carry

                        return returns.at[i].set(args.gamma * returns[i + 1] + rewards[i]), None
                        
                    returns, _ = jax.lax.scan(inner_returns, (jnp.zeros_like(rewards).at[-1].set(rewards[-1])), jnp.arange(rewards.shape[0]), reverse=True)

                    return jnp.dot(log_probs, returns)
                
                return jax.vmap(episode_loss, in_axes=(0, 0))(data.log_probs[:,:, idx], jnp.float32(data.reward[:,:, idx])).mean()

            rng, _rng = jax.random.split(rng)
            grad = outer_loss(agent_params, _rng)[idx]

            # jax.lax.cond(jnp.logical_not(idx), lambda: jax.debug.print("{}", grad.sum()), lambda: None)

            agent_params = agent_params.at[idx].set(policy.step(agent_params[idx], grad))

            return (rng, agent_params), None
        
        carry_out, _ = jax.lax.scan(train_team, (rng, agent_params), None, args.br_length)
        
        agent_params = carry_out[1]
        rng, _rng = jax.random.split(rng)

        # jax.lax.cond(jnp.logical_not(idx), lambda: jax.debug.print("{}", avg_return(_rng, agent_params, idx)), lambda: None)

        return (avg_return(_rng, agent_params)[idx] - base[idx])

    rng = jax.random.split(rng)
    team_gap = jax.vmap(train_single, in_axes=(0, None, 0))(rng, agent_params, jnp.arange(2))

    return jnp.concatenate((team_gap, adv_gap.reshape(1, )))
