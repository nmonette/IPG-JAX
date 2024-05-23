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

    def gap_fn(rng, agent_params, adv_idx):
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

                    idx = jnp.concatenate((data.obs[:, adv_idx], data.action[:, adv_idx].reshape(data.action.shape[0], -1)), axis=-1)
                    idx_fn = lambda idx: lambda_[tuple(idx)]
                    lambdas = jax.vmap(idx_fn)(idx)

                    loss = jax.vmap(fn)(jnp.float32(data.reward[:, adv_idx]), lambdas, jnp.cumsum(data.log_probs[:, adv_idx]))

                    disc = jnp.cumprod(jnp.ones_like(loss) * args.gamma) / args.gamma
                    return jnp.dot(loss, disc), jnp.dot(data.reward[:, adv_idx], disc)
                        
                grad, val =  jax.vmap(inner_loss)(data)
                
                return grad.mean(), val.mean()

            rng, _rng = jax.random.split(rng)
            grad, val = loss(agent_params, _rng)
            grad = grad[adv_idx]
            agent_params = agent_params.at[adv_idx].set(policy.step(agent_params[adv_idx], grad))

            return (rng, agent_params), (agent_params[adv_idx], val)

        # Run adversary's train loop 
        rng, _rng = jax.random.split(rng)
        _, (adv_params_new, val)  = jax.lax.scan(adv_br, (_rng, agent_params), None, args.br_length)

        # Best Iterate of Adversary BR 
        idx = jnp.argmax(val) - 1
        return avg_return(rng, agent_params.at[adv_idx].set(adv_params_new[jnp.where(idx > 0, idx, 0)]))[adv_idx] - base[adv_idx]

    return jax.vmap(gap_fn, in_axes=(0, None, 0))(jax.random.split(rng, rollout.num_agents), agent_params, jnp.arange(rollout.num_agents))
