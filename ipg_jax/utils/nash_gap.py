import jax
import jax.numpy as jnp


def compute_nash_gap(rng, args, policy, agent_params, rollout):

    rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
    data, _, _, _ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=False)
    
    base = data.reward.mean((0, 1))
        
    # Calculate adversarial best response
    def adv_br(carry, _):
        rng, agent_params = carry

        # Collect rollouts
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
        data, _, _, _, lambda_ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=True)

        lambda_ = lambda_.mean(axis=0)
        
        # Calculate Gradients
        @jax.grad
        def loss(params, data):

            def inner_loss(data):
                
                fn = lambda r, l, idx: (r - args.nu * l) * jnp.log(params[tuple(idx)])

                idx = jnp.concatenate((data.obs[:, -1], data.action[:, -1].reshape(data.action.shape[0], -1)), axis=-1)
                idx_fn = lambda idx: lambda_[tuple(idx)]
                lambdas = jax.vmap(idx_fn)(idx)

                loss = jax.vmap(fn)(jnp.float32(data.reward[:, -1]), lambdas, idx)

                disc = jnp.cumprod(jnp.ones_like(loss) * args.gamma) / args.gamma
                return jnp.dot(loss, disc)
            
            return jax.vmap(inner_loss)(data).mean()

        grad = loss(agent_params[-1], data)
        agent_params = agent_params.at[-1].set(policy.step(agent_params[-1], grad))

        return (rng, agent_params), None
        
    # Run adversary's train loop 
    rng, _rng = jax.random.split(rng)
    carry_out, _  = jax.lax.scan(adv_br, (_rng, agent_params), None, args.br_length)

    rng, adv_params = carry_out

    rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
    data, _, _, _ = rollout.batch_rollout(rollout_rng, adv_params, init_obs, init_state, adv=False)
    
    adv_gap = (data.reward.mean((0, 1)) - base)[-1]

    # Update team
    def train_single(rng, agent_params, idx):
        def train_team(carry, _):
            rng, agent_params, idx = carry

            rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
            init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
            data, _, _, _ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=False)

            @jax.grad
            def outer_loss(params, data, idx):
                
                def episode_loss(params, rewards, states, actions):

                    def inner_returns(carry, i):
                        returns = carry

                        return returns.at[i].set(args.gamma * returns[i + 1] + rewards[i]), None
                        
                    returns, _ = jax.lax.scan(inner_returns, (jnp.zeros_like(rewards).at[-1].set(rewards[-1])), jnp.arange(rewards.shape[0]), reverse=True)

                    idx = jnp.concatenate((states, actions.reshape(actions.shape[0], -1)), axis=-1)
                    idx_fn = lambda idx: params[tuple(idx)]
                    log_probs = jax.vmap(idx_fn)(idx)

                    return jnp.dot(log_probs, returns)
                
                return jax.vmap(episode_loss, in_axes=(None, 0, 0, 0))(params, jnp.float32(data.reward[:, idx]), data.obs[:, idx], data.action[:, idx]).mean()

            grad = outer_loss(agent_params[idx], data, idx)

            agent_params = agent_params.at[idx].set(policy.step(agent_params[idx], grad))

            return (rng, agent_params, idx), None
        
        carry_out, _ = jax.lax.scan(train_team, (rng, agent_params, idx), None, args.br_length)
        
        agent_params = carry_out[1]
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
        data, _, _, _ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=False)

        return (data.reward.mean((0, 1)) - base)[idx]

    rng = jax.random.split(rng, 2)
    team_gap = jax.vmap(train_single, in_axes=(0, None, 0))(rng, agent_params, jnp.arange(2))

    return jnp.concatenate((team_gap, adv_gap.reshape(1, )))
