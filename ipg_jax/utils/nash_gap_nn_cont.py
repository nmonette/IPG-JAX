import jax
import jax.numpy as jnp

def compute_nash_gap(rng, args, policy, adv_params, team_params, rollout, optimizer, adv_opt_state, team_opt_states):
    adv_state_space = [1, 1, 1, 1, 1, 1, 1, 1]

    def avg_return(rng, adv_params, team_params, idx):
        
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
        data, _, _, _ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state)

        gamma = jnp.cumprod(jnp.full((data.reward.shape[1], ), args.gamma)) / args.gamma
        
        return jnp.dot(jnp.float32(data.reward[:,:,idx]), gamma).mean(axis=0)
    
    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(rng, 3)
    base = jax.vmap(avg_return, in_axes=(0, None, None, 0))(_rng, adv_params, team_params, jnp.arange(3))

    # Calculate adversarial best response
    def adv_br(carry, _):
        rng, adv_params, team_params, adv_opt_state, team_opt_states = carry

        # Calculate Gradients
        @jax.grad
        def loss(adv_params, team_params, rng):
            # Collect rollouts
            rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
            init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
            data, _, _, _ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state)

            lambda_data = jnp.concatenate([data.obs["adversary_0"].reshape(-1, len(adv_state_space)), data.action[:,:,-1].reshape(-1, 1)], axis=-1)
            lambda_weights = jnp.repeat(jnp.cumprod(jnp.full((25, ), args.gamma)) / args.gamma, args.rollout_length)
            lambda_ = jax.scipy.stats.gaussian_kde(lambda_data.T, weights=lambda_weights)

            def inner_loss(data):

                fn = lambda r, l, lp: (r - args.nu * l) * lp
                
                idx = jnp.concatenate([data.obs["adversary_0"].reshape(-1, len(adv_state_space)), data.action[:,-1].reshape(-1, 1)], axis=-1)
                lambdas = lambda_.evaluate(idx.T)

                loss = jax.vmap(fn)(jnp.float32(data.reward[:, -1]), lambdas, jnp.cumsum(data.log_probs[:, -1]))

                disc = jnp.cumprod(jnp.ones_like(loss) * args.gamma) / args.gamma
                return jnp.dot(loss, disc)
            
            return -jax.vmap(inner_loss)(data).mean()

        rng, _rng = jax.random.split(rng)
        grad = jax.tree_map(lambda x: x[-1], loss(adv_params, team_params, _rng))
        adv_params, adv_opt_state = policy.step2(adv_params, grad, optimizer, adv_opt_state)

        return (rng, adv_params, team_params, adv_opt_state, team_opt_states), None
        
    # Run adversary's train loop 
    rng, _rng = jax.random.split(rng)
    carry_out, _  = jax.lax.scan(adv_br, (_rng, adv_params, team_params, adv_opt_state, team_opt_states), None, args.br_length)

    rng, adv_params, team_params, adv_opt_state, team_opt_states = carry_out

    rng, _rng = jax.random.split(rng)
    adv_gap = (avg_return(_rng, adv_params, team_params, -1) - base[-1])
    
    # Update team
    def train_single(rng, adv_params, team_params, adv_opt_state, team_opt_states, idx):
        def train_team(carry, _):
            rng, adv_params, team_params, adv_opt_state, team_opt_states = carry

            @jax.grad
            def outer_loss(team_params, adv_params, rng):
                rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
                init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)
                data, _, _, _ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state)
                
                def episode_loss(rewards, log_probs):

                    def inner_returns(carry, i):
                        returns = carry

                        return returns.at[i].set(args.gamma * returns[i + 1] + rewards[i]), None
                        
                    returns, _ = jax.lax.scan(inner_returns, (jnp.zeros_like(rewards).at[-1].set(rewards[-1])), jnp.arange(rewards.shape[0]), reverse=True)

                    return jnp.dot(log_probs, returns)

                return -jax.vmap(episode_loss)(jnp.float32(data.reward[:,:,idx]), data.log_probs[:,:,idx]).mean()

            rng, _rng = jax.random.split(rng)
            grad = jax.tree_map(lambda x: x[idx], outer_loss(team_params, adv_params, _rng))

            # jax.lax.cond(jnp.logical_not(idx), lambda: jax.debug.print("{}", grad.sum()), lambda: None)
           
            team_params, team_opt_states = policy.step(team_params, grad, optimizer, team_opt_states, idx)

            return (rng, adv_params, team_params, adv_opt_state, team_opt_states), None
        
        carry_out, _ = jax.lax.scan(train_team, (rng, adv_params, team_params, adv_opt_state, team_opt_states), None, args.br_length)
        
        adv_params, team_params = carry_out[1], carry_out[2]
        rng, _rng = jax.random.split(rng)

        return (avg_return(_rng, adv_params, team_params, idx) - base[idx])

    rng = jax.random.split(rng)
    team_gap = jax.vmap(train_single, in_axes=(0, None, None, None, None, 0))(rng, adv_params, team_params, adv_opt_state, team_opt_states, jnp.arange(2))

    return jnp.concatenate((team_gap, adv_gap.reshape(1, )))
