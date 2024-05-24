import jax
import jax.numpy as jnp

def make_adv_br(args, rollout, adv_state_space, policy):

    def get_br(rng, train_state, agent_idx = -1):
        def adv_br(carry, _):
            rng, train_state = carry

            # Calculate Gradients
            @jax.grad
            def loss(adv_params, team_params, rng):
                # Collect rollouts
                rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
                init_obs, init_state = rollout.batch_reset(reset_rng, args.rollout_length)

                if args.param != "c":
                    data, _, _, _, lambda_ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state, adv=True)
                    lambda_ = lambda_.mean(axis=0)
                    evaluate_lambda = lambda o, a: lambda_[tuple(o)][a]
                else:
                    data, _, _, _, lambda_ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state)
                    lambda_data = jnp.concatenate([data.obs["adversary_0"].reshape(-1, len(adv_state_space)), data.action[:,:,-1].reshape(-1, 1)], axis=-1)
                    disc = jnp.cumprod(jnp.full((rollout.train_rollout_len, ), args.gamma)) / args.gamma
                    lambda_weights = jnp.concatenate([disc for _ in range(args.rollout_length)])
                    lambda_ = jax.scipy.stats.gaussian_kde(lambda_data.T, weights=lambda_weights)
                    evaluate_lambda = lambda o, a: lambda_.evaluate(jnp.concatenate([o, a.reshape(1)]))

                def inner_loss(data):
                    
                    fn = lambda r, l, lp: (r - args.nu * l) * lp 

                    lambdas = jax.vmap(evaluate_lambda)(data.obs[:, agent_idx], jnp.int32(data.action[:, agent_idx]))

                    loss = jax.vmap(fn)(jnp.float32(data.reward[:, agent_idx]), lambdas, jnp.cumsum(data.log_probs[:, agent_idx]))

                    disc = jnp.cumprod(jnp.ones_like(loss) * args.gamma) / args.gamma
                    return jnp.dot(loss, disc)
                                
                return jax.vmap(inner_loss)(data).mean()
            
            rng, _rng = jax.random.split(rng)
            grad = loss(train_state.adv_params, train_state.team_params, _rng)

            train_state = train_state.update_adv(policy, train_state, grad)

            return (rng, train_state), None
        
        (_, train_state), _  = jax.lax.scan(adv_br, (rng, train_state), None, args.br_length)
        return train_state
        
    return get_br