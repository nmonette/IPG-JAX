import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from optax import adam

from .environments import RolloutWrapper_cont as RolloutWrapper
from .agents import SELUPolicy
from .utils import compute_nash_gap_nn_cont as compute_nash_gap

import os
from functools import partial
import pickle

def make_train(args):
    def ipg_train_fn(rng):
        # --- Instantiate Policy, Parameterizations, Rollout Manager ---
        adv_state_space = [1, 1, 1, 1, 1, 1, 1, 1]
        team_state_space = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        policy = SELUPolicy(args.eps, args.net_arch + [4], "c")
        rng, team_rng, adv_rng = jax.random.split(rng, 3)
        _rng = jax.random.split(team_rng, 2)

        ex_team_state = jnp.array(team_state_space) - 1
        team_params = jax.vmap(policy.init, in_axes=(0, None))(_rng, ex_team_state)

        ex_adv_state = jnp.array(adv_state_space) - 1
        adv_params = policy.init(adv_rng, ex_adv_state)

        optimizer = adam(args.lr)
        team_opt_states = jax.vmap(optimizer.init)(team_params)
        adv_opt_state = optimizer.init(adv_params)

        rollout = RolloutWrapper(policy, train_rollout_len=25, 
                                env_kwargs={"zero_sum":True},
                                state_action_space=adv_state_space,
                                gamma=args.gamma,
                                )

        def train_loop(carry, _):
            rng, adv_params, team_params, adv_opt_state, team_opt_states = carry
            
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
                    disc = jnp.cumprod(jnp.full((25, ), args.gamma)) / args.gamma
                    lambda_weights = jnp.concatenate([disc for _ in range(args.rollout_length)])
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
            
            # Update team
            @jax.grad
            def outer_loss(agent_params, rng, idx):
                # Collect rollouts
                rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
                init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
                data, _, _, _ = rollout.batch_rollout(rollout_rng, agent_params, init_obs, init_state, adv=False)

                def episode_loss(log_probs, rewards):
                    disc = jnp.cumprod(jnp.full_like(rewards, args.gamma)) / args.gamma
                    cs = jnp.cumsum(log_probs)

                    rlp = jnp.dot(rewards, cs)
                    return jnp.dot(rlp, disc)

                return -jax.vmap(episode_loss, in_axes=(0, 0))(data.log_probs[:,:, idx], jnp.float32(data.reward[:,:, idx])).mean()

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, rollout.num_agents - 1)

            grads = jax.vmap(lambda t, a, r, i: jax.tree_map(lambda x: x[i], outer_loss(t, a, r,i)), in_axes=(None, None, 0, 0))(team_params, adv_params, _rng, jnp.arange(rollout.num_agents - 1))
            
            def apply_grad(carry, grad):
                team_params, team_opt_states, idx = carry
                team_params, team_opt_states = policy.step(team_params, grad, optimizer, team_opt_states, idx)
                return (team_params, team_opt_states, idx+1), None
            
            carry_out, _ = jax.lax.scan(apply_grad, (team_params, team_opt_states, 0), grads)
            team_params, team_opt_states, _ = carry_out

            rng, _rng = jax.random.split(rng)

            return (rng, adv_params, team_params, adv_opt_state, team_opt_states), compute_nash_gap(_rng, args, policy, adv_params, team_params, rollout, optimizer, adv_opt_state, team_opt_states)
    
        carry_out, nash_gap = jax.lax.scan(train_loop, (rng, adv_params, team_params, adv_opt_state, team_opt_states), jnp.arange(args.iters), args.iters)

        rng, adv_params, team_params, adv_opt_state, team_opt_states = carry_out

        return nash_gap, adv_params, team_params

    return ipg_train_fn


def main(args):
    rng = jax.random.key(args.seed)
    train_fn = make_train(args)
    import time 
    start = time.time()
    # with jax.numpy_dtype_promotion('strict'):
    fn = jax.jit(train_fn)
    nash_gap, adv_params, team_params = fn(rng)
    nash_gap.block_until_ready()

    print(time.time() - start)

    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")

    for agent in range(2):
        params = jax.tree_map(lambda x: x[agent], team_params)
        with open(f"output/experiment-{experiment_num}/agent{agent+1}.pickle", 'wb') as file:
            pickle.dump(params, file)

    with open(f"output/experiment-{experiment_num}/agent3.pickle", 'wb') as file:
            pickle.dump(adv_params, file)

    nash_gap = jnp.max(nash_gap, 1)
    plt.plot(nash_gap)
    plt.xlabel("Iterations")
    plt.ylabel("Nash Gap")

    plt.savefig(f"output/experiment-{experiment_num}/nash-gap")
    
    

if __name__ == "__main__":
    main()
                

                

