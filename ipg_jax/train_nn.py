import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from optax import adam

from .environments import RolloutWrapper, GridVisualizer
from .agents import SELUPolicy
from .utils import compute_nash_gap_nn as compute_nash_gap

import os
from functools import partial
import pickle

def make_train(args):
    def ipg_train_fn(rng):
        # --- Instantiate Policy, Parameterizations, Rollout Manager ---
        state_space = [args.dim, args.dim, args.dim, args.dim, 2, args.dim, args.dim, 2] # [1] 
        state_action_space =  [args.dim, args.dim, args.dim, args.dim, 2, args.dim, args.dim, 2, 4] # [1, 2] #

        policy = SELUPolicy(args.eps, args.net_arch + [4], state_space)
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 3)

        ex_state = jnp.array(state_space) - 1
        agent_params = jax.vmap(policy.init, in_axes=(0, None))(_rng, ex_state)

        optimizer = adam(args.lr)
        optimizer_states = jax.vmap(optimizer.init)(agent_params)

        rollout = RolloutWrapper(policy, train_rollout_len=12, 
                                env_kwargs={"dim":args.dim, "max_time":12},
                                state_action_space=state_action_space,
                                gamma=args.gamma,
                                )
        
        # rollout = RolloutWrapper(policy, train_rollout_len=1, 
        #                             state_action_space=state_action_space,
        #                             env_kwargs={"num_states":1, "num_agents":3, "num_actions":2, "num_timesteps":1},
        #                             env_name="rps",
        #                             gamma=args.gamma
        #                         )
        
        def train_loop(carry, _):
            rng, agent_params, optimizer_states = carry

            old_weights = jax.tree_map(lambda x: x[:-1], agent_params)
            old_leaves = jnp.concatenate([i.flatten() for i in jax.tree_util.tree_leaves(old_weights)]).copy()

            # Calculate adversarial best response
            def adv_br(carry, _):
                rng, agent_params, optimizer_states = carry

                # Calculate Gradients
                @jax.grad
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
                        return jnp.dot(loss, disc)
                    
                    return -jax.vmap(inner_loss)(data).mean()

                rng, _rng = jax.random.split(rng)
                grad = jax.tree_map(lambda x: x[-1], loss(agent_params, _rng))
                agent_params, optimizer_states = policy.step(agent_params, grad, optimizer, optimizer_states, -1)

                return (rng, agent_params, optimizer_states), None
            
            # Run adversary's train loop 
            rng, _rng = jax.random.split(rng)
            carry_out, _  = jax.lax.scan(adv_br, (_rng, agent_params, optimizer_states), None, args.br_length)

            rng, agent_params, optimizer_states = carry_out
            
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

                return jax.vmap(episode_loss, in_axes=(0, 0))(data.log_probs[:,:, idx], jnp.float32(data.reward[:,:, idx])).mean()

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, rollout.num_agents - 1)

            grads = jax.vmap(lambda p, r, i: jax.tree_map(lambda x: x[i], outer_loss(p,r,i)), in_axes=(None, 0, 0))(agent_params, _rng, jnp.arange(rollout.num_agents - 1))
            def apply_grad(carry, grad):
                agent_params, optimizer_states, idx = carry
                agent_params, optimizer_states = policy.step(agent_params, grad, optimizer, optimizer_states, idx)
                return (agent_params, optimizer_states, idx+1), None
            
            jax.debug.print("{}", grads)
            
            carry_out, _ = jax.lax.scan(apply_grad, (agent_params, optimizer_states, 0), grads)
            agent_params, optimizer_states, _ = carry_out

            rng, _rng = jax.random.split(rng)

            team_weights = jax.tree_map(lambda x: x[:-1], agent_params)
            team_leaves = jnp.concatenate([i.flatten() for i in jax.tree_util.tree_leaves(team_weights)])
            return (rng, agent_params, optimizer_states), (agent_params, jnp.linalg.norm(team_leaves - old_leaves), optimizer_states) # compute_nash_gap(_rng, args, policy, agent_params, rollout)
        
        carry_out, (all_params, diff, all_optimizer_states) = jax.lax.scan(train_loop, (rng, agent_params, optimizer_states), jnp.arange(args.iters), args.iters)

        rng, agent_params, optimizer_states = carry_out

        # Best Iterate of Team 
        idx = jnp.argmin(diff) - 1
        idx = jnp.where(idx > 0, idx, 0)
        team_weights = jax.tree_map(lambda x: x[idx][:-1], all_params)
        team_opt_states = jax.tree_map(lambda x: x[idx][:-1], all_optimizer_states)

        def get_final_team(carry, var):
            new_team, new_opt = var
            agent_params, optimizer_states, idx = carry
            agent_params = policy.tree_change_at_idx(agent_params, new_team, idx)
            optimizer_states = policy.tree_change_at_idx(optimizer_states, new_opt, idx)
            return (agent_params, optimizer_states, idx+1), None

        (agent_params, optimizer_states, _), _ = jax.lax.scan(get_final_team, (agent_params, optimizer_states, 0), (team_weights, team_opt_states))

        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.env.reset(reset_rng)
        states = rollout.render_rollout(rollout_rng, agent_params, init_obs, init_state, False)

        return jnp.cumsum(diff) / jnp.arange(len(diff), dtype=jnp.float32), diff, agent_params, states, compute_nash_gap(rng, args, policy, agent_params, rollout, optimizer, optimizer_states)

    return ipg_train_fn

def main(args):
    rng = jax.random.key(args.seed)
    train_fn = make_train(args)
    import time 
    start = time.time()
    # with jax.numpy_dtype_promotion('strict'):
    fn = jax.jit(train_fn)
    cum_dist, dist, agent_params, states, nash_gap = fn(rng)
    cum_dist.block_until_ready()

    print(time.time() - start)

    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")

    # with jax.disable_jit(True):
    #     states_new = []
    #     for i in range(states.done.shape[0]):
    #         states_new.append(jax.tree_util.tree_map(lambda v: v[i], states))
    #     gv = GridVisualizer({"dim": args.dim, "max_time":12}, states_new, None)
        # gv.animate(f"output/experiment-{experiment_num}/game.gif", view=True)

    for agent in range(3):
        params = jax.tree_map(lambda x: x[agent], agent_params)
        with open(f"output/experiment-{experiment_num}/agent{agent+1}.pickle", 'wb') as file:
            pickle.dump(params, file)

    plt.plot(cum_dist)
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Avg Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/cumulative-distance")
    plt.close()
    
    plt.plot(dist)
    plt.xlabel("Iterations")
    plt.ylabel("Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/distance")
    plt.close()
    
    

if __name__ == "__main__":
    main()
                

                

