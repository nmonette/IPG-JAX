import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .environments import RolloutWrapper, GridVisualizer
from .agents import DirectPolicy
from .utils import parse_args, compute_nash_gap

from functools import partial

import sys, os

def make_train(args):
    def ipg_train_fn(rng):
        # --- Instantiate Policy, Parameterizations, Rollout Manager ---
        param_dims = [args.dim, args.dim, args.dim, args.dim, 2, args.dim, args.dim, 2, 4]
        
        # param_dims = [3, 2]
        policy = DirectPolicy(param_dims, args.lr, args.eps)
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 3)
        agent_params = jax.vmap(policy.init_params)(_rng)
        
        rollout = RolloutWrapper(policy, train_rollout_len=12, 
                                env_kwargs={"dim":args.dim, "max_time":12}
                                )

        # rollout = RolloutWrapper(policy, train_rollout_len=2, 
        #                             env_kwargs={"num_states":3, "num_agents":3, "num_actions":2, "num_timesteps":3},
        #                             env_name="rps",
        #                             gamma=args.gamma
        #                         )
        
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

            return (rng, agent_params), None

        def train_loop(carry, _):
            rng, agent_params = carry
            old = agent_params.copy()
            
            # Run adversary's train loop 
            rng, _rng = jax.random.split(rng)
            (_, adv_params), _  = jax.lax.scan(adv_br, (_rng, agent_params), None, args.br_length)

            agent_params = agent_params.at[-1].set(adv_params[-1])

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

            grads = jax.vmap(lambda p, r, i: outer_loss(p, r, i)[i], in_axes=(None, 0, 0))(agent_params, _rng, jnp.arange(agent_params.shape[0] - 1))

            def apply_grad(carry, idx):
                agent_params = carry

                return agent_params.at[idx].set(policy.step(agent_params[idx], grads[idx])), None
            
            agent_params, _ = jax.lax.scan(apply_grad, agent_params, jnp.arange(agent_params.shape[0] - 1))
            
            rng, _rng = jax.random.split(rng)

            return (rng, agent_params), (agent_params, jnp.linalg.norm(agent_params[:-1] - old[:-1])) # compute_nash_gap(_rng, args, policy, agent_params, rollout)
        
        carry_out, (all_params, diff) = jax.lax.scan(train_loop, (rng, agent_params), jnp.arange(args.iters), args.iters)

        rng, agent_params = carry_out

        # Best Iterate of Team 
        idx = jnp.argmin(diff) - 1
        agent_params = agent_params.at[:-1].set(all_params[jnp.where(idx > 0, idx, 0)][:-1])

        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.env.reset(reset_rng)
        states = rollout.render_rollout(rollout_rng, agent_params, init_obs, init_state, False)

        def collect_gap(rng, team_params, adv_params):
            new_params = jnp.empty_like(agent_params)
            new_params = new_params.at[:-1].set(team_params)
            new_params = new_params.at[-1].set(adv_params)
            rng, _rng = jax.random.split(rng)
            (_, adv_params), _  = jax.lax.scan(adv_br, (_rng, new_params), None, args.br_length)
            new_params = new_params.at[-1].set(adv_params[-1])
            return compute_nash_gap(rng, args, policy, new_params, rollout)
        
        nash_gap = jnp.max(jax.vmap(collect_gap)(jax.random.split(rng, len(all_params)), jnp.cumsum(all_params[:, :-1], axis=0) /  jnp.cumsum(jnp.ones_like(all_params[:, :-1]), axis=0), all_params[:, -1]), axis=1)

        return jnp.cumsum(diff) / jnp.arange(len(diff), dtype=jnp.float32), diff, agent_params, states, nash_gap

    return ipg_train_fn

def main(args):
    rng = jax.random.key(args.seed)
    train_fn = make_train(args)
    import time 
    start = time.time()
    with jax.numpy_dtype_promotion('strict'):
        fn = jax.jit(train_fn)
        cum_dist, dist, agent_params, states, nash_gap = fn(rng)
        cum_dist.block_until_ready()

    print(time.time() - start)
    # print("Nash Gap: ", nash_gap)

    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")

    # with jax.disable_jit(True):
    #     states_new = []
    #     for i in range(states.done.shape[0]):
    #         states_new.append(jax.tree_util.tree_map(lambda v: v[i], states))
    #     gv = GridVisualizer({"dim": args.dim, "max_time":12}, states_new, None)
    #     gv.animate(f"output/experiment-{experiment_num}/game.gif", view=True)

    for agent in range(len(agent_params)):
        jnp.save(f"output/experiment-{experiment_num}/agent{agent+1}", agent_params[agent])

    # nash_gap = jnp.max(nash_gap, 1)
    plt.plot(nash_gap)
    plt.xlabel("Iterations")
    plt.title("Nash Gap")

    plt.savefig(f"output/experiment-{experiment_num}/nash-gap")
    plt.close()

    plt.plot(cum_dist)
    plt.xlabel("Iterations")
    plt.title("Cumulative Avg Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/cumulative-distance")
    plt.close()
    
    plt.plot(dist)
    plt.xlabel("Iterations")
    plt.title("Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/distance")
    plt.close()

if __name__ == "__main__":
    main()
                

                

