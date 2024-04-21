import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from environments import RolloutWrapper, GridVisualizer
from agents.policy import DirectPolicy
from utils import parse_args, compute_nash_gap

import sys, os

def make_train(args):
    def ipg_train_fn(rng):
        # --- Instantiate Policy, Parameterizations, Rollout Manager ---
        param_dims = [args.dim, args.dim, args.dim, args.dim, 2, args.dim, args.dim, 2, 4]

        policy = DirectPolicy(param_dims, args.lr, args.eps)
        agent_params = jax.vmap(policy.init_params)(jnp.arange(3))
        
        rollout = RolloutWrapper(policy, train_rollout_len=12, 
                                env_kwargs={"dim":args.dim, "max_time":12}
                                )

        def train_loop(carry, iter):
            rng, agent_params = carry
            
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

            rng, agent_params = carry_out

            # Update team
            # Collect rollouts
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

            grads = jax.vmap(outer_loss, in_axes=(0, None, 0))(agent_params[:agent_params.shape[0] - 1], data, jnp.arange(agent_params.shape[0] - 1))

            def apply_grad(carry, idx):
                agent_params = carry

                return agent_params.at[idx].set(policy.step(agent_params[idx], grads[idx])), None
            
            agent_params, _ = jax.lax.scan(apply_grad, agent_params, jnp.arange(agent_params.shape[0] - 1))
            
            rng, _rng = jax.random.split(rng)

            return (rng, agent_params), compute_nash_gap(_rng, args, policy, agent_params, rollout)

        carry_out, nash_gap = jax.lax.scan(train_loop, (rng, agent_params), jnp.arange(args.iters), args.iters)

        rng, agent_params = carry_out

        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.env.reset(reset_rng)
        states = rollout.render_rollout(rollout_rng, agent_params, init_obs, init_state, False)

        return nash_gap, agent_params, states

    return ipg_train_fn

def main(cmd=sys.argv[1:]):
    args = parse_args(cmd)
    rng = jax.random.key(args.seed)
    train_fn = make_train(args)
    with jax.numpy_dtype_promotion('strict'):
        nash_gap, agent_params, states = jax.jit(train_fn)(rng)


    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")

    with jax.disable_jit(True):
        states_new = []
        for i in range(states.done.shape[0]):
            states_new.append(jax.tree_util.tree_map(lambda v: v[i], states))
        gv = GridVisualizer({"dim": args.dim, "max_time":12}, states_new, None)
        gv.animate(f"output/experiment-{experiment_num}/game.gif", view=True)

    for agent in range(len(agent_params)):
        jnp.save(f"output/experiment-{experiment_num}/agent{agent+1}", agent_params[agent])

    nash_gap = jnp.max(nash_gap, 1)
    plt.plot(nash_gap)
    plt.xlabel("Iterations")
    plt.ylabel("Nash Gap")

    plt.savefig(f"output/experiment-{experiment_num}/nash-gap")
    
    

if __name__ == "__main__":
    main()
                

                

