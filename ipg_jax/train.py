import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from optax import adam

from .environments import RolloutWrapper, GridVisualizer
from .agents import DirectPolicy, SELUPolicy
from .utils import parse_args, compute_nash_gap
from .br import make_adv_br
from .reinforce import make_reinforce

from functools import partial

import os, pickle

TIME_HORIZON = 1
NUM_STATES = 1
NUM_AGENTS = 3

def make_train(args):

    # Selecting which TrainState we want to use
    if args.param == "direct":
        from .agents.direct import TrainState
    else:
        from .agents.nn import TrainState

    def ipg_train_fn(rng):
        # --- Instantiate Policy, Parameterizations, Rollout Manager ---
        # state_action_space = [args.dim, args.dim, args.dim, args.dim, 2, args.dim, args.dim, 2, 4]
        num_agents = NUM_AGENTS
        team_state_action_space = adv_state_action_space = [1, 2]

        if args.param == "direct":
            team_policy = DirectPolicy(team_state_action_space, args.lr, args.eps)
            adv_policy = DirectPolicy(adv_state_action_space, args.lr, args.eps)

            rng, team_rng, adv_rng = jax.random.split(rng, 3)
            team_rng = jax.random.split(team_rng, num_agents - 1)
            team_params = jax.vmap(team_policy.init_params)(team_rng)
            adv_params = adv_policy.init_params(adv_rng)
            
            train_state = TrainState(
                team_params=team_params,
                adv_params=adv_params
            )

        elif args.param in ["nn", "c"]:
            team_policy = SELUPolicy(args.eps, args.net_arch + [team_state_action_space[-1]], team_state_action_space[:-1])
            adv_policy = SELUPolicy(args.eps, args.net_arch + [adv_state_action_space[-1]], adv_state_action_space[:-1])
            team_optimizer = adv_optimizer = adam(args.lr)
            
            rng, team_rng, adv_rng = jax.random.split(rng, 3)
            team_rng = jax.random.split(team_rng, num_agents - 1)

            ex_state = jnp.zeros_like(jnp.array(team_state_action_space[:-1]))
            team_params = jax.vmap(team_policy.init, in_axes=(0, None))(team_rng, ex_state)
            team_opt_states = jax.vmap(team_optimizer.init)(team_params)
            
            ex_state = jnp.zeros_like(jnp.array(adv_state_action_space[:-1]))
            adv_params = adv_policy.init(adv_rng, ex_state)
            adv_opt_state = adv_optimizer.init(adv_params)

            train_state = TrainState(
                team_params=team_params,
                adv_params=adv_params,
                team_optimizer=team_optimizer,
                adv_optimizer=adv_optimizer,
                team_opt_states=team_opt_states,
                adv_opt_state=adv_opt_state
            )

        else:
            raise NotImplementedError("Parameterization not implemented")
        
        rollout = RolloutWrapper(adv_policy, team_policy, train_rollout_len=TIME_HORIZON, 
                                    env_kwargs={"num_states":NUM_STATES, "num_agents":NUM_AGENTS, "num_actions":team_state_action_space[-1], "num_timesteps":TIME_HORIZON},
                                    env_name="rps",
                                    gamma=args.gamma,
                                    state_action_space=adv_state_action_space
                                )
        
        # rollout = RolloutWrapper(adv_policy, team_policy train_rollout_len=12, 
        #                         env_kwargs={"dim":args.dim, "max_time":12}
        #                         )

        # Get update functions
        adv_br = make_adv_br(args, rollout, adv_state_action_space[:-1], adv_policy) 
        reinforce = make_reinforce(args, rollout)

        def train_loop(carry, _):
            rng, train_state = carry
            old = train_state
            
            rng, _rng = jax.random.split(rng)
            train_state = adv_br(_rng, train_state)

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, rollout.num_agents - 1)

            grads = jax.vmap(lambda t, a, r, i: team_policy.get_agent_params(reinforce(t, a, r, i), i), in_axes=(None, None, 0, 0))(train_state.team_params, train_state.adv_params, _rng, jnp.arange(rollout.num_agents - 1))

            train_state = train_state.update_team(team_policy, train_state, grads, slice(None))
            
            rng, _rng = jax.random.split(rng)

            return (rng, train_state), (train_state, team_policy.team_diff(train_state.team_params, old.team_params))
        
        carry_out, (all_train_states, diff) = jax.lax.scan(train_loop, (rng, train_state), jnp.arange(args.iters), args.iters)

        rng, train_state = carry_out

        # Best Iterate of Team 
        idx = jnp.argmin(diff) - 1
        idx = jnp.where(idx > 0, idx, 0)
        train_state = jax.tree_map(lambda x: x[idx], all_train_states)

        def collect_gap(rng, train_state):
            rng, _rng = jax.random.split(rng)
            train_state = adv_br(_rng, train_state)
            return compute_nash_gap(rng, args, adv_policy, team_policy, rollout, train_state, adv_state_action_space[:-1])
        
        avg_params = all_train_states.replace(
            team_params = jax.tree_map(lambda x: jnp.cumsum(x, axis=0) /  jnp.cumsum(jnp.ones_like(x), axis=0), all_train_states.team_params)
        )   
        nash_gap = jnp.max(jax.vmap(collect_gap)(jax.random.split(rng, args.iters), avg_params), axis=1)
        cum_diff = jax.vmap(team_policy.team_diff)(jax.tree_map(lambda x: x[1:], avg_params.team_params), jax.tree_map(lambda x: x[:-1], avg_params.team_params)) # jnp.cumsum(diff) / jnp.arange(len(diff), dtype=jnp.float32)
        return cum_diff, diff, adv_params, team_params, nash_gap

    return ipg_train_fn

def main(args):
    rng = jax.random.key(args.seed)
    train_fn = make_train(args)
    import time 
    start = time.time()
    # with jax.numpy_dtype_promotion('strict'):
    fn = jax.jit(train_fn)
    cum_dist, dist, adv_params, team_params, nash_gap = fn(rng) 
    cum_dist.block_until_ready()

    print(time.time() - start)
    # print("Nash Gap: ", nash_gap)

    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}", exist_ok=True)

    # with jax.disable_jit(True):
    #     states_new = []
    #     for i in range(states.done.shape[0]):
    #         states_new.append(jax.tree_util.tree_map(lambda v: v[i], states))
    #     gv = GridVisualizer({"dim": args.dim, "max_time":12}, states_new, None)
    #     gv.animate(f"output/experiment-{experiment_num}/game.gif", view=True)

    for agent in range(2):
        params = jax.tree_map(lambda x: x[agent], team_params)
        with open(f"output/experiment-{experiment_num}/agent{agent+1}.pickle", 'wb') as file:
            pickle.dump(params, file)

    with open(f"output/experiment-{experiment_num}/agent3.pickle", 'wb') as file:
            pickle.dump(adv_params, file)

    return nash_gap, cum_dist, dist

if __name__ == "__main__":
    main()
                

                

