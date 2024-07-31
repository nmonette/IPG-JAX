import jax
import jax.numpy as jnp

def make_reinforce(args, rollout):
    # Update team
    @jax.grad
    def outer_loss(team_params, adv_params, rng, idx):
        # Collect rollouts
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        init_obs, init_state = rollout.batch_reset(reset_rng, args.tr)
        data, _, _, _ = rollout.batch_rollout(rollout_rng, adv_params, team_params, init_obs, init_state, adv=False)

        def episode_loss(log_probs, rewards):
            disc = jnp.cumprod(jnp.full_like(rewards, args.gamma)) / args.gamma
            cs = jnp.cumsum(log_probs)

            rlp = jnp.dot(rewards, cs)
            return jnp.dot(rlp, disc)

        return -jax.vmap(episode_loss, in_axes=(0, 0))(data.log_probs[:,:, idx], jnp.float32(data.reward[:,:, idx])).mean()
    
    return outer_loss