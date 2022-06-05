import jax.numpy as jnp
import jax
def compute_q(sf, w):
  return jnp.sum(sf*w, axis=-1)



class MsfaStats:

  def __call__(self, data, online_preds, target_preds, steps, **kwargs):

    w = online_preds.w  # [T, B, C]
    cumulants = online_preds.cumulants  # predicted  [T, B, D]
    online_sf = online_preds.sf[:,:,0]  # [T, B, A, C]
    target_sf = target_preds.sf[:,:,0]  # [T, B, A, C]


    # -----------------------
    # means across modules
    # -----------------------
    T, B, N, D = online_preds.memory_out.hidden.shape

    # [T, B, A, C] --> [T, B, A, N, C/N]
    online_sf = jnp.stack(jnp.split(online_sf, N, axis=-1), axis=3)

    # [T, B, D] --> [T, B, N, D/N]
    w = jnp.stack(jnp.split(w, N, axis=-1), axis=2)
    cumulants = jnp.stack(jnp.split(cumulants, N, axis=-1), axis=2)

    metrics={}
    compute_q_jax = jax.vmap(compute_q, in_axes=(2, None), out_axes=2)  # over A
    for idx in range(N):
      w_i = w[:, :, idx] # [T, B, C]
      phi_i = cumulants[:, :, idx] # [T, B, C]
      sf_i = online_sf[:, :, idx] # [T, B, A, C]
      q_i =  compute_q_jax(sf_i, w_i) # [T, B, A]

      metrics.update({
        f'z.w_mean_{idx}': w_i.mean(),
        f'z.q_mean_{idx}': q_i.mean(),
        f'z.phi_mean_{idx}': phi_i.mean(),
        f'z.phi_l1_{idx}': jnp.linalg.norm(phi_i, ord=1, axis=-1).mean(),
        f'z.sf_mean_{idx}': sf_i.mean(),
        f'z.sf_max_{idx}': sf_i.max(),
        f'z.sf_min_{idx}': sf_i.min(),
        f'z.sf_l1_{idx}': jnp.linalg.norm(sf_i, ord=1, axis=-1).mean(),
        })

    return 0.0, metrics

