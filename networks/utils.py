from acme.agents.jax import r2d2

def make_r2d2_networks(batch_size, env_spec, NetworkCls, NetKwargs):
  """Builds default R2D2 networks."""

  def forward_fn(x, s):
    model = NetworkCls(**NetKwargs)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = NetworkCls(**NetKwargs)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = NetworkCls(**NetKwargs)
    return model.unroll(inputs, state)

  return r2d2.make_networks(env_spec=env_spec, forward_fn=forward_fn,
                       initial_state_fn=initial_state_fn, unroll_fn=unroll_fn,
                       batch_size=batch_size)
