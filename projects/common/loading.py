from glob import glob
import os.path

from acme.agents import agent as acme_agent
from acme.jax import savers
from acme.jax.layouts import distributed_layout

import jax

from agents import td_agent
from projects.common.train import create_net_prediction_tuple

def load_ckptr(directory, learner):
  ckpt_config= distributed_layout.CheckpointingConfig(
    directory=directory,
    add_uid=False,
    max_to_keep=None)
  return savers.CheckpointingRunner(
          learner,
          key='learner',
          subdirectory='learner',
          # enable_checkpointing=False,
          **vars(ckpt_config)
          )
def load_agent(
    config,
    env_spec,
    NetworkCls,
    NetKwargs,
    LossFn,
    LossFnKwargs,
    directory,
    ckpt_path=None,
    behavior_policy_constructor=td_agent.make_behavior_policy,
    BuilderCls=td_agent.TDBuilder,
    agent_kwargs=None,
    wrap_actor=None,
    ):
  """Load agent along with parameters.
  
  Args:
      config (TYPE): Description
      env_spec (TYPE): Description
      NetworkCls (TYPE): Description
      NetKwargs (TYPE): Description
      LossFn (TYPE): Description
      LossFnKwargs (TYPE): Description
      directory (TYPE): Description
      behavior_policy_constructor (TYPE, optional): Description
      BuilderCls (TYPE, optional): Description
      agent_kwargs (None, optional): Description
      wrap_actor (None, optional): Useful for custom behavior on top of acting (e.g. saving data)
  
  Returns:
      TYPE: Description
  """
  agent_kwargs = agent_kwargs or dict()
  # -----------------------
  # prepare networks
  # -----------------------
  PredCls = create_net_prediction_tuple(config, env_spec, NetworkCls, NetKwargs)
  # insert into global namespace for pickling, etc.
  NetKwargs.update(PredCls=PredCls)

  # -----------------------
  # prepare networks
  # -----------------------
  networks=td_agent.make_networks(
        batch_size=config.batch_size,
        env_spec=env_spec,
        NetworkCls=NetworkCls,
        NetKwargs=NetKwargs,
        eval_network=True)

  # -----------------------
  # builder
  # -----------------------
  builder = td_agent.TDBuilder(
    networks=networks,
    config=config,
    LossFn=LossFn,
    LossFnKwargs=LossFnKwargs,
    )

  # -----------------------
  # learner
  # -----------------------
  key = jax.random.PRNGKey(config.seed)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=None,
        replay_client=None,
        counter=None)

  # -----------------------
  # load checkpoint
  # -----------------------
  checkpointer = load_ckptr(directory, learner)
  if ckpt_path:
    checkpointer._checkpointer._checkpoint.restore(ckpt_path)

  # -----------------------
  # actor
  # -----------------------
  policy_network = behavior_policy_constructor(networks, config)
  actor_key, key = jax.random.split(key)
  actor = builder.make_actor(
      actor_key, policy_network, variable_source=learner)

  if wrap_actor:
    actor = wrap_actor(actor=actor, networks=networks, learner=learner)

  # -----------------------
  # agent
  # -----------------------
  agent = acme_agent.Agent(actor=actor, learner=learner,
    min_observations=0,
    observations_per_step=1)

  return agent, checkpointer
