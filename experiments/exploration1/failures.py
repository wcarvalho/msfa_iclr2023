def r2d1_vae(config, env_spec):
  num_actions = env_spec.actions.num_values

  prediction = DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  vae = vae_modules.VAE(
    latent_dim=config.latent_dim,
    latent_source=config.latent_source,
    **vae_modules.small_standard_encoder_decoder(),
    )
  aux_tasks = vae.aux_task

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vae,
    memory_prep_fn=memory_prep_fn(
      num_actions=num_actions,
      extract_fn=lambda inputs, obs: obs.samples),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=prediction,
    aux_tasks=aux_tasks,
  )

def usfa_reward(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]
  prediction = UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      )

  aux_tasks = [
    ValueAuxTask([config.out_hidden_size, 1]),
    RewardAuxTask([config.out_hidden_size, state_dim])
  ]


  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=prediction,
    aux_tasks=aux_tasks,
  )

def usfa_reward_vae(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]
  prediction = UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      )

  vae = VAE(latent_dim=config.latent_dim)

  raise RuntimeError("account for when cumulants == raw state output")
  aux_tasks = [
    vae.aux_task,
    CumulantsAuxTask([state_dim])
  ]

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=vae,
    memory_prep_fn=memory_prep_fn(
      num_actions=num_actions,
      extract_fn=lambda inputs, obs: obs.samples),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=usfa_prep_fn,
    prediction=prediction,
    aux_tasks=aux_tasks,
  )

def r2d1_dummy_symbolic(config, env_spec):
  num_actions = env_spec.actions.num_values

  class DummySymbolicTorso(hk.Module):
    """Flatten."""

    def __call__(self, inputs) -> jnp.ndarray:
      return jnp.reshape(inputs, [inputs.shape[0], -1])  # [B, D]

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=DummySymbolicTorso(),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=r2d1_prediction_prep_fn,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size])
  )

def r2d1_noise_ensemble(config, env_spec):
  num_actions = env_spec.actions.num_values

  def prediction_prep_fn(inputs, memory_out, **kwargs):
    """
    Concat [task + noise] with memory output.
    """
    return QEnsembleInputs(
      w=inputs.observation.task,
      memory_out=memory_out,
      )

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=True),
    memory_prep_fn=OAREmbedding(num_actions=num_actions),
    memory=hk.LSTM(config.memory_size),
    prediction_prep_fn=prediction_prep_fn,
    prediction=QEnsembleHead(
      num_actions=num_actions,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      policy_layers=config.policy_layers,
      q_input_fn=ConcatFlatStatePolicy(config.state_hidden_size)
      )
  )


def r2d1_farm_model(config, env_spec):
  num_actions = env_spec.actions.num_values


  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=build_farm(config),
    prediction_prep_fn=flatten_structured_memory,
    prediction=DuellingMLP(num_actions, hidden_sizes=[config.out_hidden_size]),
    aux_tasks=FarmModel(
      config.model_layers*[config.module_size],
      num_actions=num_actions,
      activation=getattr(jax.nn, config.activation)),
  )

def usfa_farm_model(config, env_spec):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  if config.embed_task:
    cumulant_out_dim='identity'
    import ipdb; ipdb.set_trace()
  else:
    cumulant_out_dim='concat'
    import ipdb; ipdb.set_trace()

  aux_tasks = [
    # takes structured farm input
    FarmModel(
      config.model_layers*[config.module_size],
      num_actions=num_actions,
      activation=getattr(jax.nn, config.activation)),
    # takes structured farm input
    FarmCumulants(
      out_dim=cumulant_out_dim,
      hidden_size=config.cumulant_hidden_size,
      cumtype=config.cumtype),
  ]

  if config.mixture == "unique":
    assert config.cumtype == 'sum'
    sf_input_fn=usfa_modules.UniqueStatePolicyPairs()
  else:
    raise NotImplementedError

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm.FarmSharedOutput(
      module_size=config.module_size,
      nmodules=config.nmodules,
      out_layers=config.out_layers),
    prediction_prep_fn=usfa_prep_fn,
    prediction=UsfaHead(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      sf_input_fn=sf_input_fn,
      ),
    aux_tasks=aux_tasks,
  )

def usfa_farmflat(config, env_spec, use_seperate_eval=True):
  """
  Vision Net --> Farm Memory --> USFA Predictions
  """
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  farm_memory = build_farm(config)

  usfa_head = build_usfa_farm_head(
    config=config, state_dim=state_dim, num_actions=num_actions,
    farm_memory=farm_memory, flat=True)

  def prediction_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  def evaluation_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_eval_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  if use_seperate_eval:
    evaluation=usfa_head.evaluation
  else:
    evaluation_prep_fn=None
    evaluation=None

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=evaluation,
  )

def load_agent_settings(agent, env_spec, config_kwargs=None):
  if agent is None:
    raise RuntimeError

  elif agent == "r2d1_noise":
    config = configs.USFAConfig(**default_config)  # for convenience since has var

    NetworkCls=nets.r2d1_noise # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_dummy_symbolic":
    config = configs.R2D1Config(**default_config)

    NetworkCls=nets.r2d1_dummy_symbolic # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_noise_ensemble":
    config = configs.USFAConfig(**default_config)   # for convenience since has var
    config.loss_coeff = 0 # Turn off main loss

    NetworkCls=nets.r2d1_noise_ensemble # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[
        QLearningEnsembleLoss(
          coeff=1.,
          discount=config.discount)
      ])
    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_vae":
    # R2D1 + VAE
    config = data_utils.merge_configs(
      dataclass_configs=[configs.R2D1Config(), configs.VAEConfig()],
      dict_configs=default_config
      )

    NetworkCls =  nets.r2d1_vae # default: 2M params
    NetKwargs=dict(config=config, env_spec=env_spec)
    LossFn = td_agent.R2D2Learning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[VaeAuxLoss(
                  coeff=config.vae_coeff,
                  beta=config.beta)])

    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "r2d1_farm_model":

    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.R2D1Config(), configs.FarmConfig(), configs.FarmModelConfig()],
      dict_configs=default_config
      )

    NetworkCls=nets.r2d1_farm_model # default: 1.5M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.R2D2Learning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature,
                  )])

    loss_label = 'r2d1'
    eval_network = config.eval_network

  elif agent == "usfa_qlearning":
  # USFA Arch. with Q-learning and no SF loss
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.USFAConfig(),
        configs.RewardConfig()],
      dict_configs=default_config
      )
    config.loss_coeff = 0 # Turn off USFA Learning

    NetworkCls =  functools.partial(nets.usfa, use_seperate_eval=False)
    NetKwargs=dict(config=config,env_spec=env_spec)

    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      aux_tasks=[
        q_aux_loss(config.q_aux)(
          coeff=1.,
          discount=config.discount)
      ])
    loss_label = 'usfa'
    eval_network = False

  elif agent == "usfa_farmflat_qlearning":
  # USFA Arch. + FARM + Q-learning + __no__ SF loss
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.USFAConfig(),
        configs.FarmConfig(),
        configs.RewardConfig()],
      dict_configs=default_config
      )
    config.loss_coeff = 0 # Turn off USFA Learning

    NetworkCls =  functools.partial(nets.usfa_farmflat, use_seperate_eval=False)
    NetKwargs=dict(config=config, env_spec=env_spec)

    LossFn = td_agent.USFALearning
    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.dummy_cumulants_from_env,
      aux_tasks=[
        # Q-learning loss
        q_aux_loss(config.q_aux)(
          coeff=1.,
          discount=config.discount)
      ])
    loss_label = 'usfa'
    eval_network = False

  elif agent == "usfa_farm_model":
    # USFA which learns cumulants with structured transition model
    config = data_utils.merge_configs(
      dataclass_configs=[
        configs.ModularUSFAConfig(), configs.FarmConfig(), configs.FarmModelConfig(), configs.RewardConfig()],
      dict_configs=default_config
      )

    NetworkCls =  nets.usfa_farm_model
    NetKwargs=dict(config=config,env_spec=env_spec)
    
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.cumulants_from_preds,
      shorten_data_for_cumulant=True, # needed since using delta for cumulant
      aux_tasks=[
        cumulants.CumulantRewardLoss(
          shorten_data_for_cumulant=True,
          coeff=config.reward_coeff,  # coefficient for loss
          loss=config.reward_loss),  # type of loss for reward
        DeltaContrastLoss(
                    coeff=config.model_coeff,
                    extra_negatives=config.extra_negatives,
                    temperature=config.temperature),
      ])
    loss_label = 'usfa'
    eval_network = config.eval_network

  elif agent == "usfa_reward":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardConfig(**default_config)

    NetworkCls =  nets.usfa_reward # default: 2.1M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulant=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=functools.partial(aux_tasks.cumulant_from_reward,
        coeff=config.reward_coeff,  # coefficient for loss
        loss=config.reward_loss))   # type of loss for reward

    loss_label = 'usfa'

  elif agent == "usfa_reward_value":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardValueConfig(**default_config)

    NetworkCls =  nets.usfa_reward # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulant=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=[
        ValueAuxLoss(coeff=config.value_coeff, discount=config.discount),
        functools.partial(aux_tasks.cumulant_from_reward,
              coeff=config.reward_coeff,  # coefficient for loss
              loss=config.reward_loss)
              ])   # type of loss for reward

    loss_label = 'usfa'

  elif agent == "usfa_reward_vae":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFARewardVAEConfig(**default_config)

    NetworkCls =  nets.usfa_reward_vae # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulants=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=[
        VaeAuxLoss(coeff=config.vae_coeff),
        cumulants.CumulantRewardLoss(
          coeff=config.reward_coeff,  # coefficient for loss
          loss=config.reward_loss),  # type of loss for reward
      ])   # type of loss for reward

    loss_label = 'usfa'
    eval_network = config.eval_network

  elif agent == "usfa_farmflat_reward_value":
    # Universal Successor Features which learns cumulants by predicting reward
    config = configs.USFAFarmRewardValueConfig(**default_config)

    NetworkCls =  nets.usfa_farmflat_reward # default: 2M params
    NetKwargs=dict(config=config,env_spec=env_spec)
    LossFn = td_agent.USFALearning

    LossFnKwargs = td_agent.r2d2_loss_kwargs(config)
    LossFnKwargs.update(
      extract_cumulant=losses.cumulants_from_preds,
      # auxilliary task as argument
      aux_tasks=[
        ValueAuxLoss(coeff=config.value_coeff, discount=config.discount),
        functools.partial(aux_tasks.cumulant_from_reward,
              coeff=config.reward_coeff,  # coefficient for loss
              loss=config.reward_loss)
              ])   # type of loss for reward

    loss_label = 'usfa'
  else:
    raise RuntimeError

def usfa_farmflat_model(config, env_spec, predict_cumulants=True, learn_model=True, **net_kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  farm_memory = build_farm(config)

  usfa_head = build_usfa_farm_head(
    config=config,
    state_dim=state_dim,
    num_actions=num_actions,
    farm_memory=farm_memory,
    sf_input_fn=ConcatFlatStatePolicy(config.state_hidden_size),
    flat=True)

  aux_tasks = []
  if learn_model:
    # takes structured farm input
    aux_tasks.append(
      FarmModel(
        output_sizes=max(config.model_layers-1, 0)*[config.module_size],
        num_actions=num_actions,
        seperate_params=config.seperate_model_params,
        # activation=getattr(jax.nn, config.activation)
        ),
      )

  if predict_cumulants:
    # takes structured farm input
    if config.seperate_cumulant_params:
      cumulants_per_module = state_dim//farm_memory.nmodules
      aux_tasks.append(
        FarmIndependentCumulants(
          activation=config.cumulant_act,
          module_cumulants=cumulants_per_module,
          hidden_size=config.cumulant_hidden_size,
          layers=config.cumulant_layers,
          seperate_params=config.seperate_cumulant_params,
          construction=config.cumulant_const,
          normalize_delta=config.normalize_delta,
          normalize_state=getattr(config, "contrast_time_coeff", 0) > 0,
          normalize_cumulants=config.normalize_cumulants
          ))
    else:
      aux_tasks.append(
        FarmCumulants(
              activation=config.cumulant_act,
          module_cumulants=usfa_head.out_dim,
          hidden_size=config.cumulant_hidden_size,
          layers=config.cumulant_layers,
          aggregation='concat',
          normalize_cumulants=config.normalize_cumulants,
          normalize_delta=config.normalize_delta,
          construction=config.cumulant_const,
          ))

  def prediction_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  def evaluation_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_eval_prep_fn(inputs=inputs, 
      memory_out=flatten_structured_memory(memory_out))

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=usfa_head.evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )


def usfa_farm_model(config, env_spec, predict_cumulants=True, learn_model=True, **net_kwargs):
  num_actions = env_spec.actions.num_values
  state_dim = env_spec.observations.observation.state_features.shape[0]

  farm_memory = build_farm(config)

  cumulants_per_module = state_dim//farm_memory.nmodules
  usfa_head = FarmUsfaHead(
      num_actions=num_actions,
      cumulants_per_module=cumulants_per_module,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      policy_layers=config.policy_layers,
      multihead=config.seperate_value_params, # seperate params per cumulants
      vmap_multihead=config.farm_vmap,
      )

  assert state_dim == usfa_head.cumulants_per_module*farm_memory.nmodules

  aux_tasks = []
  if learn_model:
    # takes structured farm input
    aux_tasks.append(
      FarmModel(
        output_sizes=max(config.model_layers-1, 0)*[config.module_size],
        num_actions=num_actions,
        seperate_params=config.seperate_model_params,
        # activation=getattr(jax.nn, config.activation)
        ),
      )
  if predict_cumulants:
    # takes structured farm input
    aux_tasks.append(
      FarmIndependentCumulants(
        activation=config.cumulant_act,
        module_cumulants=cumulants_per_module,
        hidden_size=config.cumulant_hidden_size,
        layers=config.cumulant_layers,
        seperate_params=config.seperate_cumulant_params,
        construction=config.cumulant_const,
        normalize_delta=config.normalize_delta and getattr(config, "contrast_module_coeff", 0) > 0,
        normalize_state=getattr(config, "contrast_time_coeff", 0) > 0,
        normalize_cumulants=config.normalize_cumulants)
    )

  def prediction_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_prep_fn(inputs=inputs, memory_out=memory_out)

  def evaluation_prep_fn(inputs, memory_out, *args, **kwargs):
    """Concat Farm module-states before passing them."""
    return usfa_eval_prep_fn(inputs=inputs, memory_out=memory_out)

  return BasicRecurrent(
    inputs_prep_fn=convert_floats,
    vision_prep_fn=get_image_from_inputs,
    vision=AtariVisionTorso(flatten=False),
    memory_prep_fn=make_farm_prep_fn(num_actions,
      task_input=config.farm_task_input),
    memory=farm_memory,
    prediction_prep_fn=prediction_prep_fn,
    prediction=usfa_head,
    evaluation_prep_fn=evaluation_prep_fn,
    evaluation=usfa_head.evaluation,
    aux_tasks=aux_tasks,
    **net_kwargs
  )

def build_usfa_farm_head(config, state_dim, num_actions, farm_memory, sf_input_fn=None, flat=True, Cls=UsfaHead):

  if config.embed_task:
    # if embedding task, don't project delta for cumulant
    # embed task to size of delta
    if flat:
      task_embed = farm_memory.total_dim
    else:
      task_embed = farm_memory.module_size
  else:
    # if not embedding task, project delta for cumulant
    task_embed = 0

  usfa_head = Cls(
      num_actions=num_actions,
      state_dim=state_dim,
      hidden_size=config.out_hidden_size,
      policy_size=config.policy_size,
      variance=config.variance,
      nsamples=config.npolicies,
      duelling=config.duelling,
      policy_layers=config.policy_layers,
      z_as_train_task=config.z_as_train_task,
      sf_input_fn=sf_input_fn,
      multihead=config.multihead,
      concat_w=config.concat_w,
      task_embed=task_embed,
      normalize_task=config.normalize_task and config.embed_task,
      )
  return usfa_head


  elif agent == "msf_delta_model":
  # USFA + cumulants from FARM + Q-learning
    default_config['contrast_module_pred'] = 'delta'
    return usfa_farm(default_config, env_spec,
      net='exploration1',
      predict_cumulants=True,
      learn_model=True)
  elif agent == "msf_time_model":
  # USFA + cumulants from FARM + Q-learning
    return usfa_farm(default_config, env_spec,
      net='exploration1',
      predict_cumulants=True,
      learn_model=True)
  elif agent == "msf_state_model":
  # USFA + cumulants from FARM + Q-learning
    default_config['contrast_module_pred'] = 'state'
    return usfa_farm(default_config, env_spec,
      net='exploration1',
      predict_cumulants=True,
      learn_model=True)
  else:
    raise NotImplementedError(agent)