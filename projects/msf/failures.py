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

def load_agent_settings(agent, env_spec, config_kwargs=None):
  if agent is None:
    raise RuntimeError

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
