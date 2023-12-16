def main():
    import warnings
    import dreamerv3
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["medium"])
    config = config.update(
        {
            "logdir": "~/logdir/run8",
            "run.train_ratio": 64,
            "run.log_every": 30,  # Seconds
            "batch_size": 16,
            "jax.prealloc": False,
            "encoder.mlp_keys": "$^",
            "decoder.mlp_keys": "$^",
            "encoder.cnn_keys": "frontview_image",
            "decoder.cnn_keys": "frontview_image",
            # 'jax.platform': 'cpu',
        }
    )
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            # embodied.logger.WandBOutput(logdir.name, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    from robosuite.utils.camera_utils import CameraMover

    from dreamerv3.embodied.envs import from_gym

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssemblyRound",
        robots="Sawyer",
        controller_configs=controller_config,
        has_renderer=False,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names="frontview",
        camera_heights=64,
        camera_widths=64,
        reward_shaping=True,
        # single_object_mode=1,
        reward_scale=1.0,
        # control_freq=20,
        horizon=500,
        hard_reset=False,
        ignore_done=False, # BenchmarkだとTrueだが、これしないとhorizon無視して延々と続く
    )
    
    # camera_mover = CameraMover(
    #     env=env,
    #     camera="frontview",
    # )
    # camera_mover.move_camera(direction=[300.0, 3.0, 3.0], scale=1.0)
    # _ = env.reset()
    env_rec = env.copy()
    env_rec.camera_heights = 256
    env_rec.camera_widths = 256
    
    env = GymWrapper(env, keys=['frontview_image']) # observation_space: Box(0, 255, (64, 64, 3), uint8)
    env = from_gym.FromGym(env, obs_key='frontview_image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    
    env_rec = GymWrapper(env_rec, keys=['frontview_image']) # observation_space: Box(0, 255, (64, 64, 3), uint8)
    env_rec = from_gym.FromGym(env_rec, obs_key='frontview_image')  # Or obs_key='vector'.
    env_rec = dreamerv3.wrap_env(env_rec, config)
    env_rec = embodied.BatchEnv([env_rec], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    # replay = embodied.replay.Uniform(
    #     config.batch_length, config.replay_size, logdir / "replay"
    # )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )
    # embodied.run.train(agent, env, replay, logger, args)
    embodied.run.eval_only_record(agent, env, env_rec, logger, args)


if __name__ == "__main__":
    main()
