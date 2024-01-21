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
            "logdir": "./logdir/run1",
            "run.train_ratio": 64,
            "run.log_every": 30,  # Seconds
            "batch_size": 8,
            "jax.prealloc": False,
            # "encoder.mlp_keys": "$^",
            # "decoder.mlp_keys": "$^",
            "encoder.mlp_keys": "robot0_eef_pos|robot0_eef_quat|robot0_gripper_qpos",
            "decoder.mlp_keys": "robot0_eef_pos|robot0_eef_quat|robot0_gripper_qpos",
            "encoder.cnn_keys": "agentview_image|robot0_eye_in_hand",
            "decoder.cnn_keys": "agentview_image|robot0_eye_in_hand",
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
        ],
    )

    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    from dreamerv3.embodied.envs import from_gym

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[64, 64],
        camera_widths=[64, 64],
        reward_shaping=True,
        # single_object_mode=1,
        reward_scale=1.0,
        # control_freq=20,
        horizon=500,
        hard_reset=False,
        ignore_done=False, # BenchmarkだとTrueだが、これしないとhorizon無視して延々と続く
    )
    
    env = GymWrapper(env, keys=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']) # obsサイズ指定
    obs = env.reset()
    print(obs)
    env = from_gym.FromGym(env, obs_key=['agentview_image', 'robot0_eye_in_hand_image','robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])  # obs名前指定
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # env_rec = env
    print(env.obs_space)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only_record(agent, env, env_rec, logger, args)


if __name__ == "__main__":
    main()
