import yaml

with open("experiments/impala-local.yaml") as fp:
    impala_config = yaml.safe_load(fp)

    # steps
    impala_config["procgen-ppo"]["stop"]["timesteps_total"] = 8000000

    # workers
    impala_config["procgen-ppo"]["config"]["num_workers"] = 6
    impala_config["procgen-ppo"]["config"]["num_envs_per_worker"] = 12

    # gpu activation
    impala_config["procgen-ppo"]["config"]["num_gpus"] = 0.2
    impala_config["procgen-ppo"]["config"]["num_gpus_per_worker"] = 0.01

    with open("experiments/impala-baseline.yaml", "w") as fp:
        yaml.dump(impala_config, fp)

with open("experiments/impala-local.yaml") as fp:
    impala_config = yaml.safe_load(fp)

    # steps
    impala_config["procgen-ppo"]["stop"]["timesteps_total"] = 500000
    #impala_config["procgen-ppo"]["stop"]["timesteps_total"] = 8000000
    impala_config["procgen-ppo"]["checkpoint_freq"] = 10

    # workers
    impala_config["procgen-ppo"]["config"]["num_workers"] = 1
    impala_config["procgen-ppo"]["config"]["num_envs_per_worker"] = 6

    # gpu activation
    impala_config["procgen-ppo"]["config"]["num_gpus"] = 0.9
    impala_config["procgen-ppo"]["config"]["num_gpus_per_worker"] = 0.1

    with open("experiments/impala-github.yaml", "w") as fp:
        yaml.dump(impala_config, fp)
