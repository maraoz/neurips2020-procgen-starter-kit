import yaml

with open("experiments/impala-local.yaml") as fp:
    impala_config = yaml.safe_load(fp)

    # steps
    impala_config["procgen-ppo"]["stop"]["timesteps_total"] = 8000000

    # memory intensive stuff
    impala_config["procgen-ppo"]["config"]["train_batch_size"] = 16384
    impala_config["procgen-ppo"]["config"]["sgd_minibatch_size"] = 2048

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
    impala_config["procgen-ppo"]["stop"]["timesteps_total"] = 100000
    impala_config["procgen-ppo"]["checkpoint_freq"] = 10

    # memory intensive stuff
    impala_config["procgen-ppo"]["config"]["train_batch_size"] = 16384
    impala_config["procgen-ppo"]["config"]["sgd_minibatch_size"] = 2048

    # workers
    COLAB_CPUS = 2
    WORKERS = 6
    impala_config["procgen-ppo"]["config"]["num_workers"] = WORKERS
    impala_config["procgen-ppo"]["config"]["num_envs_per_worker"] = 12
    impala_config["procgen-ppo"]["config"]["num_cpus_per_worker"] = (COLAB_CPUS-1)/(WORKERS)

    # gpu activation
    impala_config["procgen-ppo"]["config"]["num_gpus"] = 0.1
    impala_config["procgen-ppo"]["config"]["num_gpus_per_worker"] = 0.01

    with open("experiments/impala-github.yaml", "w") as fp:
        yaml.dump(impala_config, fp)
