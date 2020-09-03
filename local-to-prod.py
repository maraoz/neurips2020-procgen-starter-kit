import yaml


COLAB_CPUS = 2
WORKERS = 1
SCALE_DOWN = 1
ENVS = 12//SCALE_DOWN
TRAIN_BATCH = 16384//SCALE_DOWN
SGD_MINIBATCH = 2048//SCALE_DOWN

with open("experiments/impala-local.yaml") as fp:
    config = yaml.safe_load(fp)

    # steps
    config["procgen-ppo"]["stop"]["timesteps_total"] = 8000000

    # memory intensive stuff
    config["procgen-ppo"]["config"]["train_batch_size"] = TRAIN_BATCH
    config["procgen-ppo"]["config"]["sgd_minibatch_size"] = SGD_MINIBATCH

    # workers
    config["procgen-ppo"]["config"]["num_workers"] = WORKERS
    config["procgen-ppo"]["config"]["num_envs_per_worker"] = ENVS

    # gpu activation
    config["procgen-ppo"]["config"]["num_gpus"] = 0.2
    config["procgen-ppo"]["config"]["num_gpus_per_worker"] = 0.01

    with open("experiments/impala-baseline.yaml", "w") as fp:
        yaml.dump(config, fp)

with open("experiments/impala-local.yaml") as fp:
    config = yaml.safe_load(fp)

    # steps
    config["procgen-ppo"]["stop"]["timesteps_total"] = 100000
    config["procgen-ppo"]["stop"]["timesteps_total"] = 1000000
    config["procgen-ppo"]["checkpoint_freq"] = 10

    # memory intensive stuff
    config["procgen-ppo"]["config"]["train_batch_size"] = TRAIN_BATCH
    config["procgen-ppo"]["config"]["sgd_minibatch_size"] = SGD_MINIBATCH

    # workers
    config["procgen-ppo"]["config"]["num_workers"] = WORKERS
    config["procgen-ppo"]["config"]["num_envs_per_worker"] = ENVS
    config["procgen-ppo"]["config"]["num_cpus_per_worker"] = (COLAB_CPUS-1)/(WORKERS)

    # gpu activation
    config["procgen-ppo"]["config"]["num_gpus"] = 0.1
    config["procgen-ppo"]["config"]["num_gpus_per_worker"] = 0.01

    with open("experiments/impala-github.yaml", "w") as fp:
        yaml.dump(config, fp)
