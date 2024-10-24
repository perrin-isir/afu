import argparse
import numpy as np


def run(args):
    # lazy imports:
    import jax
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
        platform = jax.lib.xla_bridge.get_backend().platform
        print("platform:", platform)
    else:
        platform = jax.lib.xla_bridge.get_backend().platform
        print("platform:", platform)
        assert platform == "gpu", \
            "JAX is not running on GPU."
    import os
    import sys
    from datetime import datetime
    from afu_rljax.algorithm import AFU
    from afu_rljax.env import make_continuous_env
    from afu_rljax.trainer import Trainer

    env = make_continuous_env(args.env_id)
    env_test = make_continuous_env(args.env_id)

    algo = AFU(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        tau=1e-2,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        gradient_reduction=1. - args.rho,
        alg=args.alg,
        ablation=args.ablation,
        variant=args.variant,
        hyperparam=args.hyperparam,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(args.save_dir,
                           "logs", args.env_id, f"seed{args.seed}-{time}")
    print("Logging in", os.path.expanduser(log_dir))
    os.makedirs(os.path.expanduser(log_dir))

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed + 1,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--env_id", type=str, default="Ant-v4")
    p.add_argument("--num_agent_steps", type=int, default=3 * 10 ** 6)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=np.random.randint(1e9))
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--alg", type=str, default="AFU")
    p.add_argument("--ablation", type=bool, default=False)
    p.add_argument("--variant", type=str, default="alpha")
    p.add_argument("--hyperparam", type=float, default=1.)
    arguments = p.parse_args()
    run(arguments)
