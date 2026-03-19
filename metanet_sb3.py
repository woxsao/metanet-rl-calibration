from stable_baselines3.common.env_checker import check_env
from metanet_gym import METANETGymEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.monitor import Monitor
import os

training_metadata = {}


def make_env(base_path, param_update_interval=1, bc_noise_std=0.02, bc_smoothness=0.97, custom_bounds=None, perturb_bc=True):
    rho_hat = np.load(base_path + "/rho_hat.npy")
    q_hat = np.load(base_path + "/q_hat.npy")
    lane_mapping = np.load(base_path + "/lane_mapping.npy")[1:-1]
    off_ramp_mapping = np.load(base_path + "/off_ramp_mapping.npy")
    on_ramp_mapping = np.load(base_path + "/on_ramp_mapping.npy")

    # Optional boundary files
    upstream_flow_path = base_path + "/upstream_flow.npy"
    downstream_density_path = base_path + "/downstream_density.npy"

    if os.path.exists(upstream_flow_path) and os.path.exists(downstream_density_path):
        upstream_flow = np.load(upstream_flow_path)
        downstream_density = np.load(downstream_density_path)
    else:
        # Treat rho_hat and q_hat as including boundaries
        upstream_flow = q_hat[:, 0].copy()
        downstream_density = rho_hat[:, -1].copy()

        # Keep only interior cells
        rho_hat = rho_hat[:, 1:-1]
        q_hat = q_hat[:, 1:-1]

    # Optional velocity file
    v_hat_path = base_path + "/v_hat.npy"
    if os.path.exists(v_hat_path):
        v_hat = np.load(v_hat_path)[
            :, 1:-1
        ]  # Assume v_hat includes boundaries and remove them
    else:
        # Compute velocity safely: v = q / rho
        with np.errstate(divide="ignore", invalid="ignore"):
            v_hat = np.divide(q_hat, rho_hat, where=rho_hat != 0)
            v_hat[rho_hat == 0] = 0.0

    init_traffic_state = (rho_hat[0, :], v_hat[0, :], upstream_flow[0], 0)
    param_update_interval = param_update_interval
    training_metadata["bc_noise_std"] = bc_noise_std
    training_metadata["bc_smoothness"] = bc_smoothness
    training_metadata["param_update_interval"] = param_update_interval
    metanet_env = METANETGymEnv(
        T=10 / 3600,
        l=0.4,
        rho_hat=rho_hat,
        q_hat=q_hat,
        lane_mapping=lane_mapping,
        off_ramp_mapping=off_ramp_mapping,
        on_ramp_mapping=on_ramp_mapping,
        upstream_flow=upstream_flow,
        downstream_density=downstream_density,
        init_traffic_state=init_traffic_state,
        bc_noise_std=bc_noise_std,
        bc_smoothness=bc_smoothness,
        param_update_interval=param_update_interval,
        custom_param_ranges=custom_bounds,
        perturb_bc=perturb_bc,
    )
    return Monitor(metanet_env)


def floatify_bounds(bounds_dict):
    for k, v in bounds_dict.items():
        if isinstance(v, list):
            bounds_dict[k] = [float(eval(str(x))) for x in v]
    return bounds_dict


def main(
    base_path,
    total_timesteps=100000,
    update_interval=1,
    base_model_dir=None,
    save_dir=None,
    num_cpus = 8,
    tensorboard_log="./metanet_sb3_tensorboard/",
    bc_smoothness=0.97,
    bc_noise_std=0.02,
    perturb_bc=True,
    n_steps=1024,
    batch_size=64
):
    custom_bounds = None
    if save_dir is not None and os.path.exists(save_dir + "/bounds.json"):
        import json

        with open(save_dir + "/bounds.json", "r") as f:
            custom_bounds = json.load(f)
        print("Using custom parameter bounds from bounds.json:")
        custom_bounds = floatify_bounds(custom_bounds)
        print(custom_bounds)

    # n_steps = 1024  # keep total rollout size fixed
    def make_env_fn(bp, ui, cb, bc_noise_std=0.02, bc_smoothness=0.97):
        def _init():
            return make_env(bp, param_update_interval=ui, custom_bounds=cb, bc_noise_std=bc_noise_std, bc_smoothness=bc_smoothness, perturb_bc=perturb_bc)
        return _init

    env = SubprocVecEnv([
        make_env_fn(base_path, update_interval, custom_bounds, bc_noise_std, bc_smoothness)
        for _ in range(num_cpus)
    ])

    save_path = (
        save_dir if save_dir is not None else f"metanet_sb3_ppo_model_{total_timesteps}"
    )
    os.makedirs(save_path, exist_ok=True)

    # print("Checking environment...")
    # if hasattr(env, 'envs'):
    #     check_env(env.envs[0], warn=True)  # check_env needs unwrapped env
    lr = 3e-4
    # n_steps = 2048
    # batch_size = 64
    n_epochs = 10
    gamma = 0.999
    gae_lambda = 0.95
    clip_range = 0.2

    network_kwargs = "default"

    if base_model_dir is not None:
        print(f"Loading model from {base_model_dir}...")
        model = PPO.load(
            base_model_dir, env=env, tensorboard_log=tensorboard_log
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            # policy_kwargs=dict(
            #     net_arch=dict(pi=[256, 256], vf=[256, 256])  # CORRECT: just the dict
            # ),
            tensorboard_log=tensorboard_log,
        )

    training_metadata["learning_rate"] = lr
    training_metadata["n_steps"] = n_steps
    training_metadata["batch_size"] = batch_size
    training_metadata["n_epochs"] = n_epochs
    training_metadata["gamma"] = gamma
    training_metadata["gae_lambda"] = gae_lambda
    training_metadata["clip_range"] = clip_range
    training_metadata["network_kwargs"] = network_kwargs

    model.learn(total_timesteps=total_timesteps)

    # str representation of total_timesteps for filename
    if save_dir is not None:
        model_save_path = f"{save_dir}/model"
        print(f"Saving model to {model_save_path}...")
        os.makedirs(save_dir, exist_ok=True)
        model.save(model_save_path)
        np.savez(f"{save_dir}/training_metadata", **training_metadata)
    else:
        model.save(f"metanet_sb3_ppo_model_{total_timesteps}/model.zip")
        np.savez(
            f"metanet_sb3_ppo_model_{total_timesteps}/training_metadata",
            **training_metadata,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train for",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=1,
        help="How often to update parameters in the environment",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=None,
        help="Directory of the trained model to evaluate",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save results (learned parameters, simulation outputs)",
    )
    parser.add_argument("--num_cpus", type=int, default=os.cpu_count())
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default="./metanet_sb3_tensorboard/",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--bc_smoothness",
        type=float,
        default=0.97,
        help="Smoothness factor for boundary condition perturbations (0-1, higher is smoother)",
    )
    parser.add_argument(
        "--bc_noise_std",
        type=float,
        default=0.02,
        help="Standard deviation of noise added to boundary conditions at each timestep",
    )
    parser.add_argument(
        "--perturb_bc",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether to perturb boundary conditions (True/False)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1024,
        help="Number of steps to run in each environment per update (total rollout size will be"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for PPO updates",
    )
    print(f"Using {parser.parse_args().num_cpus} CPUs for training.")
    args = parser.parse_args()

    main(
        args.base_path,
        args.total_timesteps,
        args.update_interval,
        args.base_model_dir,
        args.save_dir,
        args.num_cpus,
        args.tensorboard_log,
        args.bc_smoothness,
        args.bc_noise_std,
        args.perturb_bc,
        args.n_steps,
        args.batch_size
    )
