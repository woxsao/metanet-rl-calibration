from stable_baselines3.common.env_checker import check_env
from metanet_gym import METANETGymEnv
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.monitor import Monitor



def make_env(base_path):
    rho_hat = np.load(base_path + "/rho_hat.npy")
    q_hat = np.load(base_path + "/q_hat.npy")
    lane_mapping = np.load(base_path + "/lane_mapping.npy")[1:-1]
    off_ramp_mapping = np.load(base_path + "/off_ramp_mapping.npy")
    on_ramp_mapping = np.load(base_path + "/on_ramp_mapping.npy")
    upstream_flow = np.load(base_path + "/upstream_flow.npy")
    downstream_density = np.load(base_path + "/downstream_density.npy")
    v_hat = np.load(base_path + "/v_hat.npy")
     
    init_traffic_state = (rho_hat[0, :], v_hat[0, 1:-1], upstream_flow[0], 0)
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
    )
    return Monitor(metanet_env)

def main(base_path, total_timesteps=100000):
    env = DummyVecEnv([lambda: make_env(base_path)])
    print("Checking environment...")
    check_env(env.envs[0], warn=True)  # check_env needs unwrapped env

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log="./metanet_sb3_tensorboard/",
    )
    model.learn(total_timesteps=total_timesteps)
    
    # str representation of total_timesteps for filename
    model.save(f"metanet_sb3_ppo_model_{total_timesteps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps to train for")
    args = parser.parse_args()
    
    main(args.base_path, args.total_timesteps)
