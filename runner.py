from stable_baselines3 import PPO
from metanet_gym import METANETGymEnv
from metanet_calibration.metanet_dynamics import run_metanet_sim
import numpy as np
from metanet_calibration.data_processing import plot_simulation_comparison  # adjust import

# 1. Load your data
base_path = "../macroscopic-sim-calibration/eval_set/i24_westbound"
rho_hat = np.load(base_path + "/rho_hat.npy")
q_hat = np.load(base_path + "/q_hat.npy")
v_hat = np.load(base_path + "/v_hat.npy")
lane_mapping = np.load(base_path + "/lane_mapping.npy")
upstream_flow = np.load(base_path + "/upstream_flow.npy")
downstream_density = np.load(base_path + "/downstream_density.npy")
off_ramp_mapping = np.load(base_path + "/off_ramp_mapping.npy")
on_ramp_mapping = np.load(base_path + "/on_ramp_mapping.npy")

init_traffic_state = (rho_hat[0, :], v_hat[0, 1:-1], upstream_flow[0], 0)
print(lane_mapping.shape)
lane_mapping = lane_mapping[1:-1]
# 2. Create env and load model
env = METANETGymEnv(
    T=10/3600, l=0.4,
    rho_hat=rho_hat, q_hat=q_hat,
    lane_mapping=lane_mapping,
    off_ramp_mapping=off_ramp_mapping,
    on_ramp_mapping=on_ramp_mapping,
    upstream_flow=upstream_flow,
    downstream_density=downstream_density,
    init_traffic_state=init_traffic_state,
)

model = PPO.load("metanet_sb3_ppo_model_10000000")

# 3. Extract learned parameters
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
learned_params = env._scale_action(action)

# 4. Run METANET simulation with learned parameters
lanes_dict = {i: int(lane_mapping[i]) for i in range(len(lane_mapping))}

density_sim, velocity_sim, queue, total_travel_time = run_metanet_sim(
    T=10/3600,
    l=0.4,
    init_traffic_state=init_traffic_state,
    demand=upstream_flow,
    downstream_density=downstream_density,
    params=learned_params,
    lanes=lanes_dict,
    plotting=True,
)

# 5. Compute ground truth velocity (if needed)
v_true = q_hat / (rho_hat + 1e-6)

# 6. Plot comparison using your function
# Note: density_sim is (num_timesteps+1, num_segments), so trim the last row
plot_simulation_comparison(
    rho_sim=density_sim[:-1],      # Remove last timestep to match ground truth
    v_sim=velocity_sim[:-1],       # Remove last timestep
    rho_true=rho_hat,              # Ground truth density
    v_true=v_true,                 # Ground truth velocity
    q_true=q_hat,                  # Ground truth flow
    include_fd=True,
    save_path="metanet_calibration_comparison.png",
    lanes=lane_mapping             # Pass lane array for FD plot
)

# 7. Print error metrics
mape_rho = np.mean(np.abs(density_sim[:-1] - rho_hat) / (rho_hat + 1e-6))
mape_v = np.mean(np.abs(velocity_sim[:-1] - v_true) / (v_true + 1e-6))
mape_q = np.mean(np.abs(density_sim[:-1] * velocity_sim[:-1] - q_hat) / (q_hat + 1e-6))

print(f"\nCalibration Results:")
print(f"  MAPE Density:  {mape_rho:.4f}")
print(f"  MAPE Velocity: {mape_v:.4f}")
print(f"  MAPE Flow:     {mape_q:.4f}")
print(f"  Total Travel Time: {total_travel_time:.2f} hours")