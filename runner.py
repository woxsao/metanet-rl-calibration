from stable_baselines3 import PPO
from metanet_gym import METANETGymEnv
from metanet_calibration.metanet_dynamics import run_metanet_sim
import numpy as np
from metanet_calibration.data_processing import plot_simulation_comparison
import os

# 1. Load your data
base_path = "../macroscopic-sim-calibration/eval_set/I5_S_pm_115_to_9_8am_to_9am"
rho_hat = np.load(base_path + "/rho_hat.npy")
rho_hat = np.where(rho_hat == 0.0, 1e-3, rho_hat)  # avoid div by 0
q_hat = np.load(base_path + "/q_hat.npy")
q_hat = np.where(q_hat == 0.0, 1e-3, q_hat)
v_hat = q_hat / rho_hat
lane_mapping = np.load(base_path + "/lane_mapping.npy")
off_ramp_mapping = np.load(base_path + "/off_ramp_mapping.npy")
on_ramp_mapping = np.load(base_path + "/on_ramp_mapping.npy")

upstream_flow_path = base_path + "/upstream_flow.npy"
downstream_density_path = base_path + "/downstream_density.npy"
if os.path.exists(upstream_flow_path) and os.path.exists(downstream_density_path):
    upstream_flow = np.load(upstream_flow_path)
    downstream_density = np.load(downstream_density_path)
else:
    upstream_flow = q_hat[:, 0].copy()
    downstream_density = rho_hat[:, -1].copy()
    rho_hat = rho_hat[:, 1:-1]
    q_hat = q_hat[:, 1:-1]

init_traffic_state = (rho_hat[0, :], v_hat[0, 1:-1], upstream_flow[0], 0)
print(lane_mapping.shape)
lane_mapping = lane_mapping[1:-1]

# 2. Create env and load model
env = METANETGymEnv(
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
    param_update_interval=1,
)

model_dir = "pems_i5_rl_results_posttrain"
model_name = f"{model_dir}/model"
model = PPO.load(model_name)

# 3. Run episode and collect all actions
obs, _ = env.reset()
done = False
all_actions = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    all_actions.append(action)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# 4. Convert to time-varying params (shape: num_timesteps, num_segments)
params_time_varying = {
    key: [] for key in ["tau", "K", "eta_high", "rho_crit", "v_free", "a", "beta", "r"]
}
current_params = None
for t, action in enumerate(all_actions):
    if t % env.param_update_interval == 0:
        current_params = env._scale_action(action)
    for key in params_time_varying:
        params_time_varying[key].append(current_params[key])

# Stack into 2D arrays
for key in params_time_varying:
    params_time_varying[key] = np.array(
        params_time_varying[key]
    )  # (num_timesteps, num_segments)
    print(f"\nLearned parameter '{key}':")
    print(np.array(params_time_varying[key][0, :]))
    print(np.array(params_time_varying[key])[10, :])

print(f"\nLearned parameters (mean across time and space):")
for key, val in params_time_varying.items():
    print(f"  {key:12s}: mean={val.mean():.6f}, std={val.std():.6f}")

# 5. Run METANET simulation with time-varying learned parameters
lanes_dict = {i: int(lane_mapping[i]) for i in range(len(lane_mapping))}

density_sim, velocity_sim, queue, total_travel_time = run_metanet_sim(
    T=10 / 3600,
    l=0.4,
    init_traffic_state=init_traffic_state,
    demand=upstream_flow,
    downstream_density=downstream_density,
    params=params_time_varying,
    lanes=lanes_dict,
    plotting=True,
    real_data=True,
)

# 6. Compute derived quantities
v_true = q_hat / (rho_hat + 1e-6)
q_sim = density_sim[:-1] * velocity_sim[:-1]

# 7. Save all outputs as individual .npy files
os.makedirs(model_dir, exist_ok=True)

# Learned parameters — map internal keys to output filenames
param_filename_map = {
    "tau": "tau",
    "K": "K",
    "eta_high": "eta_high",
    "rho_crit": "rho_crit",
    "v_free": "v_free",
    "a": "a",
    "beta": "beta_array",
    "r": "r_inflow_array",
}
for key, fname in param_filename_map.items():
    np.save(os.path.join(model_dir, f"{fname}.npy"), params_time_varying[key])

# Simulation predictions
np.save(os.path.join(model_dir, "rho_pred.npy"), density_sim[:-1])
np.save(os.path.join(model_dir, "v_pred.npy"), velocity_sim[:-1])
np.save(os.path.join(model_dir, "q_pred.npy"), q_sim)

# Boundary conditions & lane info
np.save(os.path.join(model_dir, "upstream_flow.npy"), upstream_flow)
np.save(os.path.join(model_dir, "downstream_density.npy"), downstream_density)
np.save(os.path.join(model_dir, "num_lanes.npy"), lane_mapping)

print(f"\nSaved all outputs to '{model_dir}/'")

# 8. Plot comparison
plot_simulation_comparison(
    rho_sim=density_sim[:-1],
    v_sim=velocity_sim[:-1],
    rho_true=rho_hat,
    v_true=v_true,
    q_true=q_hat,
    include_fd=True,
    save_path="metanet_calibration_comparison.png",
    lanes=lane_mapping,
)

# 9. Print error metrics
mape_rho = np.mean(np.abs(density_sim[:-1] - rho_hat) / (rho_hat + 1e-6))
mape_v = np.mean(np.abs(velocity_sim[:-1] - v_true) / (v_true + 1e-6))
mape_q = np.mean(np.abs(q_sim - q_hat) / (q_hat + 1e-6))

print(f"\nCalibration Results:")
print(f"  MAPE Density:  {(mape_rho *100):.2f}%")
print(f"  MAPE Velocity: {(mape_v *100):.2f}%")
print(f"  MAPE Flow:     {(mape_q *100):.2f}%")
print(f"  Total Travel Time: {total_travel_time:.2f} hours")
