from typing import Optional
import numpy as np
import gymnasium as gym
from metanet_calibration.metanet_dynamics import metanet_step


class METANETGymEnv(gym.Env):

    def __init__(
        self,
        T,
        l,
        rho_hat,
        q_hat,
        lane_mapping,
        off_ramp_mapping,
        on_ramp_mapping,
        upstream_flow,
        downstream_density,
        init_traffic_state,
        perturb_bc=True,  # NEW: Enable boundary condition perturbations
        bc_noise_std=0.05,  # NEW: Std dev of perturbation (5% by default)
        bc_smoothness=0.95,
        param_update_interval=1,
        custom_param_ranges=None,  # NEW: Allow custom parameter ranges
    ):
        num_timesteps, num_segments = rho_hat.shape
        print(num_timesteps, num_segments)
        self.T = T
        self.l = l
        self.num_segments = num_segments
        self.num_timesteps = num_timesteps

        self.param_update_interval = param_update_interval
        self.current_params = None

        self.lane_mapping = lane_mapping
        self.lanes_dict = {i: lane_mapping[i] for i in range(len(lane_mapping))}

        self.upstream_flow_base = upstream_flow.copy()
        self.downstream_density_base = downstream_density.copy()

        # Perturbation settings
        self.perturb_bc = perturb_bc
        self.bc_noise_std = bc_noise_std
        self.bc_smoothness = bc_smoothness

        # Initialize with base values (will be perturbed on reset)
        self.upstream_flow = upstream_flow.copy()
        self.downstream_density = downstream_density.copy()

        self.flow_max = np.max(q_hat)
        self.rho_max = np.max(rho_hat)
        self.v_max = np.max(q_hat / (rho_hat + 1e-6))

        obs_dim = self._compute_full_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        #
        self.current_timestep = 0
        self.rho_pred = np.zeros((num_timesteps + 1, num_segments))
        self.q_pred = np.zeros((num_timesteps + 1, num_segments))
        self.v_pred = np.zeros((num_timesteps + 1, num_segments))
        self.queue = np.zeros((num_timesteps + 1, 1))
        self.flow_origin = np.zeros((num_timesteps + 1, 1))
        self.init_traffic_state = init_traffic_state

        # populate initial conditions
        initial_density, initial_velocity, initial_flow_or, initial_queue = (
            self.init_traffic_state
        )
        self.rho_pred[0] = initial_density
        self.v_pred[0] = initial_velocity
        self.q_pred[0] = initial_density * initial_velocity
        self.flow_origin[0, 0] = initial_flow_or
        self.queue[0, 0] = initial_queue

        self.rho_hat = rho_hat
        self.q_hat = q_hat

        self.vsl_speeds = np.full((num_timesteps, num_segments), 1000)
        # print("shape of off ramp mapping:", off_ramp_mapping.shape)
        self.off_ramp_mapping = off_ramp_mapping
        self.on_ramp_mapping = on_ramp_mapping

        # Use sliced version for counting (14 segments, excluding boundaries)
        self.off_ramp_mapping_interior = off_ramp_mapping[1:-1]
        self.on_ramp_mapping_interior = on_ramp_mapping[1:-1]

        self.num_off_ramp_segments = int(np.sum(self.off_ramp_mapping_interior > 0))
        self.num_on_ramp_segments = int(np.sum(self.on_ramp_mapping_interior > 0))

        # Action space: 6 params for all segments + beta only for off-ramps + r only for on-ramps
        action_dim = (
            6 * num_segments  # tau, K, eta_high, rho_crit, v_free, a for all
            + self.num_off_ramp_segments  # beta only where off-ramps exist
            + self.num_on_ramp_segments
        )  # r only where on-ramps exist

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # The actual per-parameter ranges live here
        # self.param_ranges = {
        #     "tau": (1 / 3600.0, 60.0 / 3600),
        #     "K": (1.0, 50.0),
        #     "eta_high": (1.0, 90.0),
        #     "rho_crit": (15, self.rho_max),
        #     "v_free": (50.0, self.v_max),
        #     "a": (0.5, 10.0),
        #     "beta": (0.0, 0.9),
        #     "r": (0.0, self.flow_max),
        # }
        self.param_ranges = {
            "eta_high": (15.0, 60.0),
            "tau": (15.0 / 3600, 60.0 / 3600),
            "K": (5.0, 60.0),
            "rho_crit": (15, 100),
            "v_free": (110, 150),
            "a": (0.5, 5),
            "beta": (1e-3, 0.9),
            "r": (1e-3, 2000),
        }

        if custom_param_ranges is not None:
            self.param_ranges.update(custom_param_ranges)
        print(self.param_ranges)

   

    def _generate_smooth_noise(self, length, std, smoothness):
        """Generate temporally correlated noise using AR(1) process.

        Args:
            length: Number of timesteps
            std: Standard deviation of the noise
            smoothness: Temporal correlation (0.95 = very smooth, 0.5 = more random)

        Returns:
            Array of smooth noise values
        """
        noise = np.zeros(length)
        noise[0] = np.random.normal(0, std)
        for t in range(1, length):
            # AR(1) process: smooth random walk
            noise[t] = smoothness * noise[t - 1] + (1 - smoothness) * np.random.normal(
                0, std
            )
        return noise

    def _scale_action(self, action):
        """Scale action from [-1, 1] to parameter-specific ranges."""
        action_dict = {}
        idx = 0

        # Standard params for all segments
        for param_name in ["tau", "K", "eta_high", "rho_crit", "v_free", "a"]:
            low, high = self.param_ranges[param_name]
            scaled = (action[idx : idx + self.num_segments] + 1.0) / 2.0
            scaled = scaled * (high - low) + low
            action_dict[param_name] = scaled.astype(np.float32)
            idx += self.num_segments

        # Beta only for off-ramp segments (use interior mapping)
        beta = np.zeros(self.num_segments, dtype=np.float32)
        off_ramp_indices = np.where(self.off_ramp_mapping_interior > 0)[0]
        if len(off_ramp_indices) > 0:
            low, high = self.param_ranges["beta"]
            scaled = (action[idx : idx + self.num_off_ramp_segments] + 1.0) / 2.0
            scaled = scaled * (high - low) + low
            beta[off_ramp_indices] = scaled.astype(np.float32)
            idx += self.num_off_ramp_segments
        action_dict["beta"] = beta

        # R only for on-ramp segments (use interior mapping)
        r = np.zeros(self.num_segments, dtype=np.float32)
        on_ramp_indices = np.where(self.on_ramp_mapping_interior > 0)[0]
        if len(on_ramp_indices) > 0:
            low, high = self.param_ranges["r"]
            scaled = (action[idx : idx + self.num_on_ramp_segments] + 1.0) / 2.0
            scaled = scaled * (high - low) + low
            r[on_ramp_indices] = scaled.astype(np.float32)
            idx += self.num_on_ramp_segments
        action_dict["r"] = r

        return action_dict

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_params = None

        # Perturb boundary conditions on each reset
        if self.perturb_bc:
            # Generate smooth noise (not frame-by-frame jumps)
            upstream_noise = self._generate_smooth_noise(
                len(self.upstream_flow_base), self.bc_noise_std, self.bc_smoothness
            )
            downstream_noise = self._generate_smooth_noise(
                len(self.downstream_density_base), self.bc_noise_std, self.bc_smoothness
            )

            # Apply multiplicative noise
            self.upstream_flow = self.upstream_flow_base * (1 + upstream_noise)
            self.downstream_density = self.downstream_density_base * (
                1 + downstream_noise
            )

            # Clip to physical bounds
            self.upstream_flow = np.clip(self.upstream_flow, 0, self.flow_max)
            self.downstream_density = np.clip(self.downstream_density, 0, self.rho_max)
        else:
            # Use base values without perturbation
            self.upstream_flow = self.upstream_flow_base.copy()
            self.downstream_density = self.downstream_density_base.copy()

        self.current_timestep = 0
        initial_density, initial_velocity, initial_flow_or, initial_queue = (
            self.init_traffic_state
        )

        # Reset all arrays
        self.rho_pred = np.zeros((self.num_timesteps + 1, self.num_segments))
        self.q_pred = np.zeros((self.num_timesteps + 1, self.num_segments))
        self.v_pred = np.zeros((self.num_timesteps + 1, self.num_segments))
        self.flow_origin = np.zeros((self.num_timesteps + 1, 1))
        self.queue = np.zeros((self.num_timesteps + 1, 1))

        # Set initial conditions
        self.rho_pred[0] = initial_density
        self.v_pred[0] = initial_velocity
        self.q_pred[0] = initial_density * initial_velocity
        self.flow_origin[0, 0] = initial_flow_or
        self.queue[0, 0] = initial_queue

        observation = self._get_obs()
        info = {}

        return observation, info

    def _compute_full_obs_dim(self):
        # Current state: rho, v, q, flow_origin, queue
        current_state = 3 * self.num_segments + 2
        # Ground truth at current timestep: rho_hat, q_hat
        current_ground_truth = 2 * self.num_segments
        # Timestep indicator
        timestep_info = 2

        return current_state + timestep_info + current_ground_truth

    def _get_obs(self):
        t = min(self.current_timestep, self.num_timesteps - 1)

        obs = np.concatenate(
            [
                # Current state (not full history)
                self.rho_pred[t] / self.rho_max,  # current density
                self.v_pred[t] / self.v_max,  # current velocity
                self.q_pred[t] / self.flow_max,  # current flow
                self.flow_origin[t : t + 1, 0]
                / self.flow_max,  # current flow_origin (scalar)
                self.queue[t : t + 1, 0] / self.flow_max,  # current queue (scalar)
                # Ground truth at current timestep
                self.rho_hat[t] / self.rho_max,  # ground truth density
                self.q_hat[t] / self.flow_max,  # ground truth flow
                # Timestep indicator
                np.array([t / self.num_timesteps]),
                np.array([float(t % self.param_update_interval == 0)]),
            ]
        ).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        obs = np.clip(obs, -5.0, 5.0)
        return obs

    def step(self, action, override_params=None):
        if override_params is not None:
            # DON'T extract timestep here - pass full arrays
            action_dict = override_params
        else:
            if self.current_timestep % self.param_update_interval == 0:
                # Time to update parameters
                self.current_params = self._scale_action(action)
            action_dict = self.current_params

        t = self.current_timestep
        d_tp1, v_tp1, q_tp1, fo_tp1, f_tp1 = metanet_step(
            t,
            self.rho_pred[t],
            self.v_pred[t],
            self.queue[t, 0],
            self.flow_origin[t, 0],
            T=self.T,
            l=self.l,
            vsl_speeds=self.vsl_speeds,
            demand=self.upstream_flow,
            downstream_density=self.downstream_density,
            params=action_dict,  # Pass full 2D arrays, let metanet_step index by t
            lanes=self.lanes_dict,
            real_data=True,
        )
        has_nan = (
            np.isnan(d_tp1).any()
            or np.isnan(v_tp1).any()
            or np.isnan(q_tp1)
            or np.isnan(fo_tp1)
            or np.isnan(f_tp1).any()
        )
        has_inf = (
            np.isinf(d_tp1).any()
            or np.isinf(v_tp1).any()
            or np.isinf(q_tp1)
            or np.isinf(fo_tp1)
            or np.isinf(f_tp1).any()
        )

        if has_nan or has_inf:
            # Heavy penalty for bad parameters
            reward = -100.0
            terminated = self.current_timestep >= self.num_timesteps

            # Replace NaN/inf with clamped values so simulation continues
            d_tp1 = np.nan_to_num(
                d_tp1, nan=self.rho_max, posinf=self.rho_max, neginf=0.0
            )
            v_tp1 = np.nan_to_num(v_tp1, nan=0.0, posinf=self.v_max, neginf=0.0)
            f_tp1 = np.nan_to_num(f_tp1, nan=0.0, posinf=self.flow_max, neginf=0.0)
            q_tp1 = np.nan_to_num(q_tp1, nan=0.0, posinf=self.flow_max, neginf=0.0)
            fo_tp1 = np.nan_to_num(fo_tp1, nan=0.0, posinf=self.flow_max, neginf=0.0)

        self.rho_pred[t + 1] = d_tp1
        self.v_pred[t + 1] = v_tp1
        self.q_pred[t + 1] = f_tp1

        self.queue[t + 1, 0] = q_tp1
        self.flow_origin[t + 1, 0] = fo_tp1

        self.current_timestep += 1
        terminated = self.current_timestep >= self.num_timesteps
        t_clamped = min(self.current_timestep, self.num_timesteps - 1)

        # Use normalized MSE instead of MAPE
        rho_scale = np.maximum(self.rho_hat[t_clamped], 0.1 * self.rho_max)
        q_scale = np.maximum(self.q_hat[t_clamped], 0.1 * self.flow_max)
        rho_mape = np.mean(
            np.abs(self.rho_pred[t_clamped] - self.rho_hat[t_clamped]) / rho_scale
        )
        q_mape = np.mean(
            np.abs(self.q_pred[t_clamped] - self.q_hat[t_clamped]) / q_scale
        )
        v_mape = np.mean(
            np.abs(
                self.v_pred[t_clamped]
                - self.q_hat[t_clamped] / (self.rho_hat[t_clamped] + 1e-6)
            )
            / (self.v_max + 1e-6)
        )
        reward = -(rho_mape + q_mape + 100 * v_mape)
        if terminated:
            rho_scale_all = np.maximum(self.rho_hat, 0.1 * self.rho_max)
            q_scale_all = np.maximum(self.q_hat, 0.1 * self.flow_max)
            total_rho_error = np.mean(
                np.abs(self.rho_pred[:-1, :] - self.rho_hat) / rho_scale_all
            )
            total_q_error = np.mean(
                np.abs(self.q_pred[:-1, :] - self.q_hat) / q_scale_all
            )
            reward += -10.0 * (total_rho_error + total_q_error)

        reward = np.clip(reward, -100.0, 0.0)

        observation = self._get_obs()

        # Reuse the same errors — no recomputation, stays in sync with reward
        info = {
            "timestep": self.current_timestep,
            "mape_rho": rho_mape,  # RMSE-like metric
            "mape_q": q_mape,
        }

        return observation, reward, terminated, False, info
