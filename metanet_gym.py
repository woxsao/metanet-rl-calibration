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
    ):
        num_timesteps, num_segments = rho_hat.shape
        print(num_timesteps, num_segments)
        self.T = T
        self.l = l
        self.num_segments = num_segments
        self.num_timesteps = num_timesteps

        self.off_ramp_mapping = off_ramp_mapping
        self.on_ramp_mapping = on_ramp_mapping
        self.lane_mapping = lane_mapping

        self.upstream_flow = upstream_flow
        self.downstream_density = downstream_density

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

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8 * num_segments,),
            dtype=np.float32,
        )

        # The actual per-parameter ranges live here
        self.param_ranges = {
            "tau": (1 / 3600.0, 60.0 / 3600),
            "K": (1.0, 50.0),
            "eta_high": (1.0, 90.0),
            "rho_crit": (1e-3, self.rho_max),
            "v_free": (50.0, self.v_max),
            "a": (0.5, 10.0),
            "beta": (0.0, 0.9),
            "r": (0.0, self.flow_max),
        }

    def _scale_action(self, action):
        param_names = ["tau", "K", "eta_high", "rho_crit", "v_free", "a", "beta", "r"]
        action_dict = {}
        for i, param_name in enumerate(param_names):
            start_idx = i * self.num_segments
            end_idx = (i + 1) * self.num_segments
            low, high = self.param_ranges[param_name]
            # action is already in [0, 1], just scale to [low, high]
            scaled = action[start_idx:end_idx] * (high - low) + low
            action_dict[param_name] = scaled.astype(np.float32)
        return action_dict

    def _compute_full_obs_dim(self):

        per_timestep = (
            3 * self.num_segments + 2
        )  # rho_pred, v_pred, q_pred, flow_origin, queue
        max_history = (self.num_timesteps + 1) * per_timestep
        current_ground_truth = 2 * self.num_segments  # rho_hat, q_hat
        timestep_info = 1

        return max_history + current_ground_truth + timestep_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Agent will reset when reached the last time step
        # initial conditions
        """
        density[0] = initial_density
        velocity[0] = initial_velocity
        flow[0] = np.array([initial_density[i] * initial_velocity[i] for i in range(num_segments)], dtype=float)
        flow_origin[0, 0] = initial_flow_or
        queue[0, 0] = initial_queue
        """
        super().reset(seed=seed)

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

    def _get_obs(self):
        t = min(self.current_timestep, self.num_timesteps - 1)

        obs = np.concatenate([
            self.rho_pred.flatten()    / self.rho_max,      # normalize
            self.v_pred.flatten()      / self.v_max,        # normalize
            self.q_pred.flatten()      / self.flow_max,     # normalize
            self.flow_origin.flatten() / self.flow_max,     # normalize
            self.queue.flatten()       / self.flow_max,     # normalize (or use queue_max if you have it)
            self.rho_hat[t]            / self.rho_max,      # normalize
            self.q_hat[t]              / self.flow_max,     # normalize
            np.array([t / self.num_timesteps]),             # normalize
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        obs = np.clip(obs, -5.0, 5.0)
        return obs

    def step(self, action):
        # Agent will take one metanet time step
        action_dict = self._scale_action(action)
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
            params=action_dict,
            lanes=self.lane_mapping,
        )
        has_nan = (np.isnan(d_tp1).any() or np.isnan(v_tp1).any() or 
               np.isnan(q_tp1) or np.isnan(fo_tp1) or np.isnan(f_tp1).any())
        has_inf = (np.isinf(d_tp1).any() or np.isinf(v_tp1).any() or 
                np.isinf(q_tp1) or np.isinf(fo_tp1) or np.isinf(f_tp1).any())
        
        if has_nan or has_inf:
            # Heavy penalty for bad parameters
            reward = -100.0
            terminated = True
            observation = self._get_obs()
            info = {'timestep': self.current_timestep, 'mape_rho': 1.0, 'mape_q': 1.0}
            return observation, reward, terminated, False, info
        
        self.rho_pred[t + 1] = d_tp1
        self.v_pred[t + 1] = v_tp1
        self.q_pred[t + 1] = f_tp1

        self.queue[t + 1, 0] = q_tp1
        self.flow_origin[t + 1, 0] = fo_tp1

        self.current_timestep += 1
        terminated = self.current_timestep >= self.num_timesteps
        t_clamped = min(self.current_timestep, self.num_timesteps - 1)

        # Use normalized MSE instead of MAPE
        rho_error_mse = np.mean((self.rho_pred[t_clamped] - self.rho_hat[t_clamped])**2) 
        q_error_mse = np.mean((self.q_pred[t_clamped] - self.q_hat[t_clamped])**2)

        reward = float(-(rho_error_mse + q_error_mse))
        reward = np.clip(reward, -1.0, 0.0)  # Keep in reasonable range

        observation = self._get_obs()

        # Reuse the same errors — no recomputation, stays in sync with reward
        info = {
            "timestep": self.current_timestep,
            "mape_rho": float(np.mean(rho_error_mse)),
            "mape_q": float(np.mean(q_error_mse)),
        }

        return observation, reward, terminated, False, info
