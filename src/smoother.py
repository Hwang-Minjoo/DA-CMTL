from pykalman import KalmanFilter
import numpy as np

# Function for setting new kalman filter
def create_kalman_filter(initial_state_mean, obs_cov):
    return KalmanFilter(
        initial_state_mean=initial_state_mean,
        n_dim_obs=1,
        n_dim_state=1,
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=obs_cov, 
        transition_covariance=0.1
    )
    
# obs: CGM observation, obs_cov_val: how much reflect the observation value (smaller mean more reflects)
def get_ks_data(obs, obs_cov_val = 0.1):
    ks_data = np.zeros_like(obs)
    
    kf = create_kalman_filter(obs[0], obs_cov = obs_cov_val)
    state_means, _ = kf.smooth(obs[:])
    ks_data = state_means.squeeze()
    
    return ks_data