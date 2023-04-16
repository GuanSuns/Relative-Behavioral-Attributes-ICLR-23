import numpy as np


def pad_traj(traj_states, target_traj_len):
    """
    Append zeros to the end of traj_states to make its length into target_traj_len
    """
    traj_len = traj_states.shape[0]
    if traj_len == target_traj_len:
        return traj_states
    zero_pad = np.zeros(shape=(target_traj_len - traj_len, *traj_states[0].shape), dtype=traj_states.dtype)
    padded_traj = np.concatenate((traj_states, zero_pad), axis=0)
    return padded_traj
