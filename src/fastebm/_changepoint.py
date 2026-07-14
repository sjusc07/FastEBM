import numpy as np


def l2_one_breakpoint_indices(
    values: np.ndarray,
    min_size: int = 2,
    jump: int = 5,
) -> np.ndarray:
    """Return one-breakpoint L2/SSE split indices for columns in ``values``.

    The candidate grid mirrors ``ruptures.Dynp(model="l2")`` defaults: split
    points must satisfy the minimum segment size and lie on the ``jump`` grid.
    A value of -1 means no valid split exists on that grid.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("values should be a one- or two-dimensional array.")

    n_observations, n_features = values.shape
    min_size = int(min_size)
    jump = int(jump)
    if min_size < 1:
        raise ValueError("min_size should be at least 1.")
    if jump < 1:
        raise ValueError("jump should be at least 1.")

    candidates = np.arange(min_size, n_observations - min_size + 1, dtype=int)
    candidates = candidates[candidates % jump == 0]
    if candidates.size == 0:
        return np.full(n_features, -1, dtype=int)

    prefix_sum = np.vstack(
        [np.zeros((1, n_features), dtype=float), np.cumsum(values, axis=0)]
    )
    prefix_sq_sum = np.vstack(
        [np.zeros((1, n_features), dtype=float), np.cumsum(values * values, axis=0)]
    )

    total_sum = prefix_sum[-1]
    total_sq_sum = prefix_sq_sum[-1]

    left_n = candidates[:, np.newaxis].astype(float)
    right_n = (n_observations - candidates)[:, np.newaxis].astype(float)

    left_sum = prefix_sum[candidates]
    left_sq_sum = prefix_sq_sum[candidates]
    right_sum = total_sum - left_sum
    right_sq_sum = total_sq_sum - left_sq_sum

    left_sse = left_sq_sum - (left_sum * left_sum / left_n)
    right_sse = right_sq_sum - (right_sum * right_sum / right_n)
    total_sse = left_sse + right_sse

    best_candidate_idx = np.argmin(total_sse, axis=0)
    return candidates[best_candidate_idx].astype(int, copy=False)
