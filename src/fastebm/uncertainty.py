from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ._changepoint import l2_one_breakpoint_indices

@dataclass
class FastEBMFirstPassageUncertaintyResult:
    ordered_subjects: pd.DataFrame
    base_transition_matrix: pd.DataFrame
    stationary_distribution: pd.Series
    mean_first_passage_matrix: pd.DataFrame
    first_passage_variance_matrix: pd.DataFrame
    subject_variance: pd.DataFrame
    sample_metadata: pd.DataFrame
    subject_dps_samples: pd.DataFrame
    subject_dps_summary: pd.DataFrame
    subject_rank_samples: pd.DataFrame
    subject_rank_summary: pd.DataFrame
    event_position_samples: pd.DataFrame
    event_rank_samples: pd.DataFrame
    event_summary: pd.DataFrame
    pairwise_precedence: pd.DataFrame
    subject_precedence: pd.DataFrame | None = None
    calculation_time_seconds: float | None = None
    convergence_history: pd.DataFrame | None = None


_VALID_ORDERING_METHODS = {"mean_template", "median_template"}


def _validate_ordering_method(method: str) -> str:
    if method not in _VALID_ORDERING_METHODS:
        valid = ", ".join(sorted(_VALID_ORDERING_METHODS))
        raise ValueError(
            f"Unsupported ordering_method '{method}'. Expected one of: {valid}."
        )
    return method



def _normalize_dps(source_dps: np.ndarray) -> np.ndarray:
    # FastEBM uses relative disease-progression position, so raw distances from
    # the control template are rescaled to [0, 1].
    source_dps = np.asarray(source_dps, dtype=float)
    dps_min = np.min(source_dps)
    dps_max = np.max(source_dps)
    if np.isclose(dps_min, dps_max):
        return np.zeros_like(source_dps)
    return (source_dps - dps_min) / (dps_max - dps_min)


def _normalize_sample_matrix(samples: np.ndarray) -> np.ndarray:
    # Normalize each Monte Carlo DPS sample independently. This preserves the
    # within-sample subject order while keeping every sampled DPS axis on the
    # same [0, 1] scale.
    samples = np.asarray(samples, dtype=float)
    sample_min = np.min(samples, axis=1, keepdims=True)
    sample_max = np.max(samples, axis=1, keepdims=True)
    sample_range = sample_max - sample_min
    return np.divide(
        samples - sample_min,
        sample_range,
        out=np.zeros_like(samples, dtype=float),
        where=sample_range > np.finfo(float).eps,
    )


def _summarize_samples(samples: pd.DataFrame, prefix: str) -> pd.DataFrame:
    summary = pd.DataFrame(index=samples.index)
    summary[f"{prefix}_mean"] = samples.mean(axis=1)
    summary[f"{prefix}_median"] = samples.median(axis=1)
    summary[f"{prefix}_std"] = samples.std(axis=1, ddof=0)
    summary[f"{prefix}_ci_lower"] = samples.quantile(0.025, axis=1)
    summary[f"{prefix}_ci_upper"] = samples.quantile(0.975, axis=1)
    return summary




def _rank_rows(values: np.ndarray) -> np.ndarray:
    # Convert each row of values into ranks. NaNs are treated as late/infinite
    # positions so they sort after finite event positions.
    sortable_values = np.asarray(values, dtype=float)
    sortable_values = np.where(np.isnan(sortable_values), np.inf, sortable_values)
    order = np.argsort(sortable_values, axis=1, kind="mergesort")
    ranks = np.empty(order.shape, dtype=float)
    row_index = np.arange(order.shape[0])[:, np.newaxis]
    ranks[row_index, order] = np.arange(order.shape[1], dtype=float)
    return ranks



def _pairwise_precedence_array_from_positions(position_samples: np.ndarray) -> np.ndarray:
    # Estimate P(event i precedes event j) from sampled event positions.
    # Ties contribute 0.5 so the matrix remains symmetric around indecision.
    position_samples = np.asarray(position_samples, dtype=float)
    before = position_samples[:, :, np.newaxis] < position_samples[:, np.newaxis, :]
    tied = np.isclose(position_samples[:, :, np.newaxis], position_samples[:, np.newaxis, :])
    precedence = before.mean(axis=0) + 0.5 * tied.mean(axis=0)
    np.fill_diagonal(precedence, 0.5)
    return precedence


def _pairwise_precedence_from_positions(position_samples: pd.DataFrame) -> pd.DataFrame:
    features = list(position_samples.columns)
    precedence = pd.DataFrame(index=features, columns=features, dtype=float)

    for feature_i in features:
        for feature_j in features:
            if feature_i == feature_j:
                precedence.loc[feature_i, feature_j] = 0.5
                continue

            before = position_samples[feature_i] < position_samples[feature_j]
            tied = np.isclose(position_samples[feature_i], position_samples[feature_j])
            precedence.loc[feature_i, feature_j] = before.mean() + 0.5 * tied.mean()

    return precedence


def _validate_first_passage_uncertainty_inputs(
    data: pd.DataFrame,
    n_noise_samples: int,
    n_neighbors: int,
    support_n_neighbors: int | None,
    min_segment_size: int,
    dps_noise_scale: float | None,
    convergence_check_interval: int | None,
    convergence_min_samples: int | None,
    event_rank_convergence_tol: float | None,
    pairwise_precedence_convergence_tol: float | None,
    convergence_patience: int,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a pandas DataFrame.")

    if data.shape[0] < 3:
        raise ValueError("Input data should contain at least three subjects.")

    if data.shape[1] < 2:
        raise ValueError("Input data should contain at least one feature column and one stage column.")

    if not isinstance(n_noise_samples, int) or n_noise_samples < 1:
        raise ValueError("n_noise_samples should be a positive integer.")

    if not isinstance(n_neighbors, int):
        raise TypeError("n_neighbors should be an integer.")

    if n_neighbors < 1 or n_neighbors > data.shape[0] - 1:
        raise ValueError(
            f"n_neighbors should be between 1 and {data.shape[0] - 1} "
            f"for a dataset with {data.shape[0]} subjects."
        )

    if support_n_neighbors is not None:
        if not isinstance(support_n_neighbors, int):
            raise TypeError("support_n_neighbors should be an integer or None.")
        if support_n_neighbors < 1 or support_n_neighbors > data.shape[0] - 1:
            raise ValueError(
                f"support_n_neighbors should be between 1 and {data.shape[0] - 1} "
                f"for a dataset with {data.shape[0]} subjects."
            )

    if min_segment_size < 1:
        raise ValueError("min_segment_size should be at least 1.")

    if dps_noise_scale is not None and dps_noise_scale < 0:
        raise ValueError("dps_noise_scale should be non-negative.")

    if convergence_check_interval is not None:
        if not isinstance(convergence_check_interval, int):
            raise TypeError("convergence_check_interval should be an integer or None.")
        if convergence_check_interval < 1:
            raise ValueError("convergence_check_interval should be at least 1.")

    if convergence_min_samples is not None:
        if not isinstance(convergence_min_samples, int):
            raise TypeError("convergence_min_samples should be an integer or None.")
        if convergence_min_samples < 1:
            raise ValueError("convergence_min_samples should be at least 1.")

    if event_rank_convergence_tol is not None and event_rank_convergence_tol <= 0:
        raise ValueError("event_rank_convergence_tol should be positive.")

    if pairwise_precedence_convergence_tol is not None and pairwise_precedence_convergence_tol <= 0:
        raise ValueError("pairwise_precedence_convergence_tol should be positive.")

    if not isinstance(convergence_patience, int):
        raise TypeError("convergence_patience should be an integer.")

    if convergence_patience < 1:
        raise ValueError("convergence_patience should be at least 1.")

    stage_values = data.iloc[:, -1].dropna()
    if stage_values.empty:
        raise ValueError("Last column should contain stage information.")

    if 1 not in set(stage_values.tolist()):
        raise ValueError("Control/reference subjects must be labeled stage == 1 in the last column.")


def _build_row_stochastic_transition(features: pd.DataFrame, n_neighbors: int) -> np.ndarray:
    pairwise_distance = pairwise_distances(features, features, metric="nan_euclidean") ** 2
    np.fill_diagonal(pairwise_distance, 0.0)
    if not np.all(np.isfinite(pairwise_distance)):
        raise ValueError(
            "Transition matrix construction failed because at least one pair of subjects "
            "has no finite feature overlap."
        )

    neighbor_count = min(n_neighbors + 1, len(features))
    nearest_n_dists, _ = (
        NearestNeighbors(n_neighbors=neighbor_count, p=2, metric="nan_euclidean")
        .fit(features)
        .kneighbors(n_neighbors=neighbor_count)
    )
    epsilon_squared = nearest_n_dists[:, -1] ** 2
    gamma = epsilon_squared[:, np.newaxis] + epsilon_squared
    gamma = np.maximum(gamma, np.finfo(float).eps)

    similarity = np.exp(-pairwise_distance / gamma)
    np.fill_diagonal(similarity, 0.0)

    density = np.maximum(np.sum(similarity, axis=1), np.finfo(float).eps)
    kernel = similarity / np.outer(density, density)
    np.fill_diagonal(kernel, 0.0)

    degree = np.sum(kernel, axis=1)
    if np.any(degree <= 0) or not np.all(np.isfinite(degree)):
        raise ValueError("Transition matrix construction failed because at least one row has zero degree.")

    transition = kernel / degree[:, np.newaxis]
    if not np.all(np.isfinite(transition)):
        raise ValueError("Transition matrix construction produced non-finite values.")

    return transition


def _is_support_connected(support: np.ndarray) -> bool:
    visited = np.zeros(support.shape[0], dtype=bool)
    stack = [0]
    visited[0] = True

    while stack:
        row = stack.pop()
        neighbors = np.flatnonzero(support[row])
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)

    return bool(visited.all())


def _sparsify_transition(transition: np.ndarray, support_n_neighbors: int | None) -> np.ndarray:
    # Optional support sparsification keeps only the strongest local connections.
    # It is disabled by default because the exact first-passage formulas below
    # are defined on the full transition matrix.
    if support_n_neighbors is None or support_n_neighbors >= transition.shape[0] - 1:
        return transition.copy()

    n_subjects = transition.shape[0]
    support = np.zeros_like(transition, dtype=bool)

    for row_idx in range(n_subjects):
        row = transition[row_idx].copy()
        row[row_idx] = 0.0
        supported_indices = np.argpartition(row, -support_n_neighbors)[-support_n_neighbors:]
        support[row_idx, supported_indices] = row[supported_indices] > 0

    support = support | support.T
    np.fill_diagonal(support, False)

    if not _is_support_connected(support):
        raise ValueError(
            "The requested support_n_neighbors produced a disconnected transition support. "
            "Increase support_n_neighbors or set it to None."
        )

    sparse_transition = np.where(support, transition, 0.0)
    row_sums = sparse_transition.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError(
            "The requested support_n_neighbors produced at least one empty transition row. "
            "Increase support_n_neighbors or set it to None."
        )

    return sparse_transition / row_sums[:, np.newaxis]


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    n_subjects = transition.shape[0]
    system = transition.T - np.eye(n_subjects)
    system[-1, :] = 1.0
    target = np.zeros(n_subjects)
    target[-1] = 1.0

    try:
        stationary = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eig(transition.T)
        stationary = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        if stationary.sum() < 0:
            stationary = -stationary

    stationary = np.clip(np.real(stationary), 0.0, None)
    total = stationary.sum()
    if total <= np.finfo(float).eps or not np.isfinite(total):
        raise ValueError("Could not compute a finite stationary distribution.")

    return stationary / total


def _fast_ebm_fundamental_matrix_from_transition(transition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition should be a square matrix.")

    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("transition should be row-stochastic.")

    stationary = _stationary_distribution(transition)
    stationary_matrix = np.tile(stationary, (transition.shape[0], 1))
    fundamental = np.linalg.inv(np.eye(transition.shape[0]) - transition + stationary_matrix) - np.eye(
        transition.shape[0]
    )

    if not np.all(np.isfinite(fundamental)):
        raise ValueError("FastEBM fundamental matrix contains non-finite values.")

    return fundamental, stationary


def _mean_first_passage_matrix(
    fast_ebm_fundamental: np.ndarray,
    stationary: np.ndarray,
) -> np.ndarray:
    fundamental = fast_ebm_fundamental + np.eye(fast_ebm_fundamental.shape[0])
    inverse_stationary = np.diag(1.0 / np.maximum(stationary, np.finfo(float).eps))
    diagonal_fundamental = np.diag(np.diag(fundamental))
    ones = np.ones_like(fundamental)

    mean_first_passage = (
        np.eye(fundamental.shape[0])
        - fundamental
        + ones @ diagonal_fundamental
    ) @ inverse_stationary

    if not np.all(np.isfinite(mean_first_passage)):
        raise ValueError("Mean first-passage matrix contains non-finite values.")

    return np.maximum(mean_first_passage, 0.0)


def _first_passage_variance_matrix(
    fast_ebm_fundamental: np.ndarray,
    mean_first_passage: np.ndarray,
    stationary: np.ndarray,
) -> np.ndarray:
    fundamental = fast_ebm_fundamental + np.eye(fast_ebm_fundamental.shape[0])
    inverse_stationary = np.diag(1.0 / np.maximum(stationary, np.finfo(float).eps))
    diagonal_fundamental = np.diag(np.diag(fundamental))
    ones = np.ones_like(fundamental)
    diagonal_zm = np.diag(np.diag(fundamental @ mean_first_passage))

    second_moment = (
        mean_first_passage @ (2 * diagonal_fundamental @ inverse_stationary - np.eye(fundamental.shape[0]))
        + 2 * (fundamental @ mean_first_passage - ones @ diagonal_zm)
    )
    variance = second_moment - mean_first_passage**2

    if not np.all(np.isfinite(variance)):
        raise ValueError("First-passage variance matrix contains non-finite values.")

    return np.maximum(variance, 0.0)


def _scaled_subject_noise_variance(
    subject_variance: np.ndarray,
    dps_noise_scale: float,
    normalize_subject_variance: bool,
    raw_dps_range: float | None = None,
) -> np.ndarray:
    subject_variance = np.maximum(np.asarray(subject_variance, dtype=float), 0.0)

    if normalize_subject_variance:
        positive_variance = subject_variance[subject_variance > np.finfo(float).eps]
        if positive_variance.size == 0:
            normalized_variance = np.zeros_like(subject_variance)
        else:
            normalized_variance = subject_variance / np.median(positive_variance)
    else:
        if raw_dps_range is None:
            raise ValueError("raw_dps_range is required when normalize_subject_variance is False.")
        if raw_dps_range <= np.finfo(float).eps:
            normalized_variance = np.zeros_like(subject_variance)
        else:
            normalized_variance = subject_variance / (raw_dps_range**2)

    return (dps_noise_scale**2) * normalized_variance


def _median_local_dps_spacing(dps: np.ndarray) -> tuple[float, int]:
    # Automatic DPS scale: use median spacing across a local sqrt(n)-sized
    # neighborhood so perturbations are tied to the observed DPS resolution.
    sorted_dps = np.sort(np.asarray(dps, dtype=float))
    if len(sorted_dps) <= 1:
        return 0.0, 0

    local_window = max(1, int(np.ceil(np.sqrt(len(sorted_dps)))))
    local_window = min(local_window, len(sorted_dps) - 1)
    local_differences = sorted_dps[local_window:] - sorted_dps[:-local_window]
    positive_differences = local_differences[local_differences > np.finfo(float).eps]

    if positive_differences.size > 0:
        return float(np.median(positive_differences)), local_window

    return float(local_window / (len(sorted_dps) - 1)), local_window


def _compute_dps_from_fundamental(
    fundamental: np.ndarray,
    stage: pd.Series,
    ordering_method: str,
) -> np.ndarray:
    # Match FastEBM's subject DPS definition: Euclidean distance from each
    # subject's fundamental-matrix row to the control mean/median template.
    control_mask = stage.to_numpy() == 1
    if not control_mask.any():
        raise ValueError("Control/reference subjects must be labeled stage == 1 in the last column.")

    reference_group = fundamental[control_mask, :]
    if ordering_method == "mean_template":
        centroid = np.mean(reference_group, axis=0)
    else:
        centroid = np.median(reference_group, axis=0)

    return np.sqrt(np.sum((fundamental - centroid) ** 2, axis=1))


def _ordered_subjects_from_dps(feature_data: pd.DataFrame, dps: np.ndarray) -> pd.DataFrame:
    # Reorder subjects by normalized DPS while keeping original subject indices,
    # which lets downstream sample tables map back to the input rows.
    ordered_subjects = feature_data.copy(deep=True)
    ordered_subjects["DPS"] = _normalize_dps(dps)
    return ordered_subjects.sort_values("DPS", ascending=True, kind="mergesort")



def _deterministic_event_position(
    ordered_subjects: pd.DataFrame,
    feature: str,
    anamoly_model: str,
    min_segment_size: int,
) -> float:
    # Given an ordered subject table, estimate one event position for one feature.
    # The default L2 path uses the vectorized SSE changepoint helper; other
    # anomaly models fall back to ruptures.
    valid = ordered_subjects.loc[:, [feature, "DPS"]].dropna()
    if len(valid) < 2 * min_segment_size:
        return float(valid["DPS"].median())

    if anamoly_model == "l2":
        breakpoint = l2_one_breakpoint_indices(
            valid[feature].to_numpy(dtype=float),
            min_size=min_segment_size,
        )[0]
        if breakpoint < 0:
            return float(valid["DPS"].median())
        breakpoint = min(max(int(breakpoint), 0), len(valid) - 1)
        return float(valid["DPS"].iloc[breakpoint])

    try:
        model = rpt.Dynp(model=anamoly_model, min_size=min_segment_size)
        breakpoints = model.fit(valid[feature].to_numpy(dtype=float)).predict(n_bkps=1)
        breakpoint = min(max(int(breakpoints[0]), 0), len(valid) - 1)
        return float(valid["DPS"].iloc[breakpoint])
    except (rpt.exceptions.BadSegmentationParameters, rpt.exceptions.NotEnoughPoints, ValueError):
        return float(valid["DPS"].median())


def _event_positions_for_ordered_subjects(
    ordered_subjects: pd.DataFrame,
    feature_names: list[str],
    anamoly_model: str,
    min_segment_size: int,
) -> pd.Series:
    # Batch event-position estimation. If the ordered feature matrix has no NaNs
    # and uses the L2 model, all features are processed together; otherwise the
    # code falls back to per-feature NA-aware processing.
    if anamoly_model == "l2":
        if not feature_names:
            return pd.Series(dtype=float)

        feature_frame = ordered_subjects.loc[:, feature_names]
        dps = ordered_subjects["DPS"].to_numpy(dtype=float)
        values = feature_frame.to_numpy(dtype=float)

        if not np.isnan(values).any():
            breakpoints = l2_one_breakpoint_indices(values, min_size=min_segment_size)
            median_dps = float(np.median(dps))
            positions = np.full(len(feature_names), median_dps, dtype=float)
            valid_breakpoints = breakpoints >= 0
            if valid_breakpoints.any():
                clipped_breakpoints = np.clip(
                    breakpoints[valid_breakpoints], 0, len(dps) - 1
                )
                positions[valid_breakpoints] = dps[clipped_breakpoints]
            return pd.Series(positions, index=feature_names, dtype=float)

        positions = {}
        for feature in feature_names:
            valid = ordered_subjects.loc[:, [feature, "DPS"]].dropna()
            if len(valid) < 2 * min_segment_size:
                positions[feature] = float(valid["DPS"].median())
                continue

            breakpoint = l2_one_breakpoint_indices(
                valid[feature].to_numpy(dtype=float),
                min_size=min_segment_size,
            )[0]
            if breakpoint < 0:
                positions[feature] = float(valid["DPS"].median())
                continue

            breakpoint = min(max(int(breakpoint), 0), len(valid) - 1)
            positions[feature] = float(valid["DPS"].iloc[breakpoint])

        return pd.Series(positions, dtype=float)

    positions = {}

    for feature in feature_names:
        positions[feature] = _deterministic_event_position(
            ordered_subjects=ordered_subjects,
            feature=feature,
            anamoly_model=anamoly_model,
            min_segment_size=min_segment_size,
        )

    return pd.Series(positions, dtype=float)



def _subject_precedence_from_ranks(subject_rank_samples: pd.DataFrame) -> pd.DataFrame:
    # Optional subject-level analogue of pairwise event precedence.
    subjects = list(subject_rank_samples.index)
    precedence = pd.DataFrame(index=subjects, columns=subjects, dtype=float)

    for subject_i in subjects:
        for subject_j in subjects:
            if subject_i == subject_j:
                precedence.loc[subject_i, subject_j] = 0.5
                continue

            before = subject_rank_samples.loc[subject_i] < subject_rank_samples.loc[subject_j]
            tied = subject_rank_samples.loc[subject_i] == subject_rank_samples.loc[subject_j]
            precedence.loc[subject_i, subject_j] = before.mean() + 0.5 * tied.mean()

    return precedence


def first_passage_uncertainty(
    data: pd.DataFrame,
    n_noise_samples: int = 100,
    ordering_method: str = "median_template",
    n_neighbors: int = 5,
    support_n_neighbors: int | None = None,
    anamoly_model: str = "l2",
    min_segment_size: int = 3,
    dps_noise_scale: float | None = None,
    normalize_subject_variance: bool = True,
    return_subject_precedence: bool = False,
    convergence_check_interval: int | None = None,
    convergence_min_samples: int | None = None,
    event_rank_convergence_tol: float | None = None,
    pairwise_precedence_convergence_tol: float | None = None,
    convergence_patience: int = 2,
    seed: int | None = None,
) -> FastEBMFirstPassageUncertaintyResult:
    """Propagate Kemeny-Snell first-passage variance as additive DPS noise.

    Subject-level first-passage variance is aggregated by weighting each
    destination-state variance by the stationary distribution.

    If normalize_subject_variance is True and dps_noise_scale is None, the
    noise scale is set to the median local DPS spacing over ceil(sqrt(n_subjects))
    neighbors. If normalize_subject_variance is False, raw subject variance is
    converted to the normalized DPS scale by dividing by the squared raw DPS range;
    dps_noise_scale is then an optional additional multiplier.

    calculation_time_seconds measures the core numerical propagation work:
    transition construction, first-passage moments, DPS perturbation,
    reordering, and event-position calculation. It excludes final DataFrame
    assembly, summaries, pairwise precedence, and result construction.

    If event_rank_convergence_tol or pairwise_precedence_convergence_tol is
    supplied, n_noise_samples is treated as a maximum sample count and sampling
    stops early after the requested convergence criterion is satisfied for
    convergence_patience consecutive checks.
    """
    ordering_method = _validate_ordering_method(ordering_method)
    _validate_first_passage_uncertainty_inputs(
        data=data,
        n_noise_samples=n_noise_samples,
        n_neighbors=n_neighbors,
        support_n_neighbors=support_n_neighbors,
        min_segment_size=min_segment_size,
        dps_noise_scale=dps_noise_scale,
        convergence_check_interval=convergence_check_interval,
        convergence_min_samples=convergence_min_samples,
        event_rank_convergence_tol=event_rank_convergence_tol,
        pairwise_precedence_convergence_tol=pairwise_precedence_convergence_tol,
        convergence_patience=convergence_patience,
    )

    feature_names = list(data.columns[:-1])
    feature_data = data.loc[:, feature_names]
    stage = data.iloc[:, -1]
    subject_index = data.index
    rng = np.random.default_rng(seed)

    # construct the base Markov chain, compute the FastEBM
    # fundamental matrix, derive baseline DPS, and calculate first-passage
    # moments before any Monte Carlo sampling.
    calculation_start_time = perf_counter()
    base_transition = _build_row_stochastic_transition(feature_data, n_neighbors=n_neighbors)
    base_transition = _sparsify_transition(base_transition, support_n_neighbors=support_n_neighbors)
    fast_ebm_fundamental, stationary = _fast_ebm_fundamental_matrix_from_transition(base_transition)
    fast_ebm_fundamental_for_dps = fast_ebm_fundamental.clip(min=0.0)
    raw_base_dps = _compute_dps_from_fundamental(
        fundamental=fast_ebm_fundamental_for_dps,
        stage=stage,
        ordering_method=ordering_method,
    )
    raw_base_dps_range = float(np.max(raw_base_dps) - np.min(raw_base_dps))
    base_dps = _normalize_dps(raw_base_dps)
    automatic_dps_local_window = None
    if dps_noise_scale is None:
        if normalize_subject_variance:
            effective_dps_noise_scale, automatic_dps_local_window = _median_local_dps_spacing(base_dps)
        else:
            effective_dps_noise_scale = 1.0
    else:
        effective_dps_noise_scale = dps_noise_scale

    mean_first_passage = _mean_first_passage_matrix(
        fast_ebm_fundamental=fast_ebm_fundamental,
        stationary=stationary,
    )
    first_passage_variance = _first_passage_variance_matrix(
        fast_ebm_fundamental=fast_ebm_fundamental,
        mean_first_passage=mean_first_passage,
        stationary=stationary,
    )
    subject_first_passage_variance = first_passage_variance @ stationary
    # subject_first_passage_variance = first_passage_variance.mean(axis=1)
    # Aggregate each source state's destination-wise first-passage variance into
    # one scalar variance per subject using stationary destination weights.
    subject_noise_variance = _scaled_subject_noise_variance(
        subject_variance=subject_first_passage_variance,
        dps_noise_scale=effective_dps_noise_scale,
        normalize_subject_variance=normalize_subject_variance,
        raw_dps_range=raw_base_dps_range,
    )

    convergence_enabled = (
        event_rank_convergence_tol is not None
        or pairwise_precedence_convergence_tol is not None
    )
    if convergence_enabled:
        effective_convergence_check_interval = convergence_check_interval or min(25, n_noise_samples)
    else:
        effective_convergence_check_interval = n_noise_samples
    effective_convergence_min_samples = convergence_min_samples or effective_convergence_check_interval
    effective_convergence_min_samples = min(effective_convergence_min_samples, n_noise_samples)

    run_ids = []
    subject_dps_batches = []
    subject_rank_batches = []
    event_position_batches = []
    mean_abs_dps_noise = []
    max_abs_dps_noise = []
    convergence_records = []
    previous_event_rank_mean = None
    previous_pairwise_precedence = None
    stable_convergence_checks = 0
    adaptive_converged = False

    samples_completed = 0
    while samples_completed < n_noise_samples:
        batch_size = min(effective_convergence_check_interval, n_noise_samples - samples_completed)
        batch_run_ids = [
            f"first_passage_{sample_id:04d}"
            for sample_id in range(samples_completed, samples_completed + batch_size)
        ]
        noise_batch = rng.normal(
            loc=0.0,
            scale=np.sqrt(subject_noise_variance),
            size=(batch_size, len(base_dps)),
        )
        sampled_dps_batch = _normalize_sample_matrix(base_dps[np.newaxis, :] + noise_batch)
        sampled_rank_batch = _rank_rows(sampled_dps_batch)
        event_position_batch = np.empty((batch_size, len(feature_names)), dtype=float)

        for batch_idx, sampled_dps in enumerate(sampled_dps_batch):
            sampled_ordered_subjects = _ordered_subjects_from_dps(
                feature_data=feature_data,
                dps=sampled_dps,
            )
            sampled_event_positions = _event_positions_for_ordered_subjects(
                ordered_subjects=sampled_ordered_subjects,
                feature_names=feature_names,
                anamoly_model=anamoly_model,
                min_segment_size=min_segment_size,
            )
            event_position_batch[batch_idx, :] = sampled_event_positions.reindex(feature_names).to_numpy(dtype=float)

        run_ids.extend(batch_run_ids)
        subject_dps_batches.append(sampled_dps_batch)
        subject_rank_batches.append(sampled_rank_batch)
        event_position_batches.append(event_position_batch)
        mean_abs_dps_noise.extend(np.mean(np.abs(noise_batch), axis=1).astype(float).tolist())
        max_abs_dps_noise.extend(np.max(np.abs(noise_batch), axis=1).astype(float).tolist())
        samples_completed += batch_size

        if convergence_enabled and samples_completed >= effective_convergence_min_samples:
            current_event_positions = np.vstack(event_position_batches)
            current_event_rank_array = _rank_rows(current_event_positions)
            current_event_rank_mean = current_event_rank_array.mean(axis=0)
            current_pairwise_precedence = None

            event_rank_max_change = np.nan
            if previous_event_rank_mean is not None:
                event_rank_max_change = float(
                    np.max(np.abs(current_event_rank_mean - previous_event_rank_mean))
                )

            pairwise_precedence_max_change = np.nan
            if pairwise_precedence_convergence_tol is not None:
                current_pairwise_precedence = _pairwise_precedence_array_from_positions(current_event_positions)
                if previous_pairwise_precedence is not None:
                    pairwise_precedence_max_change = float(
                        np.max(np.abs(current_pairwise_precedence - previous_pairwise_precedence))
                    )

            checks = []
            if event_rank_convergence_tol is not None:
                checks.append(
                    previous_event_rank_mean is not None
                    and event_rank_max_change <= event_rank_convergence_tol
                )
            if pairwise_precedence_convergence_tol is not None:
                checks.append(
                    previous_pairwise_precedence is not None
                    and pairwise_precedence_max_change <= pairwise_precedence_convergence_tol
                )

            convergence_passed = bool(checks) and all(checks)
            if convergence_passed:
                stable_convergence_checks += 1
            else:
                stable_convergence_checks = 0

            convergence_records.append(
                {
                    "n_samples": samples_completed,
                    "event_rank_max_change": event_rank_max_change,
                    "pairwise_precedence_max_change": pairwise_precedence_max_change,
                    "stable_checks": stable_convergence_checks,
                    "converged": stable_convergence_checks >= convergence_patience,
                }
            )

            previous_event_rank_mean = current_event_rank_mean
            if current_pairwise_precedence is not None:
                previous_pairwise_precedence = current_pairwise_precedence

            if stable_convergence_checks >= convergence_patience:
                adaptive_converged = True
                break

    calculation_time_seconds = perf_counter() - calculation_start_time

    # Convert numerical arrays into labeled pandas objects after the timed core
    # calculation so users can inspect matrices, samples, and summaries directly.
    ordered_subjects = _ordered_subjects_from_dps(feature_data=feature_data, dps=base_dps)
    base_transition_matrix = pd.DataFrame(
        base_transition,
        index=subject_index,
        columns=subject_index,
    )
    stationary_distribution = pd.Series(
        stationary,
        index=subject_index,
        name="stationary_probability",
    )
    mean_first_passage_matrix = pd.DataFrame(
        mean_first_passage,
        index=subject_index,
        columns=subject_index,
    )
    first_passage_variance_matrix = pd.DataFrame(
        first_passage_variance,
        index=subject_index,
        columns=subject_index,
    )
    subject_variance = pd.DataFrame(
        {
            "first_passage_variance": subject_first_passage_variance,
            "dps_noise_variance": subject_noise_variance,
            "dps_noise_std": np.sqrt(subject_noise_variance),
            "stationary_probability": stationary,
            "raw_dps_range": raw_base_dps_range,
            "effective_dps_noise_scale": effective_dps_noise_scale,
            "automatic_dps_local_window": automatic_dps_local_window,
        },
        index=subject_index,
    )
    metadata_records = [
        {
            "run_id": run_id,
            "sample_id": sample_id,
            "uncertainty_model": "kemeny_snell_first_passage_dps_noise",
            "ordering_method": ordering_method,
            "n_neighbors": n_neighbors,
            "support_n_neighbors": support_n_neighbors,
            "n_noise_samples_requested": n_noise_samples,
            "n_noise_samples_actual": len(run_ids),
            "dps_noise_scale": dps_noise_scale,
            "effective_dps_noise_scale": effective_dps_noise_scale,
            "automatic_dps_local_window": automatic_dps_local_window,
            "automatic_dps_noise_scale": dps_noise_scale is None and normalize_subject_variance,
            "normalize_subject_variance": normalize_subject_variance,
            "raw_dps_range": raw_base_dps_range,
            "variance_scale": (
                "median_subject_variance_times_local_dps_spacing"
                if normalize_subject_variance
                else "raw_dps_range_squared"
            ),
            "subject_variance_aggregation": "stationary_weighted_first_passage",
            "adaptive_convergence_enabled": convergence_enabled,
            "adaptive_converged": adaptive_converged,
            "convergence_check_interval": (
                effective_convergence_check_interval if convergence_enabled else None
            ),
            "convergence_min_samples": (
                effective_convergence_min_samples if convergence_enabled else None
            ),
            "event_rank_convergence_tol": event_rank_convergence_tol,
            "pairwise_precedence_convergence_tol": pairwise_precedence_convergence_tol,
            "convergence_patience": convergence_patience if convergence_enabled else None,
            "mean_abs_dps_noise": mean_abs_dps_noise[sample_id],
            "max_abs_dps_noise": max_abs_dps_noise[sample_id],
        }
        for sample_id, run_id in enumerate(run_ids)
    ]
    subject_dps_array = np.vstack(subject_dps_batches)
    subject_rank_array = np.vstack(subject_rank_batches)
    event_position_array = np.vstack(event_position_batches)
    subject_dps_samples = pd.DataFrame(
        subject_dps_array.T,
        index=subject_index,
        columns=run_ids,
    )
    subject_rank_samples = pd.DataFrame(
        subject_rank_array.T,
        index=subject_index,
        columns=run_ids,
    )
    event_position_samples = pd.DataFrame(
        event_position_array,
        index=run_ids,
        columns=feature_names,
    )
    event_rank_samples = pd.DataFrame(
        _rank_rows(event_position_array),
        index=run_ids,
        columns=feature_names,
    )
    pairwise_precedence = pd.DataFrame(
        _pairwise_precedence_array_from_positions(event_position_array),
        index=feature_names,
        columns=feature_names,
    )
    sample_metadata = pd.DataFrame(metadata_records).set_index("run_id")
    convergence_history = (
        pd.DataFrame(convergence_records)
        if convergence_records
        else None
    )

    subject_dps_summary = _summarize_samples(subject_dps_samples, prefix="dps")
    subject_rank_summary = _summarize_samples(subject_rank_samples, prefix="rank")
    event_summary = _summarize_samples(event_rank_samples.T, prefix="rank").join(
        _summarize_samples(event_position_samples.T, prefix="position")
    )

    subject_precedence = None
    if return_subject_precedence:
        subject_precedence = _subject_precedence_from_ranks(subject_rank_samples)

    return FastEBMFirstPassageUncertaintyResult(
        ordered_subjects=ordered_subjects,
        base_transition_matrix=base_transition_matrix,
        stationary_distribution=stationary_distribution,
        mean_first_passage_matrix=mean_first_passage_matrix,
        first_passage_variance_matrix=first_passage_variance_matrix,
        subject_variance=subject_variance,
        sample_metadata=sample_metadata,
        subject_dps_samples=subject_dps_samples,
        subject_dps_summary=subject_dps_summary,
        subject_rank_samples=subject_rank_samples,
        subject_rank_summary=subject_rank_summary,
        event_position_samples=event_position_samples,
        event_rank_samples=event_rank_samples,
        event_summary=event_summary,
        pairwise_precedence=pairwise_precedence,
        subject_precedence=subject_precedence,
        calculation_time_seconds=calculation_time_seconds,
        convergence_history=convergence_history,
    )

