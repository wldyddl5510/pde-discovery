"""Estimators of the solution u from noisy point observations."""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree


def estimate_grid_noise_std(observations):
    """Estimate iid noise via the normalized sixth temporal difference."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or observations.shape[1] < 7:
        raise ValueError("observations must be a spatial-by-time grid with at least 7 times.")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite.")
    weights = np.array([1, -6, 15, -20, 15, -6, 1], dtype=float)
    differences = np.diff(observations, n=6, axis=1)
    return float(np.sqrt(np.mean(differences**2)) / np.linalg.norm(weights))


def paper_filter_width(noise_std_estimate, support_points, spatial_dim=1):
    """Per-axis moving-average width from Messenger and Bortz (2022), (5.6)."""
    if not np.isfinite(noise_std_estimate) or noise_std_estimate < 0:
        raise ValueError("noise_std_estimate must be finite and nonnegative.")
    if not isinstance(support_points, (int, np.integer)) or support_points < 1:
        raise ValueError("support_points must be a positive integer.")
    if not isinstance(spatial_dim, (int, np.integer)) or spatial_dim < 1:
        raise ValueError("spatial_dim must be a positive integer.")
    dimension = spatial_dim + 1
    desired = 2.0 * (1500.0 * noise_std_estimate**2)**(1.0 / dimension)
    maximum = support_points**(1.0 / dimension) / 2.0
    return max(1, int(np.ceil(min(desired, maximum))))


def filter_grid_moving_average(observations, width):
    """Apply a box average with periodic space and reflected time boundaries."""
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width must be a positive integer.")
    return uniform_filter(
        np.asarray(observations, dtype=float), size=width, mode=("wrap", "reflect")
    )


class MovingAverageEstimator:
    """Average observations in a box around each requested space-time point.

    ``points`` has shape (number of observations, number of coordinates), with
    time as the last coordinate. ``bandwidths`` gives the box radius in the
    same physical units and coordinate order. The estimator only uses the
    observations passed to ``fit``, so callers can fit on one data split and
    predict on another.

    This is a box-kernel version of the moving average used by Messenger and
    Bortz (2022), arXiv:2211.16000. Their numerical filter is on a regular
    grid. Here the training coordinates may instead be irregular; each
    average uses the training points available in the box.
    """

    def __init__(self, bandwidths):
        widths = np.asarray(bandwidths, dtype=float)
        if widths.ndim != 1 or widths.size == 0:
            raise ValueError("bandwidths must have one entry per coordinate.")
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            raise ValueError("bandwidths must be finite and positive.")

        self.bandwidths = widths.copy()
        self._tree = None
        self._values = None

    def fit(self, points, observations):
        """Store training observations and return this fitted estimator."""
        points = np.asarray(points, dtype=float)
        observations = np.asarray(observations, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(self.bandwidths):
            raise ValueError("points must have one column per bandwidth.")
        if points.shape[0] == 0 or observations.shape != (len(points),):
            raise ValueError("observations must have one value per training point.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(observations)):
            raise ValueError("training points and observations must be finite.")

        # A unit-radius Chebyshev ball in scaled coordinates is the desired box.
        self._tree = cKDTree(points / self.bandwidths)
        self._values = observations.copy()
        return self

    def predict(self, points):
        """Return local means at query points; reject queries with no neighbors."""
        if self._tree is None:
            raise RuntimeError("Fit the estimator before calling predict.")

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(self.bandwidths):
            raise ValueError("points must have one column per bandwidth.")
        if not np.all(np.isfinite(points)):
            raise ValueError("query points must be finite.")

        estimates = np.empty(len(points), dtype=float)
        for start in range(0, len(points), 4096):
            query = points[start:start + 4096] / self.bandwidths
            neighborhoods = self._tree.query_ball_point(query, r=1.0, p=np.inf)
            for offset, neighbors in enumerate(neighborhoods):
                if not neighbors:
                    raise ValueError(
                        "A query point has no training observations in its moving-average "
                        "window; increase bandwidths."
                    )
                estimates[start + offset] = np.mean(self._values[neighbors])

        return estimates
