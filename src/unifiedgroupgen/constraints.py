"""Common affine observation fibers and metric-aware tangent projections."""

import torch


def tangent_project(velocity, jacobian, metric=None):
    """Project into ker(J) in metric M; supports redundant constraints via pinv."""
    if jacobian.numel() == 0:
        return velocity
    minv_jt = jacobian.T if metric is None else torch.linalg.solve(metric, jacobian.T)
    correction = minv_jt @ torch.linalg.pinv(jacobian @ minv_jt) @ (jacobian @ velocity)
    return velocity - correction


class AffineObservation:
    """Az=b in a fixed lifted chart. This is not a general nonlinear constraint solver."""
    def __init__(self, matrix, values):
        self.matrix, self.values = torch.as_tensor(matrix), torch.as_tensor(values)
        self._fixed_indices = None

    @classmethod
    def fixed(cls, dimension, indices, values):
        if len(indices) != len(set(indices)) or len(indices) != len(values):
            raise ValueError("Fixed indices must be unique and match the value count")
        a = torch.zeros(len(indices), dimension, dtype=torch.float64)
        for row, index in enumerate(indices):
            if not 0 <= index < dimension:
                raise ValueError("Observed index is outside the compiled state")
            a[row, index] = 1
        observation = cls(a, torch.as_tensor(values, dtype=torch.float64))
        observation._fixed_indices = torch.tensor(indices, dtype=torch.long)
        return observation

    def _tensors(self, z):
        return self.matrix.to(z), self.values.to(z)

    def retract(self, z):
        if self._fixed_indices is not None:
            return z.index_copy(0, self._fixed_indices.to(z.device), self.values.to(z))
        a, b = self._tensors(z)
        result = z + torch.linalg.pinv(a) @ (b - a @ z)
        if torch.linalg.norm(a @ result - b) > 1e-5:
            raise ValueError("Affine observations are inconsistent")
        return result

    def project(self, velocity):
        if self._fixed_indices is not None:
            return velocity.index_fill(0, self._fixed_indices.to(velocity.device), 0)
        a, _ = self._tensors(velocity)
        return tangent_project(velocity, a)

    def features(self, z):
        if self._fixed_indices is not None:
            indices = self._fixed_indices.to(z.device)
            base = torch.zeros_like(z).index_copy(0, indices, self.values.to(z))
            return base, torch.zeros_like(z).index_fill(0, indices, 1)
        a, b = self._tensors(z)
        base = torch.linalg.pinv(a) @ b
        projector = torch.linalg.pinv(a) @ a
        return base, projector.diag()
