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

    @classmethod
    def fixed(cls, dimension, indices, values):
        if len(indices) != len(set(indices)) or len(indices) != len(values):
            raise ValueError("Fixed indices must be unique and match the value count")
        a = torch.zeros(len(indices), dimension, dtype=torch.float64)
        for row, index in enumerate(indices):
            if not 0 <= index < dimension:
                raise ValueError("Observed index is outside the compiled state")
            a[row, index] = 1
        return cls(a, torch.as_tensor(values, dtype=torch.float64))

    def _tensors(self, z):
        return self.matrix.to(z), self.values.to(z)

    def retract(self, z):
        a, b = self._tensors(z)
        result = z + torch.linalg.pinv(a) @ (b - a @ z)
        if torch.linalg.norm(a @ result - b) > 1e-5:
            raise ValueError("Affine observations are inconsistent")
        return result

    def project(self, velocity):
        a, _ = self._tensors(velocity)
        return tangent_project(velocity, a)

    def features(self, z):
        a, b = self._tensors(z)
        base = torch.linalg.pinv(a) @ b
        projector = torch.linalg.pinv(a) @ a
        return base, projector.diag()
