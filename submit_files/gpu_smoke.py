"""Check the TH-HPC6 RTX 5090 runtime without starting project training."""
import sys

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_spline_conv import spline_basis


def main():
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("Memory GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    x = torch.randn(32, 32, device=device, requires_grad=True)
    (x @ x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    print("PASS: PyTorch FP32 forward/backward")

    assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
    x = torch.randn(32, 32, device=device, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = x @ x
    assert y.dtype == torch.bfloat16
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    print("PASS: PyTorch BF16 forward/backward")

    src = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1], device=device)
    out = scatter(src, index, dim=0, reduce="sum")
    torch.testing.assert_close(out, torch.tensor([3., 7.], device=device))
    out.sum().backward()
    torch.testing.assert_close(src.grad, torch.ones_like(src))
    print("PASS: torch-scatter GPU forward/backward")

    idx = torch.arange(2, device=device)
    values = torch.tensor([2., 3.], device=device, requires_grad=True)
    matrix = SparseTensor(row=idx, col=idx, value=values, sparse_sizes=(2, 2))
    rhs = torch.ones(2, 2, device=device, requires_grad=True)
    out = matrix.matmul(rhs)
    expected = torch.tensor([[2., 2.], [3., 3.]], device=device)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([2., 2.], device=device))
    torch.testing.assert_close(rhs.grad, expected)
    print("PASS: torch-sparse GPU forward/backward")

    points = torch.tensor([[0., 0.], [0.5, 0.], [3., 0.]], device=device)
    edges = radius_graph(points, r=1.0, loop=False)
    assert edges.is_cuda
    assert set(map(tuple, edges.t().cpu().tolist())) == {(0, 1), (1, 0)}
    print("PASS: torch-cluster GPU neighbor search")

    pseudo = torch.tensor([[0.2, 0.3], [0.6, 0.7]], device=device)
    kernel = torch.tensor([3, 3], dtype=torch.long, device=device)
    opened = torch.tensor([1, 1], dtype=torch.uint8, device=device)
    basis, weights = spline_basis(pseudo, kernel, opened, 1)
    assert basis.is_cuda and weights.is_cuda
    assert torch.isfinite(basis).all()
    torch.testing.assert_close(basis.sum(dim=1), torch.ones(2, device=device))
    print("PASS: torch-spline-conv GPU forward")

    torch.cuda.synchronize()
    print("ALL GPU CHECKS PASSED")


if __name__ == "__main__":
    main()
