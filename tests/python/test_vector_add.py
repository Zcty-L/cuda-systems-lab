import pytest
import torch

import cuda_lab

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")]


def test_vector_add():
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    actual = cuda_lab.vector_add(a, b)
    torch.testing.assert_close(actual, a + b, rtol=0, atol=0)
