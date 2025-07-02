import torch
from dlboost.operators.MRI import CSM


def test_csm_function_gradients():
    b, ch, d, h, w = 1, 2, 2, 3, 3
    x = torch.randn(b, d, h, w, dtype=torch.complex64, requires_grad=True)
    csm = torch.randn(b, ch, d, h, w, dtype=torch.complex64, requires_grad=True)

    op = CSM(scale_factor=None)
    op.update_parameters(csm.clone().detach().requires_grad_(True))

    y = op.A(x)
    loss = (y.abs() ** 2).sum()
    loss.backward()

    grad_x = x.grad.clone()
    grad_csm = op._csm.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    csm2 = csm.detach().clone().requires_grad_(True)
    y_ref = x2.unsqueeze(1) * csm2
    loss_ref = (y_ref.abs() ** 2).sum()
    loss_ref.backward()

    assert torch.allclose(x2.grad, grad_x, atol=1e-5)
    assert torch.allclose(csm2.grad, grad_csm, atol=1e-5)
