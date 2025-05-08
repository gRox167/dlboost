import torch

# from deepinv.loss import Loss
from deepinv.loss.loss import Loss
from torch import Tensor


def hutch_div(y, physics, f, mc_iter=1, rng=None):
    r"""
    Hutch divergence for A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :param torch.Generator rng: Random number generator. Default is None.
    :return: (float) hutch divergence.
    """
    input = y.requires_grad_(True)
    output = physics.A(f(input, physics))
    out = 0
    for i in range(mc_iter):
        b = torch.empty_like(y).normal_(generator=rng)
        x = torch.autograd.grad(output, input, b, retain_graph=True, create_graph=True)[
            0
        ]
        out += (b * x).reshape(y.size(0), -1).mean(1)

    return out / mc_iter


def exact_div(y, physics, model):
    r"""
    Exact divergence for A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module model: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :return: (float) exact divergence.
    """
    input = y.requires_grad_(True)
    output = physics.A(model(input, physics))
    out = 0
    _, c, h, w = input.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                b = torch.zeros_like(input)
                b[:, i, j, k] = 1
                x = torch.autograd.grad(
                    output, input, b, retain_graph=True, create_graph=True
                )[0]
                out += (b * x).sum()

    return out / (c * h * w)


def mc_div(y1, y, f, physics, tau, precond=lambda x: x, rng: torch.Generator = None):
    r"""
    Monte-Carlo estimation for the divergence of A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param bool pinv: If ``True``, the pseudo-inverse of the forward operator is used. Default ``False``.
    :param Callable precond: Preconditioner. Default is the identity.
    :param torch.Generator rng: Random number generator. Default is None.
    :return: (float) Ramani MC divergence.
    """
    b = torch.empty_like(y).normal_(generator=rng)
    y2 = physics.A(f(y + b * tau, physics))
    return (precond(b) * precond(y2 - y1) / tau).reshape(y.size(0), -1).mean(1)


def unsure_gradient_step(loss, param, saved_grad, init_flag, step_size, momentum):
    r"""
    Gradient step for estimating the noise level in the UNSURE loss.

    :param torch.Tensor loss: Loss value.
    :param torch.Tensor param: Parameter to optimize.
    :param torch.Tensor saved_grad: Saved gradient w.r.t. the parameter.
    :param bool init_flag: Initialization flag (first gradient step).
    :param float step_size: Step size.
    :param float momentum: Momentum.
    """
    grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
    if init_flag:
        init_flag = False
        saved_grad = grad
    else:
        saved_grad = momentum * saved_grad + (1.0 - momentum) * grad
    return param + step_size * grad, saved_grad, init_flag


class SureGaussianLoss(Loss):
    r"""
    SURE loss for Gaussian noise


    The loss is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|B(y - A\inverse{y})\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} B^{\top} \left(A\inverse{y+\tau b_i} -
        A\inverse{y}\right)

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`A` is the forward operator,
    :math:`B` is an optional linear mapping which should be approximately :math:`A^{\dagger}` (or any stable approximation),
    :math:`b\sim\mathcal{N}(0,I)` and :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence.

    This loss approximates the divergence of :math:`A\inverse{y}` (in the original SURE loss)
    using the Monte Carlo approximation in
    https://ieeexplore.ieee.org/abstract/document/4099398/

    If the measurement data is truly Gaussian with standard deviation :math:`\sigma`,
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-A\inverse{y}\|_2^2`
    where :math:`z` is the noiseless measurement.

    .. warning::

        The loss can be sensitive to the choice of :math:`\tau`, which should be proportional to the size of :math:`y`.
        The default value of 0.01 is adapted to :math:`y` vectors with entries in :math:`[0,1]`.

    .. note::

        If the noise level is unknown, the loss can be adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985,
        which also learns the noise level.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param Callable, str B: Optional linear metric :math:`B`, which can be used to improve
        the performance of the loss. If 'A_dagger', the pseudo-inverse of the forward operator is used.
        Otherwise the metric should be a linear operator that approximates the pseudo-inverse of the forward operator
        such as :func:`deepinv.physics.LinearPhysics.prox_l2` with large :math:`\gamma`. By default, the identity is used.
    :param bool unsure: If ``True``, the loss is adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985
        where the noise level :math:`\sigma` is also learned (the input value is used as initialization).
    :param float step_size: Step size for the gradient ascent of the noise level if unsure is ``True``.
    :param float momentum: Momentum for the gradient ascent of the noise level if unsure is ``True``.
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(
        self,
        sigma=1e-1,
        tau=1e-2,
        B=lambda x: x,
        unsure=False,
        step_size=1e-4,
        momentum=0.9,
        rng: torch.Generator = None,
    ):
        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = sigma**2
        self.tau = tau
        self.metric = B
        self.unsure = unsure
        self.init_flag = False
        self.step_size = step_size
        self.momentum = momentum
        self.grad_sigma = 0.0
        self.rng = rng
        if unsure:
            self.sigma2 = torch.tensor(self.sigma2, requires_grad=True)

    def forward(
        self, x_net: Tensor, x: Tensor | None, y: Tensor, physics, model, **kwargs
    ):
        r"""
        Computes the SURE Loss.

        :param torch.Tensor y: Measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction network.
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        if self.metric == "A_dagger":
            metric = lambda x: physics.A_dagger(x)
        else:
            metric = self.metric

        y1 = physics.A(x_net)
        div = (
            2 * self.sigma2 * mc_div(y1, y, model, physics, self.tau, metric, self.rng)
        )
        mse = metric(y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        loss_sure = mse + div - self.sigma2

        if self.unsure:  # update the estimate of the noise level
            self.sigma2, self.grad_sigma, self.init_flag = unsure_gradient_step(
                div.mean(),
                self.sigma2,
                self.grad_sigma,
                self.init_flag,
                self.step_size,
                self.momentum,
            )

        return loss_sure
