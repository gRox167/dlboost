from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.distance import (
    L2Distance,
)


class L2(DataFidelity):
    r"""
    Implementation of the data-fidelity as the normalized :math:`\ell_2` norm

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|\forw{x}-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.


    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>> # define a loss function
        >>> fidelity = dinv.optim.data_fidelity.L2()
        >>>
        >>> x = torch.ones(1, 1, 3, 3)
        >>> mask = torch.ones_like(x)
        >>> mask[0, 0, 1, 1] = 0
        >>> physics = dinv.physics.Inpainting(tensor_size=(1, 3, 3), mask=mask)
        >>> y = physics(x)
        >>>
        >>> # Compute the data fidelity f(Ax, y)
        >>> fidelity(x, y, physics)
        tensor([0.])
        >>> # Compute the gradient of f
        >>> fidelity.grad(x, y, physics)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
        >>> # Compute the proximity operator of f
        >>> fidelity.prox(x, y, physics, gamma=1.0)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.d = L2Distance(sigma=sigma)
        self.norm = 1 / (sigma**2)

    def prox(self, x, y, physics, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of :math:`\gamma \datafid{Ax}{y} = \frac{\gamma}{2\sigma^2}\|Ax-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \datafidname}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafidname} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Au-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator :math:`\operatorname{prox}_{\gamma \datafidname}(x)`.
        """
        return physics.prox_l2(x, y, self.norm * gamma)

    def grad(self, x, y, physics, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{\forw{x}}{y} = \left. \frac{\partial A}{\partial x} \right|_x^\top \nabla_u \distance{u}{y},

        where :math:`\left. \frac{\partial A}{\partial x} \right|_x` is the Jacobian of :math:`A` at :math:`x`, and :math:`\nabla_u \distance{u}{y}` is computed using ``grad_d`` with :math:`u = \forw{x}`. The multiplication is computed using the ``A_vjp`` method of the physics.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.grad_l2(x, y, self.norm, *args, **kwargs)
        # return physics.A_vjp(x, self.d.grad(physics.A(x), y, *args, **kwargs))
