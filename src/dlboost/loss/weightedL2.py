import torch
from deepinv.optim.distance import Distance


class WeightedL2Distance(Distance):
    r"""
    Implementation of :math:`\distancename` as the normalized :math:`\ell_2` norm

    .. math::
        f(x) = \frac{1}{2\sigma^2}\|x-y\|^2

    :param float sigma: normalization parameter. Default: 1.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.norm = 1 / (sigma**2)
        self.weight = None

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the distance :math:`\distance{x}{y}` i.e.

        .. math::

            \distance{x}{y} = \frac{1}{2}\|x-y\|^2


        :param torch.Tensor u: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{u}{y}` of size `B` with `B` the size of the batch.
        """
        z = x - y
        if self.weight is not None:
            z = z * self.weight
        d = 0.5 * torch.norm(z.reshape(z.shape[0], -1), p=2, dim=-1) ** 2 * self.norm
        return d

    def grad(self, x, y, *args, **kwargs):
        r"""
        Computes the gradient of :math:`\distancename`, that is  :math:`\nabla_{x}\distance{x}{y}`, i.e.

        .. math::

            \nabla_{x}\distance{x}{y} = \frac{1}{\sigma^2} x-y


        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (:class:`torch.Tensor`) gradient of the distance function :math:`\nabla_{x}\distance{x}{y}`.
        """
        return (x - y) * self.weight**2 * self.norm

    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of :math:`\gamma \distance{x}{y} = \frac{\gamma}{2 \sigma^2} \|x-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \distancename}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \distancename} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|u-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float gamma: thresholding parameter.
        :return: (:class:`torch.Tensor`) proximity operator :math:`\operatorname{prox}_{\gamma \distancename}(x)`.
        """
        return (x + self.norm * gamma * y) / (1 + gamma * self.norm)
