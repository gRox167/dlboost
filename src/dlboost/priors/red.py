from deepinv.optim.prior import Prior


class RED(Prior):
    r"""
    Regularization-by-Denoising (RED) prior :math:`\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)`.


    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(
        self, denoiser, unsqueeze_channel_dim=True, sigma_flag=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False
        self.unsqueeze_channel_dim = unsqueeze_channel_dim
        self.sigma_flag = sigma_flag

    def grad(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Calculates the gradient of the prior term :math:`\regname` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x g`, computed in :math:`x`.
        """
        if not self.sigma_flag:
            sigma_denoiser = 1.0
        if self.unsqueeze_channel_dim:
            return x - self.denoiser(x.unsqueeze(1), sigma_denoiser).squeeze(1)
        else:
            return x - self.denoiser(x, sigma_denoiser)
