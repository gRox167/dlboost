from deepinv.optim.prior import Prior


class PnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)`.


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

    def prox(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float sigma_denoiser: noise level parameter of the denoiser.
        :return: (torch.tensor) proximity operator at :math:`x`.
        """
        # return x
        if not self.sigma_flag:
            sigma_denoiser = 1.0
        if self.unsqueeze_channel_dim:
            return self.denoiser(x.unsqueeze(1), sigma_denoiser).squeeze(1)
        else:
            return self.denoiser(x, sigma_denoiser)
