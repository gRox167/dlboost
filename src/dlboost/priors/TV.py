import torch
from deepinv.models.tv import TVDenoiser
from deepinv.optim.prior import TVPrior


class TVDenoiser3D(TVDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y, ths=None, **kwargs):
        r"""
        Computes the proximity operator of the TV norm.

        :param torch.Tensor y: Noisy image.
        :param float, torch.Tensor ths: Regularization parameter :math:`\gamma`.
        :return: Denoised image.
        """

        restart = (
            True
            if (
                self.restart
                or self.x2 is None
                or self.u2 is None
                or self.x2.shape != y.shape
            )
            else False
        )

        if restart:
            x2 = y.clone()
            u2 = torch.zeros((*y.shape, 3), device=y.device).type(y.dtype)
            self.restart = False
        else:
            x2 = self.x2.clone()
            u2 = self.u2.clone()

        if ths is not None:
            lambd = ths

        for _ in range(self.n_it_max):
            x_prev = x2

            x = self.prox_tau_fx(x2 - self.tau * self.nabla_adjoint(u2), y)
            u = self.prox_sigma_g_conj(u2 + self.sigma * self.nabla(2 * x - x2), lambd)
            x2 = x2 + self.rho * (x - x2)
            u2 = u2 + self.rho * (u - u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - x2.flatten()
            ) / torch.linalg.norm(x2.flatten() + 1e-12)

            if _ > 1 and rel_err < self.crit:
                if self.verbose:
                    print("TV prox reached convergence")
                break

        self.x2 = x2.detach()
        self.u2 = u2.detach()

        return x2

    @staticmethod
    def nabla(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, d, h, w = x.shape
        u = torch.zeros(b, c, d, h, w, 3, device=x.device, dtype=x.dtype)
        u[:, :, :-1, :, :, 0] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        u[:, :, :, :-1, :, 1] = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        u[:, :, :, :, :-1, 2] = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return u

    @staticmethod
    def nabla_adjoint(x):
        r"""
        Applies the transpose of the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, d, h, w, _ = x.shape
        u = torch.zeros(b, c, d, h, w, device=x.device, dtype=x.dtype)
        u[:, :, 1:, :, :] += x[:, :, :-1, :, :, 0]
        u[:, :, :-1, :, :] -= x[:, :, :-1, :, :, 0]
        u[:, :, :, 1:, :] += x[:, :, :, :-1, :, 1]
        u[:, :, :, :-1, :] -= x[:, :, :, :-1, :, 1]
        u[:, :, :, :, 1:] += x[:, :, :, :, :-1, 2]
        u[:, :, :, :, :-1] -= x[:, :, :, :, :-1, 2]
        return u


class TVPrior3D(TVPrior):
    def __init__(self, def_crit=1e-8, n_it_max=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.TVModel = TVDenoiser3D(crit=def_crit, n_it_max=n_it_max)
