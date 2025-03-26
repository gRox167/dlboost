import torch
from deepinv.optim.prior import Prior
from deepinv.utils import TensorList


class SeparablePrior(Prior):
    """
    Prior with separable structure along a specific axis of the input tensor.
    Allows to extend the definition of a prior :math:`g` into a separable prior
    .. math::
            f(x) = \sum_i w_i g(x_i)
    where :math:`x_i` is a slice of :math:`x` taken along the separable axis and :math:`w=(w_1,\dots, w_I)` is a tensor of weights.
    The proximity operator of such an :math:`f` can be computed slice-by-slice and is the concatenation of :math:`\operatorname{prox}_{w_i g}(x_i)` along the separable axis.
    The separable weights (given in log-domain) are exponentiated to ensure positivity and scale the contributions of each slice.
    Expected input:
      - x: a tensor of shape [A, B, ..., I, ...] where the I-axis (indexed by separable_axis)
           contains the separable components.

    """

    def __init__(self, prior, separable_axis, separable_weights, *args, **kwargs):
        """
        :param dinv.optim.Prior prior: a Prior defining the function :math:`g`
        :param int separable_axis: index of the axis over which the prior is separable.
        :param torch.Tensor separable_weights: a tensor of weights (in log-domain) for each slice along the separable axis.
        """
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.separable_axis = separable_axis
        self.separable_weights = separable_weights
        self.explicit_prior = (
            prior[0].explicit_prior if isinstance(prior, list) else prior.explicit_prior
        )

    def fn(self, x, *args, **kwargs):
        """
        Compute the function value :math:`f(x)` as the weighted sum over slices.
        For each coordinate along the separable_axis, a slice is taken and the base prior function
        is applied. Each contribution is weighted by exp(separable_weights[coord]).
        :param torch.Tensor x: Input tensor.
        :return torch.Tensor : value of :math:`f(x)` for each batch
        """
        # Exponentiate the (log-)weights.
        eseparable_weights = self.separable_weights
        # Initialize f_total with zeros. Assumes the first dimension is the batch dimension.
        f_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # Loop over the separable axis
        for coord, sliced_x in enumerate(torch.split(x, 1, dim=self.separable_axis)):
            # Add the weighted contribution of the prior applied to the slice.
            f_total = f_total + eseparable_weights[coord] * self.prior.fn(
                sliced_x, *args, **kwargs
            )
        return f_total

    def prox(self, x, *args, gamma, **kwargs):
        """
        Compute the proximity operator associated with :math:`f`.
        Compute the proximity operator associated with :math:`f`.
        The prox is computed slice-by-slice along the separable_axis. For each slice:
        .. math::
            \operatorname{prox}_{gamma * w * g}(x_slice)
        is computed, and then the resulting slices are concatenated back along the separable_axis.
        :param x: Input tensor.
        :param gamma: A step-size parameter (on :math:`\tau f`).
        :return torch.Tensor: :math:`\operatorname{prox}_{\tau f}(x)` of the same shape as :math:`x` after applying the proximity operator.
        """
        results = []
        slices = torch.split(x, 1, dim=self.separable_axis)
        # from matplotlib import pyplot as plt

        # slice_idx = 120
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # axes[0].imshow(
        #     slices[0, 0, 0].cpu()[slice_idx],
        #     vmin=-0.4,
        #     vmax=0.4,
        #     cmap="gray",
        # )
        # axes[0].set_title("slope")
        # axes[0].axis("off")
        # axes[1].imshow(
        #     slices[1, 0, 0].cpu()[slice_idx],
        #     vmin=-0.4,
        #     vmax=0.4,
        #     cmap="gray",
        # )
        # axes[1].set_title("bias 0")
        # axes[1].axis("off")
        # axes[2].imshow(
        #     slices[2, 0, 0].cpu()[slice_idx],
        #     vmin=-0.4,
        #     vmax=0.4,
        #     cmap="gray",
        # )
        # axes[2].set_title("bias 1")
        # axes[2].axis("off")
        for i, sliced_x in enumerate(slices):
            w = self.separable_weights[i]
            result = self.prior.prox(sliced_x, *args, gamma=gamma * w, **kwargs)
            results.append(result)
        return torch.cat(results, dim=self.separable_axis)

    def forward(self, x, *args, **kwargs):
        """
        :return torch.Tensor: The value of :math:`f(x) = \sum_i w_i g(x_i)`
        """
        return self.fn(x, *args, **kwargs)


class SeparablePrior(Prior):
    """
    Prior with separable structure along a specific axis of the input tensor.
    Allows to extend the definition of a prior :math:`g` into a separable prior
    .. math::
            f(x) = \sum_i w_i g(x_i)
    where :math:`x_i` is a slice of :math:`x` taken along the separable axis and :math:`w=(w_1,\dots, w_I)` is a tensor of weights.
    The proximity operator of such an :math:`f` can be computed slice-by-slice and is the concatenation of :math:`\operatorname{prox}_{w_i g}(x_i)` along the separable axis.
    The separable weights (given in log-domain) are exponentiated to ensure positivity and scale the contributions of each slice.
    Expected input:
      - x: a tensor of shape [A, B, ..., I, ...] where the I-axis (indexed by separable_axis)
           contains the separable components.

    """

    def __init__(self, prior_list, separable_axis, separable_weights, *args, **kwargs):
        """
        :param dinv.optim.Prior prior: a Prior defining the function :math:`g`
        :param int separable_axis: index of the axis over which the prior is separable.
        :param torch.Tensor separable_weights: a tensor of weights (in log-domain) for each slice along the separable axis.
        """
        super().__init__(*args, **kwargs)
        self.prior_list = prior_list
        self.separable_axis = separable_axis
        self.separable_weights = separable_weights
        self.explicit_prior = (
            prior_list[0].explicit_prior
            if isinstance(prior_list, list)
            else prior_list.explicit_prior
        )

    def fn(self, x, *args, **kwargs):
        """
        Compute the function value :math:`f(x)` as the weighted sum over slices.
        For each coordinate along the separable_axis, a slice is taken and the base prior function
        is applied. Each contribution is weighted by exp(separable_weights[coord]).
        :param torch.Tensor x: Input tensor.
        :return torch.Tensor : value of :math:`f(x)` for each batch
        """
        # Exponentiate the (log-)weights.
        eseparable_weights = self.separable_weights
        # Initialize f_total with zeros. Assumes the first dimension is the batch dimension.
        f_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # Loop over the separable axis
        for coord, sliced_x in enumerate(torch.split(x, 1, dim=self.separable_axis)):
            # Add the weighted contribution of the prior applied to the slice.
            f_total = f_total + eseparable_weights[coord] * self.prior_list[coord].fn(
                sliced_x, *args, **kwargs
            )
        return f_total

    def prox(self, x, *args, gamma, **kwargs):
        """
        Compute the proximity operator associated with :math:`f`.
        Compute the proximity operator associated with :math:`f`.
        The prox is computed slice-by-slice along the separable_axis. For each slice:
        .. math::
            \operatorname{prox}_{gamma * w * g}(x_slice)
        is computed, and then the resulting slices are concatenated back along the separable_axis.
        :param x: Input tensor.
        :param gamma: A step-size parameter (on :math:`\tau f`).
        :return torch.Tensor: :math:`\operatorname{prox}_{\tau f}(x)` of the same shape as :math:`x` after applying the proximity operator.
        """
        results = []
        slices = torch.split(x, 1, dim=self.separable_axis)

        for i, sliced_x in enumerate(slices):
            w = self.separable_weights[i]
            result = self.prior_list[i].prox(sliced_x, *args, gamma=gamma * w, **kwargs)
            results.append(result)
        return torch.cat(results, dim=self.separable_axis)

    def forward(self, x, *args, **kwargs):
        """
        :return torch.Tensor: The value of :math:`f(x) = \sum_i w_i g(x_i)`
        """
        return self.fn(x, *args, **kwargs)


class ListSeparablePrior(Prior):
    """
    Prior with separable structure along a specific axis of the input tensor.
    Allows to extend the definition of a prior :math:`g` into a separable prior
    .. math::
            f(x) = \sum_i w_i g(x_i)
    where :math:`x=[x_1,\dots, x_I]` is a dinv.utils.TensorList and :math:`w=(w_1,\dots, w_I)` is a tensor of weights.
    The proximity operator of such an :math:`f` can be computed slice-by-slice and is the concatenation of :math:`\operatorname{prox}_{w_i g}(x_i)` along the separable axis.
    The separable weights (given in log-domain) are exponentiated to ensure positivity and scale the contributions of each slice.
    Expected input:
      - x: a dinv.utils.TensorList, i.e. a list of tensors of shape [x_1, \dots, x_I]

    """

    def __init__(self, prior, separable_weights, *args, **kwargs):
        """
        :param dinv.optim.Prior prior: a Prior defining the function :math:`g`
        :param torch.Tensor separable_weights: a tensor of weights (in log-domain) of same length as the TensorList :math:`x`.

        Note: There is no separable_axis here because the separation is given by the list structure.
        """
        super().__init__(*args, **kwargs)
        self.prior = prior
        self.separable_weights = separable_weights
        self.explicit_prior = (
            prior[0].explicit_prior if isinstance(prior, list) else prior.explicit_prior
        )

    def fn(self, x, *args, **kwargs):
        """
        Compute the function value :math:`f(x)` as the weighted sum over slices.
        For each coordinate along the separable_axis, a slice is taken and the base prior function
        is applied. Each contribution is weighted by exp(separable_weights[coord]).
        :param dinv.utils.TensorList x: Input tensor.
        :return dinv.utils.TensorList : value of :math:`f(x)` for each batch
        """
        eseparable_weights = self.separable_weights
        f_list = [
            eseparable_weights[j] * self.prior.fn(x_j, *args, **kwargs)
            for j, x_j in enumerate(x)
        ]
        return torch.stack(f_list, dim=0).sum(dim=0)

    def prox(self, x, *args, gamma, **kwargs):
        """
        Compute the proximity operator associated with :math:`f`.
        The prox is computed for each tensor in the dinv.utils.TensorList. For each tensor in the list:
        .. math::
            \operatorname{prox}_{gamma * exp(w) * g}(x_i)
        is computed, and then returned as a dinv.utils.TensorList.
        :param dinv.utils.TensorList x: A list of input tensors.
        :param gamma: A step-size parameter (on :math:`\tau f`).
        :return dinv.utils.TensorList: :math:`\operatorname{prox}_{\tau f}(x)` of the same shape as :math:`x` after applying the proximity operator.
        """
        eseparable_weights = self.separable_weights
        prox_list = [
            self.prior.prox(x_j, gamma=gamma * eseparable_weights[j], *args, **kwargs)
            for j, x_j in enumerate(x)
        ]
        return TensorList(prox_list)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass: computes the aggregated function value f(x).
        """
        return self.fn(x, *args, **kwargs)
