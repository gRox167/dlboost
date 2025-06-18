import torch
from deepinv.optim.optim_iterators import OptimIterator
from deepinv.optim.optim_iterators.gradient_descent import fStepGD, gStepGD


def linesearch(x, grad, cur_data_fidelity, cur_prior, cur_params, y, physics):
    """Line search for the stepsize.
    :param torch.Tensor x: Current iterate.
    :param torch.Tensor grad: Gradient of the cost function.
    :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
    :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
    :param dict cur_params: Dictionary containing the current parameters of the algorithm.
    :param torch.Tensor y: Input data.
    :return: Updated stepsize.
    """
    stepsize = cur_params["stepsize"]
    x_new = x - stepsize * grad
    F_new = cur_data_fidelity(x_new, y, physics)
    F = cur_data_fidelity(x, y, physics)
    while F_new > F:
        stepsize *= 0.5
        x_new = x - stepsize * grad
        F_new = cur_data_fidelity(x_new, y, physics)
    return stepsize


class GDIteration(OptimIterator):
    r"""
    Iterator for Gradient Descent.

    Class for a single iteration of the gradient descent (GD) algorithm for minimising :math:`f(x) + \lambda \regname(x)`.

    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        v_{k} &= \nabla f(x_k) + \lambda \nabla \regname(x_k) \\
        x_{k+1} &= x_k-\gamma v_{k}
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize.

    """

    def __init__(self, line_search=False, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True
        self.line_search = line_search

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single gradient descent iteration on the objective :math:`f(x) + \lambda \regname(x)`.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        with torch.no_grad():
            f_grad = self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
            # do a line search on the stepsize
            stepsize = cur_params["stepsize"]
            if self.line_search:
                stepsize = linesearch(
                    x_prev, f_grad, cur_data_fidelity, cur_prior, cur_params, y, physics
                )
                print(f"stepsize: {stepsize}")
                # save the stepsize for the next iteration
                cur_params["stepsize"] = stepsize
        if cur_params["lambda"] == 0:
            grad = stepsize * f_grad
        else:
            grad = stepsize * (self.g_step(x_prev, cur_prior, cur_params) + f_grad)
        x = x_prev - grad
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x,), "cost": F}
