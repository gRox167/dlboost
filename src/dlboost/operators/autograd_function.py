import torch
from deepinv.optim.utils import least_squares
from torch.autograd.function import once_differentiable


class ConjugateGradientFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        A, AT, y, z, init, gamma, parallel_dim, AAT, ATA, solver, max_iter, tol = inputs
        ctx.mark_non_differentiable(init)
        ctx.A = A
        ctx.AT = AT
        ctx.gamma = gamma
        ctx.parallel_dim = parallel_dim
        ctx.AAT = AAT
        ctx.ATA = ATA
        ctx.solver = solver
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.save_for_backward(y, z)

    @staticmethod
    def forward(
        A,
        AT,
        y,
        z=0.0,
        init=None,
        gamma=None,
        parallel_dim=0,
        AAT=None,
        ATA=None,
        solver="CG",
        max_iter=100,
        tol=1e-6,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of the conjugate gradient operator."""

        x = least_squares(
            A,
            AT,
            y,
            z,
            init,
            gamma,
            parallel_dim,
            AAT,
            ATA,
            solver,
            max_iter,
            tol,
            **kwargs,
        )

        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Backward pass of the conjugate gradient operator."""
        inputs = tuple(
            x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
        )
        with torch.enable_grad():
            rhs = ctx.rhs_factory(*inputs)
            operator = ctx.operator_factory(*inputs)
        inputs_with_grad = tuple(
            x
            for x, need_grad in zip(inputs, ctx.needs_input_grad[2:], strict=True)
            if need_grad
        )
        if inputs_with_grad:
            rhs_norm = (
                sum((r.abs().square().sum() for r in grad_output), torch.tensor(0.0))
                .sqrt()
                .item()
            )
            tol_ = ctx.tolerance * max(rhs_norm, 1e-6)  # clip in case rhs is 0
            with torch.no_grad():
                if isinstance(operator, LinearOperatorMatrix):
                    z = cg(
                        operator.H,
                        grad_output,
                        tolerance=tol_,
                        max_iterations=ctx.max_iterations,
                    )
                else:
                    z = cg(
                        operator.H,
                        grad_output[0],
                        tolerance=tol_,
                        max_iterations=ctx.max_iterations,
                    )
            if any(zi.isnan().any() for zi in z):
                raise RuntimeError("NaN in ConjugateGradientFunction.backward")
            with torch.enable_grad():
                residual = tuple(
                    r - ax
                    for r, ax in zip(
                        rhs, operator(*(s.detach() for s in solution)), strict=True
                    )
                )
            grads = torch.autograd.grad(
                outputs=residual,
                inputs=inputs_with_grad,
                grad_outputs=z,
                allow_unused=True,
            )
            grad_iter = iter(grads)
        else:
            grad_iter = iter(())

        grad_input = tuple(
            next(grad_iter) if need else None for need in ctx.needs_input_grad[2:]
        )
        return (None, None, *grad_input)  # operator_factory, rhs_factory, *inputs
