#!/usr/bin/python3
'''Solve a toy problem with PyTorch.'''

import functools
import math

import matplotlib.pyplot
import torch


def build_data(npoints):
    '''Create input and output data.'''
    input_ = torch.linspace(-1, 1, npoints)
    target = torch.sin(math.pi * input_)
    return (input_, target)


class Polynomial:
    '''A univariate polynomial.'''

    # `_polynomial(input_, degree, *, out=None)` basis functions.
    _polynomial = torch.special.legendre_polynomial_p

    # Use the mean-squared error for the loss function.
    loss = torch.nn.MSELoss()

    # Use the stochastic gradient decent optimizer.
    Optimizer = torch.optim.SGD

    def __init__(self, order):
        self.order = order
        # Initialize the parameters to random values and include the
        # gradient.
        self.parameters = torch.randn(self.order + 1,
                                      requires_grad=True)

    # Build `_degrees` once for efficiency.
    @functools.cached_property
    def _degrees(self):
        return torch.arange(self.order + 1).unsqueeze(-1)

    def _polynomials(self, input_, *, out=None):
        '''The (order, len(input_)) tensor
        [polynomial(input_, 0),
         polynomial(input_, 1),
         ...
         polynomial(input_, order)].'''
        return self._polynomial(input_, self._degrees, out=out)

    def forward(self, input_, *, out=None):
        '''The forward model.'''
        return self.parameters @ self._polynomials(input_, out=out)

    # Simplify calling the forward model.
    __call__ = forward

    def optimize(self, input_, target, niter, **kws):
        '''Do `niter` steps of the optimizer.'''
        optimizer = self.Optimizer([self.parameters], **kws)
        for _ in range(niter):
            optimizer.zero_grad()
            # Compute the gradient with respect to the model parameters.
            self.loss(self.forward(input_), target).backward()
            optimizer.step()


if __name__ == '__main__':
    (input_, target) = build_data(npoints=2000)

    model = Polynomial(order=3)
    model.optimize(input_, target, niter=2000, lr=1e-3)
    print(f'{model.parameters=}')

    with torch.no_grad():
        prediction = model(input_)
    axes = matplotlib.pyplot.gca()
    axes.plot(input_, target, label='target')
    axes.plot(input_, prediction, label='prediction')
    axes.set_xlabel('input_')
    axes.legend()
    matplotlib.pyplot.show()
