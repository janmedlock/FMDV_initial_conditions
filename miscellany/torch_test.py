#!/usr/bin/python3

import math

import matplotlib.pyplot
import torch


NPOINTS = 2000
ORDER = 3
LEARNING_RATE = 1e-3
NITER = 2000

# Create input and output data
x = torch.linspace(-math.pi, math.pi, NPOINTS)
y = torch.sin(x)


class Model:
    def __init__(self, order):
        self.parameters = torch.randn(order + 1,
                                      requires_grad=True)

    def __call__(self, x):
        x_pow_n = x.unsqueeze(-1) \
                   .pow(torch.arange(len(self.parameters)))
        y = x_pow_n @ self.parameters
        return y


model = Model(ORDER)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([model.parameters],
                            lr=LEARNING_RATE)
for i in range(NITER):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
print(f'{model.parameters=}')
matplotlib.pyplot.plot(x, y,
                       x, model(x).detach())
matplotlib.pyplot.show()
