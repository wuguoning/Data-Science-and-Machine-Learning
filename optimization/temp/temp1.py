import autograd.numpy as np


from autograd import elementwise_grad, value_and_grad

if __name__ == '__main__':
    def f(x,y):
        return x**2 + y**2 + x + y

    dz_dx = elementwise_grad(f, argnum=0)
    dz_dy = elementwise_grad(f, argnum=1)
    print(dz_dx(1.,1.))
    print(dz_dx)
