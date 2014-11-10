'''
In order to numerically solve a ordinary differential equation, we
will simply use the classical 4th order Runge-Kutta method, based
on the Wikipedia article on the subject:
http://en.wikipedia.org/wiki/Runge-Kutta_methods

Here we will allow the ODE to be specified as a python function,
    f = lambda t, y: ...
representing dy/dt, and an initial condition y0 = y(t0).

The code will be written mainly with clarity in mind, rather than
striving for maximal efficiency or generality.
'''

# Some basic functional programming libraries
from functools import reduce
from operator import mul

def runge_kutta(f, t0, y0, h):
    '''
    Classical Runge-Kutta method for dy/dt = f(t, y), y(t0) = y0,
    with step h, and the specified tolerance and max_steps.

    This function is a generator which can give infinitely many points
    of the estimated solution, in pairs (t[n], y[n]).

    To get only finitely many values of
    the solution we can for example do,
        >>> from itertools import islice
        >>> list(islice(runge_kutta(f, t0, h), n))
        [(t[0], y[0]), (t[1], y[1]), ..., (t[n], y[n])]

    and infact, we could define another function to do this like,
        >>> runge_kutta_N = lambda f, t0, y0, h, N: list(islice(
        ...     runge_kutta(f, t0, y0, h), N))

    It would also be easy to change this function to take an extra
    parameter N and then return a list of the first N, (t_n, y_n),
    directly (by replacing the while loop with for n in range(N)).

    Note also that whilst the question asks for a solution, this
    function only returns an approximation of the solution at
    certain points. We can turn use this to generate a continuous
    function passing through the points specified using either of
    interpolation methods specified lower down the file.
    '''
    # y and t represent y[n] and t[n] respectively at each stage
    y = y0
    t = t0

    # Whilst it would be more elegant to write this recursively,
    # in Python this would be very inefficient, and lead to errors when
    # many iterations are required, as the language does not perform
    # tail call optimisations as would be the case in languages such
    # as C, Lisp, or Haskell.
    #
    # Instead we use a simple infinite loop, which will yield more values
    # of the function indefinitely.
    while True:
        # Generate the next values of the solution y
        yield t, y

        # Values for weighted average (compare with Wikipedia)
        k1 = f(t, y)
        k2 = f(t + h/2, y + (h/2)*k1)
        k3 = f(t + h/2, y + (h/2)*k2)
        k4 = f(t + h/2, y + h*k3)

        # Calculate the new value of y and t as per the Runge-Kutta method
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

def linear_interpolation(tys):
    '''Takes a list of (t, y(t)) values (presumed to be in increasing
    order of t), and interpolates to a piecewise linear function
    passing through each point.

    Heavily inspired by: http://en.wikipedia.org/wiki/Linear_interpolation
    '''
    # The t values
    ts = [t for t, y in tys]
    # The y values
    ys = [y for t, y in tys]
    # The number of values
    n = len(tys) # = len(ts) = len(ys)

    if n < 2:
        raise ValueError("Not enough points to interpolate!")

    def z(x):
        # If x is beyond the lower end of the data, we extrapolate
        # based on the first two values.
        if x <= min(ts):
            i = 0
            return ys[0] + (ys[1] - ys[0]) * (x - ts[0])/(ts[1] - ts[0])
        # If x is beyond the upper end of the data, we extrapolate
        # based on the last two values.
        elif x >= max(ts):
            i = n-2
        else:
            # Find the interval i, that x lies within
            i = next(i for i in range(n - 1) if ts[i] <= x <= ts[i+1])

        # Interpolate within our chosen interval
        return ys[i] + (ys[i+1] - ys[i]) * (x - ts[i])/(ts[i+1] - ts[i])

    # Return a continous function z which interpolates between the given values
    return z

# A concise definition of the product of a list/iterable of numbers
# (sum is built in but product is not)
product = lambda xs: reduce(mul, xs, 1)

def polynomial_interpolation(tys):
    '''
    Takes a list of n (t, y(t)) values (presumed to be in increasing
    order of t), and interpolates to a nth order polynomial
    passing through each point.

    Uses the Lagrange polynomial:
     - http://en.wikipedia.org/wiki/Lagrange_polynomial
     - http://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html

    Note that this implementation could easily be made much faster
    at the expense of clarity (and similarity to the Wikipedia
    article).
    '''
    # The t values
    ts = [t for t, y in tys]
    # The y values
    ys = [y for t, y in tys]
    # The number of values
    n = len(tys) # = len(ts) = len(ys)

    # Return the interpolation function
    return lambda x: sum(ys[j]*product((x - ts[k])/(ts[j] - ts[k])
        for k in range(n) if j != k) for j in range(n))

if __name__ == '__main__':
    # A demonstration of the method for,
    # dy/dt = (1/3)*y + e^t*y^4, y(0) = 1
    # (chosen to test the method, as an explict solution does exist)
    print('dy/dt = (1/3)*y + e^t*y^4, y(0) = 1')
    from itertools import islice
    from math import exp
    # Evaluate the solution at some points
    tys = list(islice(runge_kutta(
        lambda t, y: (1/3)*y + exp(t)*y**4, # f(t, y)
        0, # t0
        1, # y0
        0.001,
    ), 200))
    # Print some values
    print('\n'.join('y({:.3f}) = {}'.format(t, y) for t, y in tys))

    # The explicit solution of the equation
    z = lambda t: ((-2)**(1/3)*exp(t/3))/(3*exp(2*t) - 5)**(1/3)
    # Calculate maximum absolute error
    e_abs_max = max(abs(y - z(t)) for t, y in tys)
    # Calculate maximum relative error
    e_rel_max = max(abs((y - z(t))/z(t)) for t, y in tys)
    # Print the errors
    print('Maximum absolute error: ', e_abs_max)
    print('Maximum relative error:', e_rel_max)

    # To check the interpolation is not completely messed up, check it
    # agrees on the actual data points
    z_linear = linear_interpolation(tys)
    z_polynomial = polynomial_interpolation(tys)
    print('Linear interpolation agrees?',
        'Yes' if all(y == z_linear(t) for t, y in tys) else 'No')
    print('Polynomial interpolation agrees?',
        'Yes' if all(y == z_polynomial(t) for t, y in tys) else 'No')

    try:
        # Plot some graphs to compare the results, this time with only
        # 20 points to compare interpolation methods more clearly
        import matplotlib.pyplot as plt

        # Evaluate the solution at some points
        tys2 = list(islice(runge_kutta(
            lambda t, y: (1/3)*y + exp(t)*y**4, # f(t, y)
            0, # t0
            1, # y0
            0.02,
        ), 10))
        ts = [t for t, y in tys2]
        ys = [y for t, y in tys2]
        z_linear_2 = linear_interpolation(tys2)
        z_polynomial_2 = polynomial_interpolation(tys2)
        ts_dense = [n*0.001 for n in range(200)]
        plt.plot(ts_dense, [z_linear_2(t) for t in ts_dense], 'g',
            label='Linear')
        plt.plot(ts_dense, [z_polynomial_2(t) for t in ts_dense], 'r',
            label='Polynomial')
        plt.plot(ts_dense, [z(t) for t in ts_dense], 'y',
            label='Exact solution')
        plt.plot(ts, ys, 'b^', label='Runge-Kutta')
        plt.legend()
        plt.ylabel("y")
        plt.xlabel("t")
        plt.show()
    except ImportError:
        print("Matplotlib required for graphs")
