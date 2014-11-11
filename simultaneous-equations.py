'''
In order to solve simultaneous equations in 2 variables, we rely on
basic linear algebra, specialised to 2x2 matricies, to keep things
simple (and avoid having to pay for generality we do not intend
to use). It should become apparent that once we have represented our
data, and defined some common matrix operations, the actual solution
of the system is pretty easy.

It would be easy enough to generalise all of this to nxn matricies,
however, there are is a case to be made for specialised 2x2 or 3x3
matrix implementations, as when these are all that are required,
they can be implemented very efficiently using CPU vector operations
which do not extend to the general case (a feature not used in the
following implementation). For example, many graphics APIs will
include specialised 2d/3d vector implementations along these lines.

I have included 3 separate methods of solution:
 - Inverting the matrix via the 2x2 case of the Cayley-Hamilton
   theorem.
 - Gauss-Jordan elimination.
 - Numerical approximation via the Jacobi method.
'''

from operator import mul, neg
from functools import partial
from copy import deepcopy


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # By predefining the attributes of our vector, we can reduce the
    # size of the Vectors in memory
    __slots__ = ('x', 'y')

    def __getitem__(self, i):
        # Indexing of the vector
        if i == 1:
            return self.x
        elif i == 2:
            return self.y
        else:
            raise IndexError("This vector is of length 2")

    def __iter__(self):
        yield self.x
        yield self.y

    def map(self, f):
        # Elementwise application of a function to a vector
        return Vector(f(self.x), f(self.y))

    def __mul__(self, other):
        if isinstance(other, Vector):
            raise ValueError("We cannot multiply vectors!")
        elif isinstance(other, (int, float, complex)):
            # Multiply the vector by a scalar
            return self.map(partial(mul, other))
        else:
            return NotImplemented

    __rmul__ = __mul__ # This is to say, multiplication is symmetric

    def dot(self, other):
        '''The scalar product of two vectors.'''
        return self.x*other.x + self.y*other.y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        else:
            return NotImplemented

    def __neg__(self):
        return self.map(neg)

    def norm(self):
        '''The max norm of the vector.'''
        return max(abs(self.x), abs(self.y))

    # Operations for Gaussian elimination
    def scale_x(self, a):
        self.x *= a

    def scale_y(self, a):
        self.y *= a

    def scale_sub_x_y(self, a):
        self.y -= a*self.x

    def scale_sub_y_x(self, a):
        self.x -= a*self.y

    def flip_x_y(self):
        self.x, self.y = self.y, self.x

    def __repr__(self):
        return 'Vector({}, {})'.format(self.x, self.y)

    def __str__(self):
        return '[{}, {}]^T'.format(self.x, self.y)

def dot(u, v):
    return u.dot(v)


class Matrix:
    '''
    A 2x2 Matrix.

    Internally we will represent the matrix as two column vectors u, v
    so M := [u|v]. As will be apparent in the implementation, this
    implementation makes things fairly easy.

    We can construct a matrix like Matrix([x1, x2], [y1, y2])
    for the matrix
    [[x1, x2],
     [y1, y2]]

    For convenience, we can also specify u, v directly, like
    Matrix(u=Vector(x1, y1), v=Vector(x2, y2)) for the same matrix.
    '''
    def __init__(self, xs=None, ys=None, u=None, v=None):
        if isinstance(u, Vector) and isinstance(v, Vector):
            # We want genuine new copies of our vectors, not links to the old
            # ones.
            self.u, self.v = deepcopy(u), deepcopy(v)
        elif xs and ys and len(xs) == 2 and len(ys) == 2:
            self.u = Vector(xs[0], ys[0])
            self.v = Vector(xs[1], ys[1])
        else:
            raise ValueError('Invalid arguments')

    def transpose(self):
        return Matrix(xs=list(self.u), ys=list(self.v))

    def map(self, f):
        return Matrix(u=self.u.map(f), v=self.v.map(f))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            T = self.transpose()
            return Matrix(
                [dot(T.u, other.u), dot(T.u, other.v)],
                [dot(T.v, other.u), dot(T.v, other.v)],
            )
        elif isinstance(other, (int, float, complex)):
            return self.map(partial(mul, other))
        elif isinstance(other, Vector):
            return other.x*self.u + other.y*self.v
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        elif isinstance(other, Vector):
            raise ValueError("We can only multiply matricies by vectors on "
                "the right")
        else:
            raise NotImplemented

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(u=self.u + other.u, v=self.v + other.v)
        else:
            raise NotImplemented

    def __neg__(self):
        return self.map(neg)

    def norm(self):
        '''The max row sum norm'''
        return max(
            abs(self.u.x) + abs(self.u.y),
            abs(self.v.x) + abs(self.v.y),
        )

    def __str__(self):
        T = self.transpose()
        return '[{}, {}]'.format(repr(list(T.u)), repr(list(T.v)))

    def __repr__(self):
        T = self.transpose()
        return 'Matrix({}, {})'.format(repr(list(T.u)), repr(list(T.v)))

    def det(self):
        return self.u.x * self.v.y - self.u.y*self.v.x

    def adjugate(self):
        return Matrix([self.v.y, -self.v.x], [-self.u.y, self.u.x])

    def inverse(self):
        '''Calculate the inverse using the general equation.

        This is nice and simple but does not generalise well to nxn matrices.
        '''
        return (1/self.det())*self.adjugate()

    def D_inverse(self):
        '''The inverse of the diagonal.'''
        return Matrix([1/self.u.x, 0], [0, 1/self.v.y])

    @property
    def D(self):
        '''The diagonal.'''
        return Matrix([self.u.x, 0], [0, self.v.y])

    @property
    def L(self):
        '''The lower triangle.'''
        return Matrix([0, 0], [self.u.y, 0])

    @property
    def U(self):
        '''The upper triangle.'''
        return Matrix([0, self.v.x], [0, 0])

    # Operations for Gaussian elimination
    def scale_x(self, a):
        self.u.scale_x(a)
        self.v.scale_x(a)

    def scale_y(self, a):
        self.u.scale_y(a)
        self.v.scale_y(a)

    def scale_sub_x_y(self, a):
        self.u.scale_sub_x_y(a)
        self.v.scale_sub_x_y(a)

    def scale_sub_y_x(self, a):
        self.u.scale_sub_y_x(a)
        self.v.scale_sub_y_x(a)

    def flip_x_y(self):
        self.u.flip_x_y()
        self.v.flip_x_y()

class LinearSystem:
    '''The system At = c where t = [x, y]
    '''
    def __init__(self, A, c):
        self.A = A # Should be a matrix
        self.c = c # Should be a vector

    def solve_inverse(self):
        '''Solve the system by computing the matrix inverse.'''
        return self.A.inverse() * self.c

    def solve_gauss_jordan(self):
        '''Solve by Gauss Jordan elimination whereby the we perform
        elimination steps until the matrix is reduced to a diagonal
        matrix, and the solution is presented directly.'''
        # We work on a copy as not to disturb the system
        S = deepcopy(self)

        # We check for a zero in the upper diagonal, and
        # flip is necessary
        if S.A.u.x == 0:
            if S.A.u.y == 0:
                raise ValueError('This matrix is not invertible.')
            else:
                S.flip_x_y()

        # We scale to give a 1 in the upper diagonal
        S.scale_x(1/S.A.u.x)

        # We eliminate the lower diagonal left entry
        S.scale_sub_x_y(S.A.u.y)

        # We check that we have not been left with a zero in
        # the lower diagonal
        if S.A.v.y == 0:
            raise ValueError('This matrix is not invertible')

        # We scale the lower diagonal
        S.scale_y(1/S.A.v.y)

        # We eliminate the upper diagonal
        S.scale_sub_y_x(S.A.v.x)

        # We now have the solution
        return S.c

    def solve_jacobi(self, c0, tolerance, max_steps):
        '''Numerically approximate the solution using the Jacobi
        method with starting value c0, and the given tolerance and
        max_steps.

        This is a fixed point iteration on g(l) where,
        g(l) = -D^(-1)*(L + U)l + D^(-1)c

        This is not always guarenteed to converge, but will, for
        example, if the matrix is strictly diagonally dominant.
        '''
        l = deepcopy(c0)
        S = deepcopy(self)

        # If possible, flip some rows to remove zeros from the
        # diagonal
        if S.A.u.x == 0:
            if S.A.u.y == 0:
                raise ValueError("This matrix is not invertible")
            else:
                S.flip_x_y()
        elif S.A.v.y == 0:
            if S.A.v.x == 0:
                raise ValueError("This matrix is not invertible")
            else:
                S.flip_x_y()

        # Compute constants used in fixed point iteration
        Di = S.A.D_inverse()
        T = -Di*(S.A.L + S.A.U)
        b = Di*S.c

        for n in range(max_steps):
            l_old = deepcopy(l)
            l = T*l_old + b # Calculate the next step of the iteration

#            print(n, l, (l - l_old).norm())

            if (l - l_old).norm() < tolerance:
                # We are within tolerance, return the approximation
                return l

        raise Exception("Did not converge")

    # Operations for Gaussian elimination
    def scale_x(self, a):
        self.A.scale_x(a)
        self.c.scale_x(a)

    def scale_y(self, a):
        self.A.scale_y(a)
        self.c.scale_y(a)

    def scale_sub_x_y(self, a):
        self.A.scale_sub_x_y(a)
        self.c.scale_sub_x_y(a)

    def scale_sub_y_x(self, a):
        self.A.scale_sub_y_x(a)
        self.c.scale_sub_y_x(a)

    def flip_x_y(self):
        self.A.flip_x_y()
        self.c.flip_x_y()


if __name__ == '__main__':
    # A demonstration for the system 7x + 2y = 3, 4x + 5y = 6
    A = Matrix([7, 2], [4, 5])
    c = Vector(3, 6)
    S = LinearSystem(A, c)

    print('Inverse: ', S.solve_inverse())
    print('Gauss-Jordan elimination: ', S.solve_gauss_jordan())
    print('Jacobi method: ', S.solve_jacobi(Vector(0,0), 10**(-5), 100))
