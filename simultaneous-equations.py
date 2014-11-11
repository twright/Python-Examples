'''
In order to solve simultaneous equations in 2 variables, we rely on
basic linear algebra, specialised to 2x2 matrices, to keep things
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

It would have been fairly easy to implement this more concisely just
representing each row/column vector as a list, and manipulating them directly,
but this way is cleaner, and the matrix operations are easy to reuse (if this
was for an interview, or otherwise timed, it would be quicker to forget
abstraction and compress it all into a single class/function, with ). Of
course, in practice, you would just use a library such as numpy for the
matrices/vectors to avoid reinventing the wheel. 

I have included 3 separate methods of solution:
 - Inverting the matrix via the 2x2 case of the Cayley-Hamilton
   theorem.
 - Gauss-Jordan elimination.
 - Numerical approximation via the Jacobi method.

I suggest you start by looking at the methods in LinearSystem and only look
at the implementations of the individual matrix/vector operations lower down as
interested.

A quick summary of the API of Matrix:
create a new matrix:
A = Matrix([a, b], [c, d])

access the elements:
a == A.u.x
b == A.v.x
c == A.u.y
d == A.v.y

split out lower, upper, and diagonal matrices:
L = A.L
U = A.U
D = A.D

the basic operations of Gaussian elimination (which work in place):
A.scale_x(a)
A.scale_y(a)
A.flip_x_y()
A.scale_sub_x_y(a)
A.scale_sub_y_x(a)
'''

from operator import mul, neg
from functools import partial
from copy import deepcopy


class LinearSystem:
    '''The system At = c where t = [x, y]'''
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


class Vector:
    '''
    A class defining the functionality of vectors.

    We can initialise a new vector like v = Vector(x, y) storing its
    x and y components, accessible via v.x and v.y.

    We will define what all of the standard operations (+, -, *, etc) mean
    for vectors, as well as defining a methods, e.g. v.norm() for the
    max norm.
    '''

    def __init__(self, x, y):
        # Initialise a new vector
        self.x = x
        self.y = y

    # By predefining the attributes of our vector, we can reduce the
    # size of the Vectors in memory
    __slots__ = ('x', 'y')

    def __getitem__(self, i):
        '''We can access each of the vector's components by index, so
        v.x == v[1] and v.y == v[2].'''
        # Indexing of the vector
        if i == 1:
            return self.x
        elif i == 2:
            return self.y
        else:
            raise IndexError("This vector is of length 2")

    def __iter__(self):
        '''We can loop through a vector component by component.'''
        yield self.x
        yield self.y

    def map(self, f):
        '''Elementwise application of a function to a vector.'''
        return Vector(f(self.x), f(self.y))

    def __mul__(self, other):
        '''Multiply a vector by a scalar.'''
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
        '''Add two vectors.'''
        if isinstance(other, Vector):
            # [a, b] + [c, d] = [a + c, b + d]
            return Vector(self.x + other.x, self.y + other.y)
        else:
            return NotImplemented

    def __sub__(self, other):
        '''Subtract a vector from another.'''
        if isinstance(other, Vector):
            # [a, b] - [c, d] = [a - c, b - d]
            return Vector(self.x - other.x, self.y - other.y)
        else:
            return NotImplemented

    def __neg__(self):
        '''Negate a vector (e.g. -v).'''
        # This just says that -[x, y]^T = [-x, -y]^T
        return self.map(neg)

    def norm(self):
        '''The max norm of the vector.'''
        return max(abs(self.x), abs(self.y))

    # Operations for Gaussian elimination
    def scale_x(self, a):
        '''Scale the x component by factor a.'''
        self.x *= a

    def scale_y(self, a):
        '''Scale the y component by factor a.'''
        self.y *= a

    def scale_sub_x_y(self, a):
        '''Scale and subtract the x component from the y component.'''
        self.y -= a*self.x

    def scale_sub_y_x(self, a):
        '''Scale and subtract the y component from the x component.'''
        self.x -= a*self.y

    def flip_x_y(self):
        '''Flip the x and y components.'''
        self.x, self.y = self.y, self.x

    def __repr__(self):
        return 'Vector({}, {})'.format(self.x, self.y)

    def __str__(self):
        return '[{}, {}]^T'.format(self.x, self.y)


def dot(u, v):
    '''Shortcut function to find the scalar product between vectors u and v.'''
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
    # Initialise a new matrix
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
        '''Calculate the transpose of a metrix.'''
        return Matrix(xs=list(self.u), ys=list(self.v))

    def map(self, f):
        '''Apply function f elementwise to a matrix, that is,
        given a matrix [[a, b], [c, d]] and function f, this returns,
        [[f(a), f(b)], [f(c), f(d)]].'''
        return Matrix(u=self.u.map(f), v=self.v.map(f))

    def __mul__(self, other):
        '''Multiply a matrix by another matrix, a scalar, or a vector (in this
        case multiplication is only defined on the correct side).'''
        if isinstance(other, Matrix):
            # Multiply two matrices (self * other)
            T = self.transpose()
            return Matrix(
                [dot(T.u, other.u), dot(T.u, other.v)],
                [dot(T.v, other.u), dot(T.v, other.v)],
            )
        elif isinstance(other, (int, float, complex)):
            # Multiply a matrix by a scalar
            return self.map(partial(mul, other))
        elif isinstance(other, Vector):
            # Multiply a matrix by a vector (matrix self * vector other)
            return other.x*self.u + other.y*self.v
        else:
            return NotImplemented

    def __rmul__(self, other):
        '''Define multiplication on the right also.'''
        if isinstance(other, (int, float, complex)):
            # This case is commutative
            return self.__mul__(other)
        elif isinstance(other, Vector):
            raise ValueError("We can only multiply matrices by vectors on "
                "the right")
        else:
            raise NotImplemented

    def __add__(self, other):
        '''Add two matrices.'''
        if isinstance(other, Matrix):
            # We simply define this in terms of vector addition
            return Matrix(u=self.u + other.u, v=self.v + other.v)
        else:
            raise NotImplemented

    def __neg__(self):
        '''The negation of a matrix (-A).'''
        return self.map(neg)

    def norm(self):
        '''The max row sum norm.'''
        return max(
            abs(self.u.x) + abs(self.u.y),
            abs(self.v.x) + abs(self.v.y),
        )

    def __str__(self):
        '''Represent the matrix as a string.'''
        T = self.transpose()
        return '[{}, {}]'.format(repr(list(T.u)), repr(list(T.v)))

    def __repr__(self):
        '''Represent the matrix as python code.'''
        T = self.transpose()
        return 'Matrix({}, {})'.format(repr(list(T.u)), repr(list(T.v)))

    def det(self):
        '''The matrix determinant.'''
        return self.u.x * self.v.y - self.u.y*self.v.x

    def adjugate(self):
        '''The matrix adjugate.'''
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
        self.c.flip_x_y()


if __name__ == '__main__':
    # A demonstration for the system 7x + 2y = 3, 4x + 5y = 6
    A = Matrix([7, 2], [4, 5])
    c = Vector(3, 6)
    S = LinearSystem(A, c)

    print('Inverse: ', S.solve_inverse())
    print('Gauss-Jordan elimination: ', S.solve_gauss_jordan())
    print('Jacobi method: ', S.solve_jacobi(Vector(0,0), 10**(-5), 100))
