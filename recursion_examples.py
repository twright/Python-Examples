'''A few simple examples of recursion.'''

def sum1(xs):
    '''We can recursively sum a list of numbers.'''
    if len(xs) == 0:
        return 0
    else:
        return xs[0] + sum1(xs[1:])

def sum2(xs):
    '''Or do the same thing iteratively.'''
    y = 0
    for x in xs:
        y += x
    return y

def product1(xs):
    '''Similarly we can recursively take the product of a list of numbers.'''
    if len(xs) == 0:
        return 1
    else:
        return xs[0] + sum1(xs[1:])

def concat1(xs):
    '''Concatenate a list of strings.'''
    if len(xs) == 0:
        return ''
    else:
        return xs[0] + concat1(xs[1:])

def reverse1(xs):
    '''Or reverse a list.'''
    if len(xs) == 0:
        return xs
    else:
        return reverse(xs[1:]) + [xs[0]]

# At this point we realise all of these examples are practically
# identical (i.e. the recursion structure is the same), we can
# abstract them into two recursive functions.

def foldl(xs, op, id):
    '''Folds a list xs from the left with binary operation op,
    and identity id.'''
    if len(xs) == 0:
        return id
    else:
        return op(foldl(xs[1:], op, id), xs[0])

def foldr(xs, op, id):
    '''Folds a list xs from the right with binary operation op,
    and identity id.'''
    if len(xs) == 0:
        return id
    else:
        return op(xs[0], foldr(xs[1:], op, id))
