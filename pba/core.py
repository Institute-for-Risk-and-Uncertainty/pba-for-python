from .interval import Logical
def env(x,y):
    if x.__class__.__name__ == 'Pbox':
        return x.env(y)
    elif y.__class__.__name__ == 'Pbox':
        return y.env(x)
    else:
        raise NotImplementedError('At least one argument needs to be a Pbox')

def min(x,y):
    if x.__class__.__name__ == 'Pbox':
        return x.min(y)
    if y.__class__.__name__ == 'Pbox':
        return y.min(x)
    else:
        raise NotImplementedError('At least one argument needs to be a Pbox')

def max(x,y):
    if x.__class__.__name__ == 'Pbox':
        return x.max(y)
    if y.__class__.__name__ == 'Pbox':
        return y.max(x)
    else:
        raise NotImplementedError('At least one argument needs to be a Pbox')

def always(logical: Logical or bool):

    if logical.__class__.__name__ != 'Logical':
        return logical
    if logical.left == 1 and logical.right == 1:
        return True
    else:
        return False


def sometimes(logical: Logical or bool):

    if logical.__class__.__name__ != 'Logical':
        return logical

    elif logical.left == 1 or logical.right == 1:
        return True
    else:
        return False

def xtimes(logical: Logical or bool):
    '''
    exclusive sometimes
    
    Returns true if the logical function is sometimes True but not always true
    If the input is not a Logical class then function will always return false
    '''

    if logical.__class__.__name__ != 'Logical':
        return False

    elif logical.left ^ logical.right:
        return True
    else:
        return False
