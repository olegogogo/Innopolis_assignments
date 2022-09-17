from task1 import decorator0
from inspect import signature, getsource


def decorator1(func):
    def describ(*args, **kwargs):
        result = decorator0(func)(*args, **kwargs)
        print('Name: {}'.format(func.__name__))
        print('Type: {}'.format(str(type(func))))
        print('Sign: {}'.format(str(signature(func))))
        print('Args: {} positional'.format(str(args)))
        print('Args: {} keyworded'.format(str(kwargs)))
        print('Doc: {}'.format(func.__doc__))
        print('Source: {}'.format(getsource(func)))
        print('Output: ')
        print(func(*args))
        return result

    return describ
