import functools

from .logging import init_default_logger

logger = init_default_logger(__name__, fstring="[%(levelname)s]: %(message)s")


def optional_arg_decorator(func):
    """Enables a decorator to have optional arguments.

    https://stackoverflow.com/questions/3888158
    """

    def wrapped_decorator(*args):
        # usage case of @decorator
        if len(args) == 1 and callable(args[0]):
            return func(args[0])
        # usage case of @decorator(...)
        else:

            def real_decorator(decoratee):
                return func(decoratee, *args)

            return real_decorator

    return wrapped_decorator


@optional_arg_decorator
def onlycallonce(func, update_flag=None):
    """Custom Decorator. Ensures decorated method is only called once.

    Creates a flag under self.'update_flag' which is referenced to check whether
    the decorated method has already been called or not.
    In case...
        ...of a first time call:
        --> Method is called and self.'update_flag' is set to True.

        ...the method has already been called:
        --> Method is not called, an error is logged instead.

    Usage:
        @onlycallonce
        def dosomething(self, x, y):
            pass

        ---> default, will use self.__dosomething_HasBeenCalled as a flag.

        or

        @onlycallonce("customFlagName")
        def dosomething(self, x, y):
            pass

        ---> custom update flag named self.customFlagName is used instead.
    """
    update_flag = update_flag or f"__{func.__name__}_HasBeenCalledFlag"

    @functools.wraps(func)  # preserve docstring of function using functools
    def wrapper_func(self, *args, **kwargs):
        if getattr(self, update_flag, False):
            error_msg = (
                f"{func.__name__}() has already been called and cannot be called again."
            )
            logger.error(error_msg)
        else:
            out = func(self, *args, **kwargs)
            setattr(self, update_flag, True)
            return out

    return wrapper_func
