import inspect
import builtins as __builtin__


_DBPRINT = False

def print(*args, **kwargs) -> None:
    if _DBPRINT:
        stack = inspect.stack()
        elem = stack[1][0].f_code.co_name
        __builtin__.print(elem,'>>>', *args,**kwargs)

