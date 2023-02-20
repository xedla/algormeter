from typing import Optional

'''Calls Counter'''
_counter = dict()
__countEnabled = True

def reset() -> None:
    global _counter,__countEnabled
    _counter = {}
    __countEnabled = True

def enable() -> None:
    global __countEnabled
    __countEnabled = True

def disable() -> None:
    global __countEnabled
    __countEnabled = False

def log(value : str, elem  : str, cls : str = '') -> None:
    if __countEnabled == False:
        return 
    key = '.'.join((cls,elem)) if cls != '' else elem
    _counter[key] = value

def up(elem  : str, cls : str ='', number : int = 1) -> None:
    if __countEnabled == False:
        return 
    key = '.'.join((cls,elem)) if cls != '' else elem
    _counter[key] = _counter[key] + number if key in _counter else number

def get(elem : str, cls : str ='') -> Optional[str | None] :
    key = '.'.join((cls,elem)) if cls != '' else elem
    if key in _counter: 
        return  _counter[key]
    return None

def report() -> dict[str,str]:
    return {k : v for k,v in sorted(_counter.items(), key = lambda t : t[0])}

if __name__ == "__main__":
    up('e1')
    up('e1')
    assert get('e1') == 2

    up('e2',cls='c1')
    up('e2',cls='c1')
    up('e2',cls='c2')
    assert get('e2',cls='c1') == 2
    assert get('e2') == None
    assert get('e3') == None

    log('delta','stop')
    print(report())
