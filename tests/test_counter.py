from algormeter import *
from algormeter.libs import *
from algormeter.tools import counter, dbx


def test_e1():
    counter.reset()
    counter.up('e1')
    counter.up('e1')
    assert counter.get('e1') == 2

def test_e2_c1():
    counter.reset()
    counter.up('e2',cls='c1')
    counter.up('e2',cls='c1')
    counter.up('e2',cls='c2')
    assert counter.get('e2',cls='c1') == 2
    assert counter.get('e2') == None

def test_log():
    counter.reset()
    counter.log('hi', 'e2',cls='c1')
    assert counter.get('e2',cls='c1') == 'hi'
    assert counter.get('e2') == None

def test_get():
    assert counter.get('e3') == None

def test_report():
    counter.reset()
    counter.up('e1',cls='c1')
    counter.up('e2',cls='c2')
    assert len(counter.report()) == 2
