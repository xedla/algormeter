import warnings

from .algormeter import algorMeter, Algorithms, Algorithm, Problems,Problem, TuneParameters
from .perfprof import perfProf
from .kernel import AlgorMeterWarning, Array1D
from .libs import *

oldsw = warnings.showwarning

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, AlgorMeterWarning):
        print(f"{category.__name__}: {message} (in {filename}, line {lineno})")
    else:
    # Usa il comportamento predefinito per altre categorie
        oldsw(message, category, filename, lineno, file, line)

warnings.showwarning = custom_showwarning

# warnings.warn("aaa", AlgorMeterWarning)
# warnings.warn("bbb.")
# exit()
