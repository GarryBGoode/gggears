from importlib.metadata import version, PackageNotFoundError
from gggears.curve import *
from gggears.defs import *
from gggears.function_generators import *
from gggears.gggears_convert import *
from gggears.gggears_core import *
from gggears.gggears_build123d import *
from gggears.gggears_wrapper import *
from gggears.gearmath import *


try:
    __version__ = version("bd_warehouse")
except PackageNotFoundError:
    __version__ = "unknown version"
