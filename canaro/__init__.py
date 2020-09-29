# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

from ._version import version as __version__
from ._version import author as __author__

from .model import saveModel
from .model import testModel

from .train import lr_schedule
from .train import train