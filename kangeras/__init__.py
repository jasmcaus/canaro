# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

from .version import version as __version__
from .version import author as __author__

from .model import saveModel
from .model import testModel

from .train import lr_schedule
from .train import train