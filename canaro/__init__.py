# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

from ._meta import version as v
from ._meta import author as a
from ._meta import release as r
from ._meta import contributors as c
__author__ = a
__version__ = '1.0.5'
__contributors__ = c
__license__ = 'MIT License'
__copyright__ = 'Copyright (c) 2020 Jason Dsouza'
version = v
release = r
contributors = c

from .model import saveModel
from .model import testModel

from .train import lr_schedule
from .train import train

from .tools import expand
from .tools import random_distortion
from .tools import random_patch
from .tools import random_resize
from .tools import resize_image_fixed
from .tools import resize_image
from .tools import clip_boxes
from .tools import flip_image
from .tools import patch_image