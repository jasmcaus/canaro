# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

from ._meta import version as v
from ._meta import author as a
from ._meta import release as r
from ._meta import contributors as c
__author__ = a
__version__ = v
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