# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

from .models import createDefaultModel
from .models import createSimpsonsModel
from .models import LeNet
from .models import VGG16
from .models import saveModel
from .models import testModel

from .train import lr_schedule
from .train import train