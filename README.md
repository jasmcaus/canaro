# Canaro
A Python library including support for Deep Learning models built using the Keras framework

## Installation
To install the current release:

```shell
$ pip install caer
```

Optionally, Canaro can also install [caer](https://github.com/jasmcaus/caer) if you install it with `pip install caer[caer]`

### Installing from Source
First, clone the repo on your machine and then install with `pip`:

```shell
git clone https://github.com/jasmcaus/canaro.git
cd canaro
pip install -e .
```

You can run the following to verify things installed correctly:

```python
import canaro

print(f'Canaro version {canaro.__version__}')
```

## License

Caer is released under the [MIT License](https://github.com/jasmcaus/caer/blob/master/LICENSE)