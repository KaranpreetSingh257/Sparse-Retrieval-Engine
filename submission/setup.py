import sys
import socket

# Set socket timeout to 10s to prevent network hanging during docker build
socket.setdefaulttimeout(10.0)

try:
    import nltk
    nltk.download('punkt', quiet=True, raise_on_error=False)
    nltk.download('stopwords', quiet=True, raise_on_error=False)
    nltk.download('punkt_tab', quiet=True, raise_on_error=False)
except BaseException:
    pass

try:
    from setuptools import setup
    setup(
        name="submission",
        version="1.0.0",
        py_modules=[],
        packages=[],
    )
except BaseException:
    pass
