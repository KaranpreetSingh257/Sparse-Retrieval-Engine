import sys

# Pre-download required NLTK resources during container build (per TA announcement)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

try:
    from setuptools import setup
    setup(
        name="submission",
        version="1.0.0",
        py_modules=[],
        packages=[],
    )
except Exception:
    pass
