from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

AUTHOR = "Verve"
VERSION = "0.0.2"
DESCRIPTION = "Simple interface for neural chatbots."

setup(
    name="neural_chat",
    version=VERSION,
    author=AUTHOR,
    author_email="<verve_is_god@protonmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["numpy", "nltk", "tensorflow", "gtts", "pyttsx3"],
    keywords=["python", "neural", "machine learning", "chatbots",
              "chat", "artificial intelligence", "virtual assistant", "voice"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ]
)
