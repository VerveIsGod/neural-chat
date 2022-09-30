__title__ = 'neural-chat'
__author__ = 'Verve'
__license__ = 'MIT'
__copyright__ = 'Copyright 2022 Verve'
__version__ = '0.0.2'

from json import loads
from requests import get
from .modules.assistants.voice_assistant import VoiceAssistant
from .modules.assistants.generic_assistant import GenericAssistant

__newest__ = loads(get("https://pypi.org/pypi/amino.fix/json").text)["info"]["version"]

if __version__ != __newest__:
    print(f"New version of {__title__} available: {__newest__} (Using {__version__})")
    print("Run \"pip install neural-chat -U\"")
