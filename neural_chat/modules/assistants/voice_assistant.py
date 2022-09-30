from typing import Text
from pyttsx3 import init

engine = init()
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)


class VoiceAssistant:
    def __init__(self, audiostring: Text):
        """
        The __init__ function is the constructor for a class. It is called whenever an instance of the class is created.
        The __init__ function can take arguments, but self is always the first one.

        :param self: Refer to the instance of the class
        :param audiostring:Text: Set the audiostring attribute of the audiofile class
        :return: The object itself
        """
        self.audiostring = audiostring

    def say(self):
        """
        The say function takes a string as an argument and speaks it to the user.
        It also uses the pyttsx3 module to accomplish this.

        :param self: Reference the object itself
        :return: The audiostring, which is the text that will be read
        """
        print(self.audiostring)
        engine.say(self.audiostring)
        engine.runAndWait()
