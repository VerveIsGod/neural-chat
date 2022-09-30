import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
import random
import pickle
import numpy as np
from json import load
from warnings import warn
from gtts import gTTS, langs
from abc import ABCMeta, abstractmethod
from nltk.stem import WordNetLemmatizer
from voice_assistant import VoiceAssistant
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout
from typing import Union, List, Dict, Text, Optional, Any
from tensorflow.keras.models import Sequential, load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self) -> Optional[Any]:
        """
        The train_model function trains the model.

        :param self: Access the attributes and methods of the class in python
        :return: The trained model
        """

    @abstractmethod
    def request(self, message: Text) -> Optional[Any]:
        """
        The request function is used to send a request to the server.
        It takes in a message, and returns the response from the server as a string.
        If there is no response from the server, it will return None.

        :param self: Access the attributes and methods of the class in python
        :param message:Text: Pass the message to be sent to the api
        :return: A response object
        """


class GenericAssistant(IAssistant):
    def __init__(self,
                 intents: Text,
                 intent_methods: Dict = {},
                 model_name: Text = "assistant_model",
                 voice_assistant: bool = False,
                 voice_saving: bool = False,
                 language: Text = "english",
                 encoding: Text = "utf-8") -> None:
        """
        The __init__ function is called when an instance of the class is created.
        It initializes all the variables that are defined by the class and become
        attributes for it. In this case, we initialize a variable intents to store
        the name of our intents file, intent_methods to store a dictionary with all
        of our methods from our intents file, model_name to store the name for your model,
        voice_assistant and voice_saving as boolean values (True or False), language and encoding as strings.

        :param self: Reference the object to which the function is applied
        :param intents: str: Pass the name of the intents file
        :param intent_methods: dict = {}: Store the methods of intents file
        :param model_name: str = "assistant_model": Name the model
        :param voice_assistant: bool = False: Play the answer
        :param voice_saving: bool = False: Save the voice of the answer to a file
        :param language: str = "english": Set the language for the nltp
        :param encoding: str = "utf-8": Specify the encoding of your model file
        :return: None
        """
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        self.language = language
        self.encoding = encoding
        self.voice_assistant = voice_assistant
        self.voice_saving = voice_saving

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents: Text) -> None:
        """
        The load_json_intents function loads the intents from a JSON file.
        The function takes one argument, which is the name of the JSON file to be loaded.
        It returns nothing.

        :param self: Access variables that belongs to the class
        :param intents: str: Specify the path to the json file that contains all of your intents
        :return: A dictionary that contains the intents and their corresponding sentences
        """
        with open(intents, encoding=self.encoding) as intents_file:
            self.intents = load(intents_file)

    def train_model(self) -> None:
        """
        The train_model function trains the model for NLTP.
        It takes in a list of words, classes and documents as input.
        The function then creates an empty array to hold the bag of words(0 or 1 depending on if they appear).
        The output is also created as an empty array with each index representing one class.
        Then it loops through all the patterns in our training data, tokenizes them using nltk word_tokenize and lemmatizer from NLTK package.
        For every word that is not a !.? character it checks if it's in our list of words and appends 1 to bag_of

        :param self: Reference the class itself inside the function
        :return: None
        """
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern, language=self.language)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def save_model(self, model_name: Text = None) -> None:
        """
        The save_model function saves the trained model and its associated words and classes to a directory.
        The function takes in an optional model_name parameter, which is used as the name of the file that
        the model will be saved to. If no value is passed in for this parameter, then save_model will use
        the value stored in self.model_name.

        :param self: Allow the function to refer to the class itself
        :param model_name: str = None: Specify a new name for the model
        :return: None
        """
        self.model.save(f"{model_name if model_name is not None else self.model_name}.h5", self.hist)
        with open(f'{model_name if model_name is not None else self.model_name}_words.pkl', 'wb') as file:
            pickle.dump(self.words, file)
        with open(f'{model_name if model_name is not None else self.model_name}_classes.pkl', 'wb') as file:
            pickle.dump(self.classes, file)

    def load_model(self, model_name: Text = None) -> None:
        """
        The load_model function loads the existing model.

        :param self: Access variables that belongs to the class
        :param model_name: str = None: Load a model with a different name than the default one
        :return: None
        """
        with open(f'{model_name if model_name is not None else self.model_name}_words.pkl', 'rb') as file:
            self.words = pickle.load(file)
        with open(f'{model_name if model_name is not None else self.model_name}_classes.pkl', 'rb') as file:
            self.classes = pickle.load(file)
        self.model = load_model(f'{self.model_name}.h5')

    def _clean_up_sentence(self, sentence: Text) -> List[Text]:
        """
        The _clean_up_sentence function takes a sentence as input and returns a list of the words in the
        sentence. The function also makes all the words lowercase, and lemmatizes them.

        :param self: Allow the function to reference attributes of the class in which it is defined
        :param sentence: str: Pass the sentence that needs to be tokenized and lemmatized
        :return: A list of words
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence: Text, words: Text):
        """
        The _bag_of_words function takes in a sentence and a list of words, cleans up the sentence
        and creates a bag of words using those cleaned words. It returns an array with 1's at the indices
        where those words are found.

        :param self: Reference the class itself
        :param sentence: str: Pass the sentence that we want to classify
        :param words: str: Pass the words we want to use for training
        :return: A list of 1s and 0s
        """
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence: Text) -> List[Dict[Text, Union[Text, List[Text]]]]:
        """
        The _predict_class function takes a sentence and returns the intent class with the highest probability.
        The function first tokenizes the sentence into words, then it creates a bag of words from all training examples.
        It calculates probabilities for every intent based on similarity between input and training sentences, using cosine similarity.
        The most probable class is returned as a list of dictionaries containing each intent name and its probability.

        :param self: Reference the class itself
        :param sentence: str: Pass the user's question to the _bag_of_words function
        :return: A list of predictions
        """
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': Text(r[1])})
        return return_list

    def _get_response(self, ints: list, intents_json: Dict) -> Text:
        """
        The _get_response function is a helper function that takes in two parameters:
            - ints: the list of intent classes returned by the classifier
            - intents_json: a json file containing all possible intents and responses for this bot.

          The _get_response function then returns the response string from one of those two sources,
          depending on which is easier to access given its current use.
          If an intent is found, it will return a random response from that intent's list of responses.
          If no intent is found, it will raise an error and return &quot;I don't understand you!

        :param self: Allow the method to refer to itself
        :param ints: list: Get the intent of the user input
        :param intents_json: dict: Get the list of intents from the json file
        :return: The response of the chatbot for a given input
        """
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            warn("Cannot find any trained intent for this text", NoTrainedIntent)
        return result or "I don't understand you!"

    def request(self, message: Text) -> Text:
        """
        The request function is used to send a message to the NLTP and receive an answer.
        It takes one argument, which is the message you want to send it.
        The request function returns the response of your message.

        :param self: Reference the class instance
        :param message: str: Pass the message to nltp
        :return: The answer of the nltp
        """
        ints = self._predict_class(message)

        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            response = self._get_response(ints, self.intents)
            if self.voice_saving:
                tts = gTTS(text=response,
                           lang=list(langs._langs.keys())[
                               list(langs._langs.values()).index(self.language.capitalize())])
                tts.save(f"{self.model_name}_answer.mp3")
            if self.voice_assistant:
                VoiceAssistant(response).say()
            else:
                return response
