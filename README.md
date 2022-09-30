# neural_chat

## Setting Up A Basic Assistant

```python
from neural_chat import GenericAssistant

assistant = GenericAssistant("intents.json",
                             model_name="test_model")
assistant.train_model()
assistant.save_model()

while True:
    message = input("Enter a message: ").lower()
    if message == "stop":
        break
    else:
        assistant.request(message)
```

## Also support a voice assistant

```python
from neural_chat import GenericAssistant

assistant = GenericAssistant("intents.json",
                             model_name="test_model",
                             voice_assistant=True)
assistant.train_model()
assistant.save_model()

while True:
    message = input("Enter a message: ").lower()
    if message == "stop":
        break
    else:
        response = assistant.request(input(">>> "))
        if type(response) == str:
            print(response)
        # else: will speak the answer text and print it.
        #       Print function hardcoded in voice_assistant.py
```

## Binding Functions To Requests

```python
from neural_chat import GenericAssistant

def function_for_greetings():
    print("You triggered the greetings intent!")
    # Some action you want to take

def function_for_stocks():
    print("You triggered the stocks intent!")
    # Some action you want to take

mappings = {
    "greeting" : function_for_greetings,
    "stocks" : function_for_stocks
}

assistant = GenericAssistant("intents.json",
                             intent_methods=mappings,
                             model_name="test_model")
assistant.train_model()
assistant.save_model()

while True:
    message = input("Enter a message: ")
    if message == "stop":
        break
    else:
        assistant.request(message)
```

## Sample intents.json File

```json
{"intents": [
  {"tag": "greeting",
    "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up", "Hey", "greetings"],
    "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
    "context_set": ""
  },
  {"tag": "goodbye",
    "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "bye", "cao", "see ya"],
    "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"],
    "context_set": ""
  },
  {"tag": "stocks",
    "patterns": ["what stocks do I own?", "how are my shares?", "what companies am I investing in?", "what am I doing in the markets?"],
    "responses": ["You own the following shares: ABBV, AAPL, FB, NVDA and an ETF of the S&P 500 Index!"],
    "context_set": ""
  }
]
}
```