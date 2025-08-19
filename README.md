# Intelligent AI Web Chatbot

A complete end-to-end intelligent chatbot system with neural network training and web deployment. Features intent classification, natural language processing, and a responsive web interface built with Flask, TensorFlow, and modern web technologies.

## Author
**Macha Praveen**

## Overview

This project implements a sophisticated AI chatbot that understands user intents and provides appropriate responses. The system consists of two main components: a model training pipeline that creates a neural network classifier, and a Flask web application that serves the chatbot with a modern, interactive interface.

## Features

- **Intent Classification**: Neural network-based intent recognition with high accuracy
- **Natural Language Processing**: Advanced text preprocessing using NLTK and lemmatization
- **Modern Web Interface**: Responsive, chat-like UI with sliding animations
- **Real-time Communication**: AJAX-powered instant messaging
- **Model Persistence**: Trained models saved and loaded efficiently
- **Extensible Architecture**: Easy to add new intents and responses

## Architecture

### Model Training Pipeline
The system trains a neural network to classify user intents using:

**Text Preprocessing:**
```python
def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
```

**Bag of Words Vectorization:**
```python
def bag_of_words(sentence):
    words = pickle.load(open('model/words.pkl', 'rb'))
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
```

**Neural Network Architecture:**
```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
```

### Web Application
Flask-based web server with REST API and interactive frontend:

**Intent Prediction:**
```python
def predict_class(sentence):
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')
    
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list
```

**Response Generation:**
```python
def get_response(intents_list):
    intents_json = json.load(open('model/intents.json'))
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result
```

## Intent Configuration

The chatbot understands various intents defined in JSON format:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
      "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
      "context_set": ""
    },
    {
      "tag": "programming",
      "patterns": ["What is progamming?", "What is coding?", "Tell me about programming"],
      "responses": ["Programming, coding or software development, means writing computer code to automate tasks."],
      "context_set": ""
    }
  ]
}
```

Current supported intents:
- **Greeting**: Welcome messages and casual greetings
- **Goodbye**: Farewell messages
- **Programming**: Questions about programming and coding
- **Status**: Asking about the bot's wellbeing
- **Flask**: Questions about Flask framework

## Installation

### Prerequisites
- Python 3.7+
- Node.js (for frontend dependencies)
- NLTK data packages

### Dependencies Installation
```bash
# Install Python packages
pip install flask tensorflow keras nltk numpy pickle-mixin

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Setup
1. **Prepare the model** (first time only):
   ```bash
   cd "Intelligent AI Web Chatbot/Preparation"
   python model_training.py
   ```

2. **Start the web application**:
   ```bash
   cd "Intelligent AI Web Chatbot/Flask App"
   python app.py
   ```

3. **Access the chatbot**: Open http://0.0.0.0:5000 in your browser

## Usage

### Web Interface
1. **Open the application** in your web browser
2. **Click the chat icon** in the bottom-right corner
3. **Type your message** in the input field
4. **Press Enter or click Send** to get a response
5. **Continue the conversation** with the AI chatbot

### API Endpoint
The chatbot also provides a REST API endpoint:

```bash
curl -X POST http://127.0.0.1:5000/handle_message \
     -d '{"message":"what is coding"}' \
     -H "Content-Type: application/json"
```

Response:
```json
{
  "response": "Programming, coding or software development, means writing computer code to automate tasks."
}
```

### Training New Intents
1. **Edit intents.json** in the Preparation folder
2. **Add new intent patterns and responses**
3. **Retrain the model**:
   ```bash
   cd "Preparation"
   python model_training.py
   ```
4. **Copy updated model files** to Flask App/model/
5. **Restart the Flask application**

## Project Structure

```
Intelligent AI Web Chatbot/
├── README.md
├── Flask App/                      # Web application
│   ├── app.py                     # Flask server
│   ├── utils.py                   # NLP utilities
│   ├── model/                     # Trained models
│   │   ├── chatbot_model.keras    # Neural network model
│   │   ├── words.pkl              # Vocabulary
│   │   ├── classes.pkl            # Intent classes
│   │   └── intents.json           # Intent definitions
│   └── templates/
│       └── index.html             # Web interface
├── Preparation/                    # Model training
│   ├── model_training.py          # Training script
│   ├── chatbot.py                 # CLI version
│   └── intents.json               # Training data
└── model/                          # Training artifacts
    ├── chatbot_model.keras
    ├── words.pkl
    └── classes.pkl
```

## Technology Stack

### Backend
- **Flask**: Web framework for API and static file serving
- **TensorFlow/Keras**: Deep learning framework for neural network
- **NLTK**: Natural language processing toolkit
- **NumPy**: Numerical computing for data processing
- **Pickle**: Model serialization and persistence

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript/jQuery**: Interactive functionality
- **Font Awesome**: Icons and visual elements
- **Google Fonts**: Typography (Roboto)

### Machine Learning
- **Neural Network**: Sequential model with Dense layers
- **Dropout Regularization**: Prevents overfitting (0.5 dropout rate)
- **SGD Optimizer**: Stochastic gradient descent with momentum
- **Softmax Activation**: Multi-class probability distribution

## Model Performance

### Training Configuration
- **Architecture**: 2 hidden layers (128, 64 neurons)
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: SGD (lr=0.01, momentum=0.9, Nesterov=True)
- **Training**: 200 epochs, batch size 5
- **Regularization**: 50% dropout to prevent overfitting

### Performance Features
- **Intent Recognition**: High accuracy with confidence thresholds
- **Error Threshold**: 0.25 minimum confidence for predictions
- **Response Selection**: Random choice from appropriate responses
- **Lemmatization**: Improved text normalization
- **Model Persistence**: Fast loading of pre-trained models

## Future Enhancements

- **Context Management**: Multi-turn conversation support
- **User Authentication**: Personalized chat experiences
- **Database Integration**: Conversation history storage
- **Advanced NLP**: BERT or GPT integration
- **Voice Interface**: Speech-to-text and text-to-speech
- **Mobile App**: React Native or Flutter implementation
- **Analytics Dashboard**: Conversation insights and metrics

## Development Notes

- The model uses categorical crossentropy loss for multi-class classification
- Bag of words representation creates sparse feature vectors
- Intent confidence scoring helps handle uncertain predictions
- The web interface uses modern CSS for smooth animations
- AJAX communication provides seamless user experience

## License

This project is open-source and available under the MIT License.
