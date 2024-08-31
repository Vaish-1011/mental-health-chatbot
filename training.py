import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
import random
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming 'training' is a list of pairs (X, Y)
random.shuffle(training)

# Separate features (X) and labels (Y)
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Pad sequences to a fixed length
train_x_padded = pad_sequences(train_x)

# Convert to NumPy arrays
train_x_np = np.array(train_x_padded)
train_y_np = np.array(train_y)

print("Training data created")
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Assuming you have created your train_x and train_y data

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons, and 3rd output layer contains number of neurons
# equal to the number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Fix the deprecated warning
tf.compat.v1.get_default_graph
from tensorflow.keras.optimizers import SGD

# Assuming you have a model defined as 'model'
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Assuming you have created your model here
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('my_model.keras', hist)
# Assuming you have a test set (test_x, test_y)
loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y))
print(f'Test Accuracy: {accuracy}')
print("model created")