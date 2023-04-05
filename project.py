import spacy
from spacy import displacy
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
# import gtts   
from playsound import playsound  
# import pyttsx3
import os
import speech_recognition as sr
import pyaudio
import wave
import IPython.display as ipd
from IPython.display import Audio
from gtts import gTTS
import time
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from flashtext import KeywordProcessor

NER = spacy.load("en_core_web_sm")

def output_speech(mytext):
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("chat_speech.mp3")
    os.system("chat_speech.mp3")
    print("RECRUITER:",mytext)

def input_speech():
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "input.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    sound = AudioSegment.from_wav("input.wav")
    sound = sound.set_channels(1)
    sound.export("input.wav", format="wav")
    r = sr.Recognizer()
    with sr.WavFile("input.wav") as source:
        audio_data = r.listen(source)
        # convert speech to text
        text = r.recognize_google(audio_data)
        print("ME:",text)
    sentiment = predict_sentiment(text)
    return audio_data, predict_emo('input.wav')-sentiment

def place_info(P):
    place_d = {'Hyderabad':"it's famous for it's biryani !" }
    return place_d[P]

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype = "float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result

emotions = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry',
            '06':'fearful','07':'disgust','08':'surprised' }

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']
def load_data(test_size=0.2):
    x,y = [],[]
    for file in glob.glob("C:/Users/lione/OneDrive/Documents/SER/Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=101)

X_train, X_test, y_train, y_test = load_data(test_size=0.25)

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                      hidden_layer_sizes=(300,200,100), learning_rate='adaptive', max_iter=500)
model.fit(X_train,y_train)
model
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

def predict_emo(file):
    x = []
    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    np.array(x)
    l = model.predict(x)
    s = ''
    s.join(l)
    dec = -100
    if s in ['sad','fearful']:
        dec = 100
    if s in ['disgust','angry']:
        dec = 200
    return dec
place_info("Hyderabad")

def personal_round():
    r = sr.Recognizer()
    points_scored = 1000 #points keep decreasing based on answers by the candidate
    
    #question 1
    mytext = "Hello, let's start with some personal details about you. What is your name, where are you from and where did u work earlier?"
    language = 'en'
    output_speech(mytext)
    time.sleep(11)
    audio_data, emote_points = input_speech()  
    points_scored -= int(emote_points)
    text1 = NER(r.recognize_google(audio_data))
    for word in text1.ents:
        if word.label_=="PERSON":
            name=word.text
        if word.label_=="GPE":
            place=word.text
        if word.label_=="ORG":
            prev=word.text
    N="Ayush"        
    mytext = "Hi " + N + ", all the best for your interview!"
    output_speech(mytext)
    time.sleep(5)
    
    mytext = N + ", What are some of your strengths?"
    output_speech(mytext)
    time.sleep(6)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    strenghts = ['enthusiasm','enthusiastic','trustworthiness','trustworthy','creativity','creative','discipline','disciplined','patience','patient','respectfulness','respectfull','determination','determined','dedication','dedicated','honesty','honest','versatility','versatile','liveliness','lively','hard working']
    keywordprocessor = KeywordProcessor(case_sensitive=False)
    keywordprocessor.add_keywords_from_list(strenghts)
    Extractedkeywords = keywordprocessor.extract_keywords(r.recognize_google(audio_data))
    points_scored += len(keywordprocessor)*100
    mytext = N + ", What are some of your weaknesses?"
    output_speech(mytext)
    time.sleep(6)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    keywordprocessor = KeywordProcessor(case_sensitive=False)
    keywordprocessor.add_keywords_from_list(strenghts)
    Extractedkeywords = keywordprocessor.extract_keywords(r.recognize_google(audio_data))
    points_scored -= len(keywordprocessor)*50
    
    P = "Hyderabad"
    mytext = "Tell me about " + P + ",how is it there and what is " + P + "famous for." + " Heard " + place_info(place)
    output_speech(mytext)
    time.sleep(6)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    
    prev = "Samsung"
    mytext = "Why did you leave from " + prev
    output_speech(mytext)
    time.sleep(6)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    
    mytext = "Your score till now is" + str(points_scored)
    output_speech(mytext)
    time.sleep(6)
    
    mytext = "Let's move to some technical questions!"
    output_speech(mytext)
    time.sleep(6)

def technical_round(points_scored):
    
    
    #question 1
    mytext = "What is a doubly linked list and specify its applications?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 1
    answer = "This is a complex type of a linked list wherein a node has two references: One that connects to the next node in the sequence Another that connects to the previous node. This structure allows traversal of the data elements in both directions (left to right and vice versa). Applications of DLL are: A music playlist with next song and previous song navigation options. The browser cache with BACK-FORWARD visited pages The undo and redo functionality on platforms such as word, paint etc, where you can reverse the node to get to the previous page."
    
    #question 2
    mytext = "What is a priority queue?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 2
    answer = " A priority queue is an abstract data type that is like a normal queue but has priority assigned to elements. Elements with higher priority are processed before the elements with a lower priority. In order to implement this, a minimum of two queues are required - one for the data and the other to store the priority."
    
    #question 3
    mytext = "What is a AVL tree?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 3
    answer = "AVL trees are height balancing BST. AVL tree checks the height of left and right sub-trees and assures that the difference is not more than 1. This difference is called Balance Factor and is calculates as. BalanceFactor = height(left subtree) − height(right subtree)"
    
    #question 4
    mytext = "What is a heap?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 4
    answer = "Heap is a special tree-based non-linear data structure in which the tree is a complete binary tree. A binary tree is said to be complete if all levels are completely filled except possibly the last level and the last level has all elements towards as left as possible."
    
    #question 5
    mytext = "What is the difference between BFS and DFS?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 5
    answer = "BFS and DFS both are the traversing methods for a graph. Graph traversal is nothing but the process of visiting all the nodes of the graph. The main difference between BFS and DFS is that BFS traverses level by level whereas DFS follows first a path from the starting to the end node, then another path from the start to end, and so on until all nodes are visited. Furthermore, BFS uses queue data structure for storing the nodes whereas DFS uses the stack for traversal of the nodes for implementation. DFS yields deeper solutions that are not optimal, but it works well when the solution is dense whereas the solutions of BFS are optimal. You can learn more about BFS here: Breadth First Search and DFS here: Depth First Search."
    
    #question 6
    mytext = "What is your work experience such as internship and freelancing?"
    output_speech(mytext)
    time.sleep(8)
    audio_data, emote_points = input_speech()
    points_scored += emote_points
    #answer 5
    companies_pos = ["apple","samsung","foxconn","alphabet", "microsoft" ,"huawei","dell","hitachi", "amazon","flipkart","atlassian","google","oracle","fiverr","fiver"]
    companies_neg = ["no","sorry","not","couldn't"]
    keywordprocessor = KeywordProcessor(case_sensitive=False)
    keywordprocessor.add_keywords_from_list(companies_pos)
    Extractedkeywords = keywordprocessor.extract_keywords(r.recognize_google(audio_data))
    points_scored += len(keywordprocessor)*50
    keywordprocessor = KeywordProcessor(case_sensitive=False)
    keywordprocessor.add_keywords_from_list(companies_neg)
    Extractedkeywords = keywordprocessor.extract_keywords(r.recognize_google(audio_data))
    points_scored -= len(keywordprocessor)*50
    
import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
#from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

train = pd.read_csv('C:/Users/lione/OneDrive/Documents/sentiment/train.csv')
train.head(15)
len(train)
train['sentiment'].unique()
train = train[['selected_text','sentiment']]
train["selected_text"].fillna("No content", inplace = True)
def depure_data(data):
    
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
        
    return data

temp = []
#Splitting pd.Series to list
data_to_list = train['selected_text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
list(temp[:5])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
        # deacc=True removes punctuations
        

data_words = list(sent_to_words(temp))

print(data_words[:10])
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])
data = np.array(data)
labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'neutral':
        y.append(0)
    if labels[i] == 'negative':
        y.append(1)
    if labels[i] == 'positive':
        y.append(2)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
del y
from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)
X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)
print (len(X_train),len(X_test),len(y_train),len(y_test))
model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model2.add(layers.Dense(3,activation='softmax'))
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=5,validation_data=(X_test, y_test),callbacks=[checkpoint2])

best_model = keras.models.load_model("best_model2.hdf5")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)
predictions = best_model.predict(X_test)
# predictions
# X_test.shape
def predict_sentiment(x):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    x = url_pattern.sub(r'', x)
    # Remove Emails
    x = re.sub('\S*@\S*\s?', '', x)
    # Remove new line characters
    x = re.sub('\s+', ' ', x)
    # Remove distracting single quotes
    x = re.sub("\'", "", x)
    
    
    temp=[]
    temp.append(x)
    data_words = list(sent_to_words(temp))
    
    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))
    
    data = np.array(data)
    max_words = 5000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    num_val = pad_sequences(sequences, maxlen=max_len)
    
    y = best_model.predict(num_val)
    if max(y[0]) == y[0][0]:
        return 25
    if max(y[0]) == y[0][1]:
        return -50
    if max(y[0]) == y[0][2]:
        return 50
predict_sentiment("I`d have responded, if I were going")
import gensim
from gensim.models import KeyedVectors
import os
sentence_1 = "Obama speaks to the media in Illinois"
sentence_2 = "President greets the press in Chicago"
sentence_3 = "Nokia is my favorite company"

s1='my name is ayush'
s2='ayush is the name'
personal_round()