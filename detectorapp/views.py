# <-- without using scikit-learn ( lighter variation ) -->

from django.shortcuts import render
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import math

# Initialize the chat words dictionary (same as your current one)
chat_words = {
    "AFAIK": "As Far As I Know", "AFK": "Away From Keyboard",  # (your existing words here)
    # Add the rest of your chat words...
}

# Chat word conversion function
def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

# Stemming function
def stem_words(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

# Naive Bayes classifier implementation
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = {}
        self.class_probabilities = {}
        self.word_probabilities = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()

    def train(self, X, y):
        total_docs = len(X)
        class_counts = defaultdict(int)

        for i, doc in enumerate(X):
            label = y[i]
            class_counts[label] += 1
            for word in doc.split():
                self.vocabulary.add(word)
                self.word_probabilities[label][word] += 1

        for label in class_counts:
            self.class_probabilities[label] = class_counts[label] / total_docs
            total_words_in_class = sum(self.word_probabilities[label].values()) + len(self.vocabulary)

            for word in self.vocabulary:
                self.word_probabilities[label][word] = (self.word_probabilities[label][word] + 1) / total_words_in_class

    def predict(self, text):
        text_words = text.split()
        log_probabilities = defaultdict(float)

        for label in self.class_probabilities:
            log_probabilities[label] = math.log(self.class_probabilities[label])
            for word in text_words:
                log_probabilities[label] += math.log(self.word_probabilities[label].get(word, 1 / len(self.vocabulary)))

        return max(log_probabilities, key=log_probabilities.get)

def home(request):
    # Load dataset
    df = pd.read_csv('dataset.csv')
    df.drop_duplicates(inplace=True)
    df['Message'] = df['Message'].str.lower()
    df['Message'] = df['Message'].str.replace('#', '')
    df['Message'] = df['Message'].str.replace('@', '')
    df['Message'] = df['Message'].str.replace(r'^https?:\/\/.*[\r\n]*', '', regex=True)
    df['Message'] = df['Message'].str.translate(str.maketrans('', '', string.punctuation))

    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    df['Message'] = df['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['Message'] = df['Message'].apply(chat_conversion)

    nltk.download('punkt')
    df['stem_msg'] = df['Message'].apply(stem_words)

    # Text preprocessing and label encoding
    X = df['stem_msg'].values.tolist()
    y = df['Category'].values.tolist()

    # Split into training and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train Naive Bayes classifier
    model = NaiveBayesClassifier()
    model.train(X_train, y_train)

    result = None
    if request.method == 'POST':
        email_text = request.POST.get('email_text', '')
        if email_text:
            # Preprocess the input text (same steps as training)
            input_text = email_text.lower()
            input_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', input_text)
            input_text = input_text.translate(str.maketrans('', '', string.punctuation))
            input_text = ' '.join([word for word in input_text.split() if word not in stop_words])
            input_text = chat_conversion(input_text)
            input_text_stemmed = stem_words(input_text)

            # Predict the category
            prediction = model.predict(input_text_stemmed)
            result = prediction

    # Pass results to the template
    context = {'result': result}
    return render(request, 'index.html', context)
from django.shortcuts import render

# <-- Alternate variation using scikit-learn -->
# <-- pip install scikit-learn -->

# from django.shortcuts import render
#
#
# import pandas as pd
# import string
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
#
# # Initialize the chat words dictionary
# chat_words = {
#     "AFAIK": "As Far As I Know",
#     "AFK": "Away From Keyboard",
#     "ASAP": "As Soon As Possible",
#     "ATK": "At The Keyboard",
#     "ATM": "At The Moment",
#     "A3": "Anytime, Anywhere, Anyplace",
#     "BAK": "Back At Keyboard",
#     "BBL": "Be Back Later",
#     "BBS": "Be Back Soon",
#     "BFN": "Bye For Now",
#     "B4N": "Bye For Now",
#     "BRB": "Be Right Back",
#     "BRT": "Be Right There",
#     "BTW": "By The Way",
#     "B4": "Before",
#     "B4N": "Bye For Now",
#     "CU": "See You",
#     "CUL8R": "See You Later",
#     "CYA": "See You",
#     "FAQ": "Frequently Asked Questions",
#     "FC": "Fingers Crossed",
#     "FWIW": "For What It's Worth",
#     "FYI": "For Your Information",
#     "GAL": "Get A Life",
#     "GG": "Good Game",
#     "GN": "Good Night",
#     "GMTA": "Great Minds Think Alike",
#     "GR8": "Great!",
#     "G9": "Genius",
#     "IC": "I See",
#     "ICQ": "I Seek you (also a chat program)",
#     "ILU": "ILU: I Love You",
#     "IMHO": "In My Honest/Humble Opinion",
#     "IMO": "In My Opinion",
#     "IOW": "In Other Words",
#     "IRL": "In Real Life",
#     "KISS": "Keep It Simple, Stupid",
#     "LDR": "Long Distance Relationship",
#     "LMAO": "Laugh My A.. Off",
#     "LOL": "Laughing Out Loud",
#     "LTNS": "Long Time No See",
#     "L8R": "Later",
#     "MTE": "My Thoughts Exactly",
#     "M8": "Mate",
#     "NRN": "No Reply Necessary",
#     "OIC": "Oh I See",
#     "PITA": "Pain In The A..",
#     "PRT": "Party",
#     "PRW": "Parents Are Watching",
#     "QPSA?": "Que Pasa?",
#     "ROFL": "Rolling On The Floor Laughing",
#     "ROFLOL": "Rolling On The Floor Laughing Out Loud",
#     "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
#     "SK8": "Skate",
#     "STATS": "Your sex and age",
#     "ASL": "Age, Sex, Location",
#     "THX": "Thank You",
#     "TTFN": "Ta-Ta For Now!",
#     "TTYL": "Talk To You Later",
#     "U": "You",
#     "U2": "You Too",
#     "U4E": "Yours For Ever",
#     "WB": "Welcome Back",
#     "WTF": "What The F...",
#     "WTG": "Way To Go!",
#     "WUF": "Where Are You From?",
#     "W8": "Wait...",
#     "7K": "Sick:-D Laugher",
#     "TFW": "That feeling when",
#     "MFW": "My face when",
#     "MRW": "My reaction when",
#     "IFYP": "I feel your pain",
#     "TNTL": "Trying not to laugh",
#     "JK": "Just kidding",
#     "IDC": "I don't care",
#     "ILY": "I love you",
#     "IMU": "I miss you",
#     "ADIH": "Another day in hell",
#     "ZZZ": "Sleeping, bored, tired",
#     "WYWH": "Wish you were here",
#     "TIME": "Tears in my eyes",
#     "BAE": "Before anyone else",
#     "FIMH": "Forever in my heart",
#     "BSAAW": "Big smile and a wink",
#     "BWL": "Bursting with laughter",
#     "BFF": "Best friends forever",
#     "CSL": "Can't stop laughing"
#
# }
#
# def chat_conversion(text):
#     new_text = []
#     for i in text.split():
#         if i.upper() in chat_words:
#             new_text.append(chat_words[i.upper()])
#         else:
#             new_text.append(i)
#     return " ".join(new_text)
#
# def stem_words(text):
#     stemmer = PorterStemmer()
#     return " ".join([stemmer.stem(word) for word in text.split()])
#
# def home(request):
#     # Load the dataset
#     df = pd.read_csv('dataset.csv')  # Update the path as needed
#
#     # Preprocessing steps
#     df.drop_duplicates(inplace=True)
#     df['Message'] = df['Message'].str.lower()
#     df['Message'] = df['Message'].str.replace('#', '')
#     df['Message'] = df['Message'].str.replace('@', '')
#     df['Message'] = df['Message'].str.replace(r'^https?:\/\/.*[\r\n]*', '', regex=True)
#     df['Message'] = df['Message'].str.translate(str.maketrans('', '', string.punctuation))
#
#     nltk.download('stopwords')
#     stop_words = stopwords.words('english')
#     df['Message'] = df['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#     df['Message'] = df['Message'].apply(chat_conversion)
#
#     nltk.download('punkt')
#     nltk.download('wordnet')
#     df['text_sent_token'] = df['Message'].apply(sent_tokenize)
#     df['stem_msg'] = df['Message'].apply(stem_words)
#
#     # Text Representation
#     cv = CountVectorizer()
#     X = cv.fit_transform(df['stem_msg']).toarray()
#
#     # Encoding y
#     le = LabelEncoder()
#     y = le.fit_transform(df['Category'])
#
#     # Train-Test Split
#     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0)
#
#     # Model Training
#     models = {
#         'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#         'SVC': SVC(),
#         'Decision Tree': DecisionTreeClassifier(random_state=42),
#         'Random Forest': RandomForestClassifier(random_state=42)
#     }
#
#     model = LogisticRegression(max_iter=1000, random_state=42)  # Choose one model for prediction
#     model.fit(X_train, y_train)
#
#     # Handle form submission
#     result = None
#     if request.method == 'POST':
#         email_text = request.POST.get('email_text', '')
#         if email_text:
#             # Preprocess the input text (same steps as training)
#             input_text = email_text.lower()
#             input_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', input_text)
#             input_text = input_text.translate(str.maketrans('', '', string.punctuation))
#             input_text = ' '.join([word for word in input_text.split() if word not in stop_words])
#             input_text = chat_conversion(input_text)
#             input_text_stemmed = stem_words(input_text)
#
#             # Convert input text to vector
#             input_vector = cv.transform([input_text_stemmed]).toarray()
#
#             # Predict
#             prediction = model.predict(input_vector)
#             result = le.inverse_transform(prediction)[0]  # Get the category name
#
#     # Pass results to the template
#     context = {
#         'result': result
#     }
#
#     return render(request, 'index.html', context)
#




