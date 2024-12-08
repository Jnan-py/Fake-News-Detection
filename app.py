import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake["class"] = 0
data_true['class'] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data = data.sample(frac=1).reset_index(drop=True)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

def vectorize():
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    return vectorization, xv_train, xv_test

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0)
}

def train_model(model_name, model, xv_train, y_train):
    model.fit(xv_train, y_train)

def output_label(n):
    return "The given content is **Fake News**" if n == 0 else "The given content is **Not A Fake News**"

st.title("Fake News Detection App")

st.sidebar.title("Choose Classifier Model")
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

st.subheader("Enter the news content below to check if it's Fake or Not:")
user_input = st.text_area("News Content", height=150)

if st.button("Classify"):

    if user_input:
        st.subheader(f'Using {selected_model} algorithm')
        with st.spinner('Classifying....Please wait..'):
            v,tr,te = vectorize()
            processed_text = wordopt(user_input)
            input_vector = v.transform([processed_text])
            
            train_model(selected_model,  models[selected_model], tr, y_train)
            prediction = models[selected_model].predict(input_vector)[0]
            
            st.subheader("**Prediction:**")
            st.write(output_label(prediction))
        
    else:
        st.warning("Please enter news content to classify.")
