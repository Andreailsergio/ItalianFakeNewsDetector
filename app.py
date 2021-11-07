''' Streamlit is a Python framework to build web apps quickly for data science projects
deploying machine learning models to the web, and adding great visualizations of your data
with a powerful caching mechanism, that optimizes the performance of your app.
Streamlit Sharing is a service provided freely by the library creators, that lets you easily deploy and share your app
https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2  '''

''' we can execute the streamlit run app.py command to run the application locally, or use the free Streamlit Sharing service to deploy it  '''
# DA CMD.exe prompt (Anaconda3)
# streamlit run C:/Users/andrea_sergiacomi/Desktop/HOMEworks/Python+ML/FAKE_detector/app.py

# pip install streamlit
import streamlit as st
import matplotlib.pyplot as plt
import spacy
from spacy.lang.el.stop_words import STOP_WORDS
# pip install wordcloud
from wordcloud import WordCloud
# da utils.py
from utils import get_page_text

# st.set_page_config(page_title = "Italian Fake News Detector")
st.write("Italian Fake News Detector")

# st.cache decorator lets Streamlit store the model in a local cache to improve performance
@st.cache(allow_output_mutation=True)
# function to load the Spacy text classification model - trained within the GFN_train.py
def get_nlp_model(path):   
    return spacy.load(path)

# function that prints the classification result using markdown() & HTML tags
def generate_output(text):
     cats = nlp(text).cats
     if cats['FAKE'] > cats['REAL']:
         st.markdown("<h1><span style='color:red'>This is a fake news article!</span></h1>",
                     unsafe_allow_html=True)
     else:
         st.markdown("<h1><span style='color:green'>This is a real news article!</span></h1>",
                     unsafe_allow_html=True)
             
     q_text = '> '.join(text.splitlines(True))   
     q_text = '> ' + q_text
	 # the article text is printed
     st.markdown(q_text)

     wc = WordCloud(width = 1000, height = 600,
                    random_state = 1, background_color = 'white',
                    stopwords = STOP_WORDS).generate(text) 
     
     fig, ax = plt.subplots()
	 # the word cloud is printed
     ax.imshow(wc)
     ax.axis('off')
     st.pyplot(fig)
     print(cats)

nlp = get_nlp_model('C:/Users/andrea_sergiacomi/Desktop/HOMEworks/Python+ML/FAKE_detector/model')

desc = "This web app detects fake news written in the Italian language.\
        You can either enter the URL of a news article, or paste the text directly (works better).\
        This app was developed with the Streamlit and spacy Python libraries.\
        The Github repository of the app is available [here](https://github.com/Andreailsergio/).\
        Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/andreasergiacomi/)\
        or via [e-mail](mailto:andrea.sergiacomi@regione.marche.it)."

''' creating the application LAYOUT '''
# setting the page title and description
st.title("Italian Fake News Detector")
st.markdown(desc)
# the user can input the article URL or directly a text
# in both cases a button widget is used to call the generate_output() function
# to classify the article and print the result
st.subheader("Enter the URL address/text of a news article written in Italian")
select_input = st.radio("Select Input:", ["URL", "Text"])

if select_input == "URL":
    url = st.text_input("URL")   
    if st.button("Run"):
		# scraping the text
        text = get_page_text(url)
        generate_output(text)  

else:
    text = st.text_area("Text", height=300)
    if st.button("Run"):
        generate_output(text) 