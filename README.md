# ItalianFakeNewsDetector

IFN_train.py addestra il modello (riconoscimento fake news basato su un dataset di allenamento italian_fake_news.csv con oltre 1000 articoli)
(usa la libreria Spacy per l'elaborazione del linguaggio naturale)

APP.py lancia una web application basata su STREAMLIT che usa il modello di cui sopra per riconoscere se un articolo - dato il testo o un url - è verosimilmente veritiero o fake
