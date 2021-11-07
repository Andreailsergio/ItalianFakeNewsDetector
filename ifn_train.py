# How I Created a Fake News Detector with Python (Spacy and Streamlit)
# https://towardsdatascience.com/how-i-created-a-fake-news-detector-with-python-65b1234123c4 by Giannis Tolios
# GITHUB repository https://github.com/derevirn/gfn-detector

import pandas as pd
import spacy #The code is not compatible with spaCy v3, please use v2 instead.
# ERRORE con pip install spacy==2.3.7
# pip install spacy-2.3.7-cp38-cp38-win_amd64.whl
# it_core_news_sm: ["2.3.0"]
# it_core_news_md: ["2.3.0"]
# it_core_news_lg: ["2.3.0"]
# python -m spacy download it_core_news_md (da ANACONDA command CMD.exe prompt)
from spacy.lang.it.examples import sentences
from spacy.util import minibatch, compounding
# https://spacy.io/models/it
import random

# TEST ITALIAN SPACY MODEL
nlp_test = spacy.load("C:/Users/andrea_sergiacomi/anaconda3/Lib/site-packages/it_core_news_sm/it_core_news_sm-2.3.0")
doc = nlp_test(sentences[0])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)

''' shuffling the data, assigning a category to each new article, splitting the dataset into train & test subsets  '''
def load_data(train_data, limit=0, split=0.8):
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"REAL": not bool(y), "FAKE": bool(y)} for y in labels]
    split = int(len(train_data) * split)
    
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

''' calculating metrics (precision, recall, F-score) to evaluate the text classifier performance '''
def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "FAKE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

''' loading the SpaCy pre-trained model 
NLP library supporting more than 60 languages 
including components for named entity recognition (NER), part-of-speech tagging, 
sentence segmentation, text classification, lemmatization, morphological analysis '''  
nlp = spacy.load('C:/Users/andrea_sergiacomi/anaconda3/Lib/site-packages/it_core_news_md/it_core_news_md-2.3.0')
# pacchetto it 3.0 va bene con spacy 2.0?
# nlp = spacy.load('el_core_news_md')

''' loading the Italian Fake News GFN dataset to a Pandas dataframe  '''
df = pd.read_csv('data/italian_fake_news.csv')
# df = pd.read_csv('data/greek_fake_news.csv')
df.replace(to_replace='[\n\r\t]', value=' ', regex=True, inplace=True)

''' training the textcat component with the GFN dataset  '''
textcat=nlp.create_pipe( "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
nlp.add_pipe(textcat, last=True)
nlp.pipe_names

textcat.add_label("REAL")
textcat.add_label("FAKE")

df['tuples'] = df.apply(lambda row: (row['text'], row['is_fake']), axis=1)
train = df['tuples'].tolist()

# calling the load_data() function to load the dataset
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(train, split=0.9)
train_data = list(zip(train_texts,[{'cats': cats} for cats in train_cats]))

print('\n',' *** TOTAL DATA ***','\n')
print(5+int(len(train_cats)/0.9),' records \n')
print('\n',' *** TRAINING DATA ***','\n')
# print(train_texts,'\n')
# print(train_cats,'\n')
print(len(train_cats),' records \n')
n_fake=0
n_good=0

for elemento in train_cats:
    for k,v in elemento.items():
        # print(k,v)
        if (k=='REAL' and v is True):
            n_good=n_good+1
        if (k=='FAKE' and v is True):
            n_fake=n_fake+1
print("TRAIN DATASET - Fake news: ",n_fake," Good news: ",n_good,'\n')


n_iter = 20
# Disabling other components
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
   
    print("Training the model...")
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))

    # Performing training
    for i in range(n_iter):
        losses = {}
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                       losses=losses)

      # Calling the evaluate() function and printing the scores
        with textcat.model.use_params(optimizer.averages):
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  
              .format(losses['textcat'], scores['textcat_p'],
                      scores['textcat_r'], scores['textcat_f']))

''' once completed the training, the model is saved with the to_disk() function  '''
with nlp.use_params(optimizer.averages):
            nlp.to_disk('model')