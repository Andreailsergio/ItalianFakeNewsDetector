# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:05:11 2021

@author: andrea_sergiacomi

source: https://machinelearningknowledge.ai/nltk-tokenizer-tutorial-with-word_tokenize-sent_tokenize-whitespacetokenizer-wordpuncttokenizer/
"""

# pip install nltk
# (Requirement already satisfied)
import nltk
# nltk.download('all')


# Character Tokenization
text="Hello world"
lst=[x for x in text]
print(lst,"\n")
# ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']


# Word Tokenization
from nltk.tokenize import word_tokenize
text="Hello there! Welcome to the programming world."
print(word_tokenize(text))
# ['Hello', 'there', '!', 'Welcome', 'to', 'the', 'programming', 'world', '.']
text2="We are learning Natural Language Processing."
print(word_tokenize(text2),"\n")
# ['We', 'are', 'learning', 'Natural', 'Language', 'Processing', '.']


# Sentence Tokenization
from nltk.tokenize import sent_tokenize
text3="It’s easy to point out someone else’s mistake. Harder to recognize your own."
print(sent_tokenize(text3))
# ['It’s easy to point out someone else’s mistake.', 'Harder to recognize your own.']
text4="Laughing at our mistakes can lengthen our own life. Laughing at someone else's can shorten it."
print(sent_tokenize(text4),"\n")
# ['Laughing at our mistakes can lengthen our own life.', "Laughing at someone else's can shorten it."]


# Whitespace (space, tab, newline) Tokenization
from nltk.tokenize import WhitespaceTokenizer
s="Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
Tokenizer=WhitespaceTokenizer()
print(Tokenizer.tokenize(s),"\n")
# ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']


# Word Punctuation Tokenization
from nltk.tokenize import WordPunctTokenizer
text5="We're moving to L.A.!"
Tokenizer=WordPunctTokenizer()
print(Tokenizer.tokenize(text5),"\n")
# ['We', "'", 're', 'moving', 'to', 'L', '.', 'A', '.!']


# Removing Punctuations
from nltk.tokenize import RegexpTokenizer
text6="The children - Pierre, Laura, and Ashley - went to the store."
tokenizer = RegexpTokenizer(r"\w+")
lst2=tokenizer.tokenize(text6)
print(' '.join(lst2),"\n")
# The children Pierre Laura and Ashley went to the store


# Tokenization Dataframe Columns
import pandas as pd
# from nltk.tokenize import word_tokenize

df = pd.DataFrame({'Phrases': ['The greatest glory in living lies not in never falling, but in rising every time we fall.', 
                              'The way to get started is to quit talking and begin doing.', 
                              'If life were predictable it would cease to be life, and be without flavor.',
                              "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success."]})
df['tokenized'] = df.apply(lambda row: nltk.word_tokenize(row['Phrases']), axis=1)
print(df.head(),"\n")
'''
	Phrases	tokenized
0	The greatest glory in living lies not in never…	[The, greatest, glory, in, living, lies, not, …
1	The way to get started is to quit talking and …	[The, way, to, get, started, is, to, quit, tal…
2	If life were predictable it would cease to be …	[If, life, were, predictable, it, would, cease…
3	If you set your goals ridiculously high and it…	[If, you, set, your, goals, ridiculously, high…
'''


# NLTK Tokenize vs Split
'''
The split function is usually used to separate strings with a specified delimiter, 
e.g. the TAB (\t), the NEWLINE (\n) or a specific character
'''
# from nltk.tokenize import word_tokenize
text7="This is the first line of text.\nThis is the second line of text."
print("1.",text7)                   #Printing the two lines as they are
print("2.",text7.split('\n'))       #Splitting the text by '\n'.
print("3.",text7.split('\t'))       #Splitting the text by '\t'.
print("4.",text7.split('s'))        #Splitting by character 's'.
print("5.",text7.split())           #Splitting the text by space.
print("6.",word_tokenize(text7))    #Tokenizing by using word
'''
1. This is the first line of text.
This is the second line of text.
2. ['This is the first line of text.', 'This is the second line of text.']
3. ['This is the first line of text.\nThis is the second line of text.']
4. ['Thi', ' i', ' the fir', 't line of text.\nThi', ' i', ' the ', 'econd line of text.']
5. ['This', 'is', 'the', 'first', 'line', 'of', 'text.', 'This', 'is', 'the', 'second', 'line', 'of', 'text.']
6. ['This', 'is', 'the', 'first', 'line', 'of', 'text', '.', 'This', 'is', 'the', 'second', 'line', 'of', 'text', '.']
'''



'''
F I N A L   O U T P U T
runfile('C:/Users/andrea_sergiacomi/Desktop/HOMEworks/Python+ML/FAKE_detector/NLTK_tokenize.py', wdir='C:/Users/andrea_sergiacomi/Desktop/HOMEworks/Python+ML/FAKE_detector')

['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'] 

['Hello', 'there', '!', 'Welcome', 'to', 'the', 'programming', 'world', '.']
['We', 'are', 'learning', 'Natural', 'Language', 'Processing', '.'] 

['It’s easy to point out someone else’s mistake.', 'Harder to recognize your own.']
['Laughing at our mistakes can lengthen our own life.', "Laughing at someone else's can shorten it."] 

['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.'] 

['We', "'", 're', 'moving', 'to', 'L', '.', 'A', '.!'] 

The children Pierre Laura and Ashley went to the store 

                                             Phrases                                          tokenized
0  The greatest glory in living lies not in never...  [The, greatest, glory, in, living, lies, not, ...
1  The way to get started is to quit talking and ...  [The, way, to, get, started, is, to, quit, tal...
2  If life were predictable it would cease to be ...  [If, life, were, predictable, it, would, cease...
3  If you set your goals ridiculously high and it...  [If, you, set, your, goals, ridiculously, high... 

1. This is the first line of text.
This is the second line of text.
2. ['This is the first line of text.', 'This is the second line of text.']
3. ['This is the first line of text.\nThis is the second line of text.']
4. ['Thi', ' i', ' the fir', 't line of text.\nThi', ' i', ' the ', 'econd line of text.']
5. ['This', 'is', 'the', 'first', 'line', 'of', 'text.', 'This', 'is', 'the', 'second', 'line', 'of', 'text.']
6. ['This', 'is', 'the', 'first', 'line', 'of', 'text', '.', 'This', 'is', 'the', 'second', 'line', 'of', 'text', '.']
'''