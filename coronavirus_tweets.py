# Part 3: Text mining.

import pandas as pd
import requests
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding = 'latin-1')
	return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return df['Sentiment'].value_counts().index[1]


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	return df[df['Sentiment'] == 'Extremely Positive']['TweetAt'].value_counts().index[0]


# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()
    return df


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z\s]', ' ', regex=True)
    return df


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ')
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()
	return df


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    return tdf['OriginalTweet'].explode().count()


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	return tdf['OriginalTweet'].explode().nunique()


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	return tdf['OriginalTweet'].explode().value_counts().index[:k].tolist()


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stop_words = requests.get('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').text.split()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [word for word in x if word not in stop_words])
	return tdf


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [stemmer.stem(word) for word in x])
	return tdf


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 

def mnb_predict(df):
    le = LabelEncoder()
    sentiments = df['Sentiment'].copy()
    df['Sentiment'] = le.fit_transform(df['Sentiment'])
    X = df['OriginalTweet']
    y = df['Sentiment']
    vectorizer = CountVectorizer(analyzer= 'word', binary = False, max_df = 0.9, max_features=None, min_df = 2, ngram_range = (1, 3), stop_words=None, strip_accents = None)
    X = vectorizer.fit_transform(X)
    mnb = MultinomialNB()
    mnb.fit(X, y)
    y_pred = mnb.predict(X)
    y_pred = le.inverse_transform(y_pred)
    df['Sentiment'] = sentiments
    return y_pred


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
	return round(accuracy_score(y_pred, y_true), 3)