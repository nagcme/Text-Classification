'''
Module : 7CCSMDM1 DATA MINING
Coursework 2 : Activity 1 : Text Mining
Student ID : 20112906
Candidate ID : AB01910
References: https://pythonspot.com/nltk-stop-words/
            https://docs.python.org/3/library/re.html
            https://scikit-learn.org/stable/modules/generated/
            sklearn.naive_bayes.MultinomialNB.html
            https://scikit-learn.org/stable/modules/generated/
            sklearn.feature_extraction.text.CountVectorizer.html
            https://scikit-learn.org/stable/modules/generated/s
            klearn.feature_extraction.text.TfidfTransformer.html
Assumptions:
            1. The following file is present in the same directory as the
            python file:
               Corona_NLP_train.csv
            2. An 'outputs' folder will already be created in the current
            directory where the code and the data files are kept
'''

# Import statements
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import numpy as np

# Print the start time of the code execution
print(datetime.now())

# Read the Corona_NLP_train dataset in python dataframe
corona_df = pd.read_csv('Corona_NLP_train.csv', encoding = 'iso-8859-1')

# Question 1
print('Question 1...')
# List the possible sentiments that a tweet may have
print('List of all possible sentiments in the file: '
      +str(corona_df.Sentiment.unique()))

# Fetch the second most popular sentiment in the tweets
sentiment_freq = corona_df.Sentiment.value_counts()
second_pop = dict(sentiment_freq[1:2])
print('Second most popular sentiment: '+str(second_pop))

# Fetch the date with the greatest number of extremely positive tweets
ex_positive_tweet = corona_df[corona_df['Sentiment'] ==
                              'Extremely Positive'].TweetAt.value_counts()
ex_positive_tweet = dict(ex_positive_tweet[0:1])
print('Date with the greatest no. of extremely positive tweets: '
      +str(ex_positive_tweet))

# Text pre processing

# Convert the messages to lower case
corona_df["OriginalTweet"] = corona_df["OriginalTweet"].str.lower()

# replace non-alphabetical characters with whitespaces
corona_df["OriginalTweet_Cleaned"] = [re.sub("[^a-z]+", " ", str(x))
                                    for x in corona_df["OriginalTweet"]]
# check the words of a message are separated by a single whitespace
# corona_df["OriginalTweet_Cleaned"].replace(to_replace ="  ", value =" ")
corona_df["OriginalTweet_Cleaned"] = [re.sub(' +', ' ', str(x))
                                    for x in corona_df["OriginalTweet_Cleaned"]]

# Question 2
print('Question 2...')
print(datetime.now())
# Tokenise the tweets in the Dataframe using split()
corona_df["Tweet_tokenised"] = corona_df["OriginalTweet_Cleaned"].str.split()

# Count the total number of all words (including repetitions)
# First map the length of each tweet document and then sum the total
print('Total number of all words: '+str(sum(map(len,
                                                corona_df["Tweet_tokenised"]))))

# Count the total number of distinct words
results = set()
corona_df["Tweet_tokenised"].apply(results.update)
print('Total number of distinct words: '+str(len(results)))

# Calculate the word frequency
word_freq = pd.DataFrame(
    corona_df["Tweet_tokenised"].to_list()).stack().value_counts()

# Print the 10 most frequent words in the corpus
print('The 10 most frequent words in the corpus:')
print(word_freq.head(10))

# Stopwords Removal

# Fetch the list of stopwords from NLTK
stopwords_nltk = nltk.corpus.stopwords.words('english')

# Remove all stopwords and words having length less than or equal to 2 character
corona_df["Tweet_WO_SW"] = corona_df["Tweet_tokenised"].apply(lambda x:
                           [item for item in x if item not in stopwords_nltk
                                                          and len(item) > 2])

# Count the total number of all words (including repetitions)
# First map the length of each tweet document and then sum the total
print('Total number of all words: '+str(sum(map(len,
                                                corona_df["Tweet_WO_SW"]))))

# Count the total number of distinct words
corona_df['Tweet_WO_SW_Uniq'] = corona_df['Tweet_WO_SW'].apply(set)
results = set()
corona_df["Tweet_WO_SW"].apply(results.update)
print('Total number of distinct words: '+str(len(results)))

# Print the 10 most frequent words in the corpus
word_freq = pd.DataFrame(
    corona_df["Tweet_WO_SW"].to_list()).stack().value_counts()
print('The 10 most frequent words in the corpus:')
print(word_freq.head(10))

# Question 3
print('Question 3...')
print(datetime.now())
# Plot a line chart with word frequencies, where the horizontal axis corresponds
# to words, while the vertical axis indicates the fraction of documents in a
# which a word appears.
# The words should be sorted in increasing order of their frequencies

# Remove duplicate words from each row
corona_df['Tweet_WO_SW_Uniq'] = corona_df['Tweet_WO_SW'].apply(set)

# Fetch the frequency of occurrence of each word in a tweet
word_freq = pd.DataFrame(
    corona_df["Tweet_WO_SW_Uniq"].to_list()).stack().value_counts()

## Plot all the words having the below axes details:
# x-axis: word index
# y-axis: fraction of documents where the word is present
word_docs = (word_freq.sort_values())/len(corona_df)
word_docs = word_docs.reset_index()
word_docs.plot(legend=False)
plt.title('Word Frequencies')
plt.xlabel('Words')
plt.ylabel('Fraction of documents')
print('Saving the document frequency graph in the "outputs" folder...')
plt.savefig('outputs/word_freq_all.png')

# Commented the below code, this was for analysis part
# DId not remove the code, as the output graph is there in the report
# Start of comment
# # Plot only the top 20 words (sorted in increasing order of their frequencies)
# # This plot is used to get a better insight of the corpus
# word_docs_50 = word_freq.sort_values().tail(50)
# # Calculate the fraction of document where the word exists
# word_docs_50_frac = word_docs_50/len(corona_df)
#
# # Plot the graph
# word_docs_50_frac.plot(figsize=(12,14))
# tick_labels = tuple(word_docs_50_frac.index)
# plt.xticks(range(0, 50), tick_labels, rotation=90)
# plt.xlabel('Words')
# plt.ylabel('Fraction of documents')
# Save the graph
# plt.savefig('outputs/word_freq.png')
# End of comment

# Question 4
print('Question 4...')
print(datetime.now())
# Produce a Multinomial Naive Bayes classifier for the
# Coronavirus Tweets NLP data set using scikit-learn.

# Get an instance of the Count Vectorizer
# Count Vectorizer converts a collection of text documents
# to a matrix of token counts
vectorizer = CountVectorizer()

# Store the corpus in a numpy array
corpus_np = corona_df['OriginalTweet_Cleaned'].to_numpy()
# Fit and transform the corpus using vectorizer
# This returns a sparse matrix
tweets = vectorizer.fit_transform(corpus_np)

# Fitting the Multinominal Naive Bayes Theorem for the corpus
clf = MultinomialNB()
clf.fit(tweets, np.array(corona_df['Sentiment']))

# Calculate and print the Error Rate
# The data has not been split into training and test set
# Hence the entire data is used to calculate the error rate
print('Error Rate: '+str(1 - round(clf.score(tweets,np.array(corona_df[
                                                            'Sentiment'])),2)))

# Print the End time of code execution
print(datetime.now())