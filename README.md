# Text-Classification
Sentiment Analysis using Python
Versions:
python '3.8.5'
pandas '1.1.4'
sklearn '0.24.1'
matplotlib '3.3.2'
numpy '1.19.3'
skimage '0.18.1'

Files: The below files (python and data) should be placed in the same folder:
	text_mining.py
	Corona_NLP_train.csv
  
  
  *******************************************************************************************************
*******************************************TEXT MINING*************************************************
*******************************************************************************************************

Files:
text_mining.py
Corona_NLP_train.csv

Code Structure:
The Corona_NLP_train.csv file has been read into pandas dataframe for text mining.
1. a) All the possible sentiments have been fetched from the Sentiment column of the dataframe
   b) The frequency of each of the sentiments has been computed and the second most popular sentiment
has been returned.
   c) The date that received the maximum no. of 'Extremely Positive' tweets has been fetched by 
counting the rows with 'Extremely Positive' tweets by grouping the 'TweetAt' column. The first item
has been returned.
   d) Text Preprocessing: Data in the 'OriginalTweet' column is first converted to lower case
			  All non-alphabetical characters has been replaced with space
			  A check is performed to determine two words are separated only by a single 
			  space.
2. a) A new column is generated in the existing dataframe that consists of the tokenised tweets. The
tokenisation of tweets is achieved by using split() of python.
   b) First map the length of each tokenised tweet document and then sum the total to get the total
no. of words in the corpus 
   c) To fetch the total no. of distinct words, a set() is applied on the dataframe column to remove
the duplicate words. Then the length of the set is calculated to generate the total no. of distinct
words in the corpus.
   d) The word frequency of the corpus is calculated by using the functions to_list(), stack() and 
value_counts() and then the top 10 frequent words are displayed.
   e) The stopwords from nltk are fetched. All the words that belongs to the stopwords set and any word
that has length less than or equal to 2chars are removed to clean the corpus and the above steps are 
again performed.
3. Duplicate words from each document is first removed to generate the graph showing the fraction of
documents in which a word appears. Then the word frequency is generated and the line chart is plotted 
where Y axis shows the fraction of documents and X axis shows the word index. 
4. a) The preprocessed cleaned tweets are saved in a numpy array
   b) Use CountVectoriser to fit and transform the corpus into a sparse matrix of terms
   c) MultinomialNB() function from sklearn is used to create the classifier
   d) The sparse matrix computed using CountVectoriser is passed into the classifier along with the 
Sentiment column
   e) Calculate the Error Rate of the classifier

*******************************************************************************************************
***********************************************END*****************************************************
*******************************************************************************************************
