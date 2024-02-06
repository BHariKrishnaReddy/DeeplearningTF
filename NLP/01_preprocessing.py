import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
stopwords_english = stopwords.words('english')

nltk.dowload('twitter_samples')
nltk.download('stopwords')

positive = twitter_samples.strings('positive_tweets.json')
negative = twitter_samples.strings('negative_tweets.json')

#visual
fig = plt.figure(figsize=(5,5))
labels = 'Positives','Negatives'

sizes = [len(postive),len(negative)]
# Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.show()

tweet = positive[2277] # Sample

# Preprocess RAW
# hyperlinks, Twitter marks and styles
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2) # removing hyperLInks
tweet2 = re.sub(r'#', '', tweet2) # remove hashtags

#tokennization
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet2)

# Remove stop words and punctuations
tweets_clean = []

for word in tweet_tokens: # Go through every word in your tokens list
    if (word not in stopwords_english and  # remove stopwords
        word not in string.punctuation):  # remove punctuation
        tweets_clean.append(word)

print('removed stop words and punctuation:')
print(tweets_clean)

#Stemming
stemmer = PorterStemmer()
tweets_stem = []

for word in tweets_clean:
    stem_word = stemmer.stem(word)  # stemming word
    tweets_stem.append(stem_word)  # append to the list

print('stemmed words:')
print(tweets_stem)

from utils import process_tweet # Import the process_tweet function

# choose the same tweet
tweet = all_positive_tweets[2277]

print()
print('\033[92m')
print(tweet)
print('\033[94m')

# call the imported function
tweets_stem = process_tweet(tweet); # Preprocess a given tweet

print('preprocessed tweet:')
print(tweets_stem) # Print the result
