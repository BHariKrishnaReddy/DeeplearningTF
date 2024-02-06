# NLP HandBook

### Preprocess raw text for Sentiment analysis
Data preprocessing is one of the critical steps in any machine learning project. It includes cleaning and formatting the data before feeding into a machine learning algorithm. For NLP, the preprocessing steps are comprised of the following tasks:
* Tokenizing the string 
* Lowercasing 
* Removing stop words and punctuation 
* Stemming

#### Stemming
Stemming is the process of converting a word to its most general form, or stem. This helps in reducing the size of our vocabulary.
Consider the words:
* learn 
* learning 
* learned 
* learnt
<br>All these words are stemmed from its common root **learn**. However, in some cases, the stemming process produces words that are not correct spellings of the root word.NLTK has different modules for stemming and we will be using the PorterStemmer module which uses the Porter Stemming Algorithm. Let's see how we can use it in the cell below.