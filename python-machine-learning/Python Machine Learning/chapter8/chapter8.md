# Chapter 8
## This chapter will cover the following topics

* Cleaning and preparing text data
* Building feature vectors from text documents
* Training a machine learning model to classify positive and negative movie reviews
* Working with large text datasets using out-of-core learning

##### Sentiment analysis (opinion mining)
* analyzes the [polarity] of documents
  - positive or negative characters in a document
* sub-discipline of natural language processing (NLP)

##### bag-of-words
* represent text as numerical feature vectors
  1. create a vocabulary of unique tokens - for example, words - from the entire set of documents.
  2. We construct a feature vector from each document that contains the counts of how often each word occurs in the particular document.
* raw term frequencies: *tf(t,d)*
  - the number of times a term *t* occurs in a document *d*
  - use CountVectorizer(ngram_range=(2,2)) to do 2-gram
  - (t,d) the number of times a term *t* occurs in a document *d*
* term frequency-inverse document frequency (tf-idf)
  - used to downweight frequently occuring words in the feature vectors
  - product of the term frequency and inverse document frequency
* idf(t,d) = inverse document frequency
  - idf(t,d)=log ( n<sub>d</sub> / 1+df(d,t) )
  - n<sub>d</sub> total number of documents
  - df(d,t) number of documents *d* that contain the term *t*
  - log is used that low document frequencies are not given too much weight
  - adding a constant is optional
    - assigns a non-zero value to terms that occur in all training samples
* cleaning data can come in the form of removing html or removing noise data from a set.
* NLTK library is used for finding the root words of a given word.
  - running -> run
* stop-words are words like *is*, *and*, and *has*.
  - can be removed with the NLTK library

##### Working with bigger data - online algorithms and out-of-core learning
* split the documentation into mini-batches and running the code with Hashing Vectorizer runs the code in 50secs instead of the original 40mins
 - This technique is better to use on ordinary machines
 - has a 87% accuracy rate instead of 90% using the more computer computational approach.
* use Latent Dirichlet allocation for sentence structure and grammar.

##### Summary
* how to classify text documents based on their polarity
  - a basic task in sentiment analysis
* how to encode documents
  - using bag-of-words models
  - how to weight the term frequency by relevance using term frequency-inverse document frequency
* use out-of-core or incremental learning to train a machine learning algorithm without loading all into memory


[polarity]: (../GLOSSARY#polarity)
