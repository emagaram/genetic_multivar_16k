import re
import nltk
from langdetect import detect
from collections import Counter

# Download necessary resources
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('names')
nltk.download('movie_reviews')
nltk.download('webtext')
nltk.download('reuters')
nltk.download('inaugural')
nltk.download('state_union')
nltk.download('twitter_samples')

# Import necessary corpora
from nltk.corpus import (
    gutenberg, brown, wordnet, names, movie_reviews, webtext, reuters, inaugural, state_union, twitter_samples
)

# Function to tokenize text into individual characters, allowing spaces
def tokenize_no_spaces(text):
    return list(re.sub(r'\s+', '', text))  # Remove spaces and tokenize into characters

def tokenize_with_spaces(text):
    return list(text)  # Tokenize into characters, including spaces

# Function to generate n-grams with skips
def generate_ngrams(tokenized_text, skip):
    ngrams = []
    length = len(tokenized_text)
    
    for i in range(length):
        if i + skip + 1 < length:
            ngrams.append((tokenized_text[i], tokenized_text[i + skip + 1]))
    
    return ngrams

# Function to process a corpus and generate bigrams, skip-1-grams, and skip-2-grams
def process_corpus(text):
    tokenized_text = tokenize_no_spaces(text)
    
    bigrams = list(nltk.ngrams(tokenized_text, 2))
    skip_1_grams = generate_ngrams(tokenized_text, 1)
    skip_2_grams = generate_ngrams(tokenized_text, 2)
    
    bigram_freq = Counter(bigrams)
    skip_1_gram_freq = Counter(skip_1_grams)
    skip_2_gram_freq = Counter(skip_2_grams)
    
    return bigram_freq, skip_1_gram_freq, skip_2_gram_freq

# Function to detect the language of a given text
def detect_language(text, corpus_name):
    try:
        lang = detect(text)
        if lang != 'en':
            print(f"Warning: {corpus_name} is detected as {lang}, not English.")
        else:
            print(f"{corpus_name} is English.")
    except:
        print(f"Error: Unable to detect language for {corpus_name}.")

# Function to concatenate texts from various corpora
def concatenate_corpora():
    large_text = ""
    
    # List of corpus functions
    corpora = [
        ("Gutenberg", lambda: " ".join([gutenberg.raw(fileid) for fileid in gutenberg.fileids()])),
        ("Brown", lambda: " ".join([word for fileid in brown.fileids() for word in brown.words(fileid)])),
        ("WordNet", lambda: " ".join([synset.definition() for synset in wordnet.all_synsets()])),
        ("Names", lambda: " ".join(names.words())),
        ("Movie Reviews", lambda: " ".join([word for fileid in movie_reviews.fileids() for word in movie_reviews.words(fileid)])),
        ("WebText", lambda: " ".join([webtext.raw(fileid) for fileid in webtext.fileids()])),
        ("Reuters", lambda: " ".join([word for fileid in reuters.fileids() for word in reuters.words(fileid)])),
        ("Inaugural Address", lambda: inaugural.raw()),
        ("State of the Union", lambda: state_union.raw()),
        ("Twitter Samples", lambda: " ".join(twitter_samples.strings()))
    ]
    
    # Loop through each corpus, concatenate its text, and check language
    for corpus_name, corpus_func in corpora:
        corpus_text = corpus_func()
        detect_language(corpus_text, corpus_name)  # Check the language
        large_text += corpus_text
    
    return large_text

# Concatenate all the texts from various corpora
large_text = concatenate_corpora()
print("Large text length:", len(large_text))

# Process the large combined text to get n-gram frequencies
bigram_freq, skip_1_gram_freq, skip_2_gram_freq = process_corpus(large_text)

# Display the frequencies
print("Bigram Frequencies:", bigram_freq.most_common(20))
print("Skip-1-gram Frequencies:", skip_1_gram_freq.most_common(10))
print("Skip-2-gram Frequencies:", skip_2_gram_freq.most_common(10))