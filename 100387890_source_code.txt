from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


stop_words = set(stopwords.words('english'))

def extract_html(html_files):
    soup = BeautifulSoup(html_files, 'html.parser')
    text = soup.get_text()
    title = soup.title.text
    text = re.sub(r'\s+', ' ', text)
    return title, text
documents = {}

files = []

# Reading documents from file
with open("videogame-labels.txt", 'r') as file:
    for line in file:
        file_path = 'videogames/' + line.strip()
        with open(file_path, "r") as doc:
            data = extract_html(doc.read())
            title = data[0]
            text = data[1]
            files.append(text)
            documents[title] = text

def data(collection, stop_words=stop_words):
    Lemmatization  = WordNetLemmatizer()
    tokens = []

    for document in collection:
        words = word_tokenize(document.lower())
        words = [Lemmatization .lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        tokens.append(words)

    return tokens

# inverted index
def inverted_index(documents):
    inverted = {}
    for documented, doc in enumerate(documents):
        for term in set(doc):
            if term not in inverted:
                inverted[term] = set()
            inverted[term].add(documented)
    return inverted
inverted = inverted_index(files)


changed_data = data(files)
#  calculate TF-IDF vectors using inverted index
def calculateTfidf(documents, search, inverted):
    vector = TfidfVectorizer()
    vector.fit(documents)

    # Calculate TF-IDF vectors for documents and query
    doc_tfidf = vector.transform(documents)
    query_tfidf = vector.transform([search])

    return query_tfidf, doc_tfidf


#command line
search = input("Please enter a query: ")

# Calculate TF-IDF vectors
query_tfidf, doc_tfidf = calculateTfidf(files, search, inverted)

# Calculate cosine similarities
similarities = cosine_similarity(query_tfidf, doc_tfidf).flatten()
topSearches = np.argsort(similarities)[::-1][:10]

# Print the results in rank order
print("Top 10 Similar Documents:")
for rank, doc in enumerate(topSearches, start=1):
    title = list(documents.keys())[doc]
    similarity = similarities[doc]
    print(f"Rank {rank}: {title} - Similarity: {similarity:.4f}")
