#need gensim 3.8.3
#
#

from collections import defaultdict
from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
         "A survey of user opinion of computer system response time",
         "The EPS user interface management system",
         "System and human system engineering testing of EPS",
         "Relation of user perceived response time to error measurement",
         "The generation of random binary unordered trees",
         "The intersection graph of paths in trees",
         "Graph minors IV Widths of trees and well quasi ordering",
         "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

print("texts",texts)
print()

# --------------------------------------------------
from gensim import corpora
dictionary = corpora.Dictionary(texts)
print("dictionary.token2id", dictionary.token2id)
print()

corpus = [dictionary.doc2bow(text) for text in texts]
print("corpus", corpus)
print()


# --------------------------------------------------
from gensim import models

tfidf = models.TfidfModel(corpus)
print("tfidf", tfidf)
print()

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

# --------------------------------------------------
print()
print("lsi_model")
lsi_model = models.LsiModel( corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf]
lsi_model.print_topics(2)

for doc, as_text in zip(corpus_lsi, documents):
    print( doc, as_text)

# --------------------------------------------------

from gensim.models.nmf import Nmf 

print()
print("nmf")
corpus_nmf = Nmf( corpus_tfidf, num_topics=2)
print("corpus_nmf", corpus_nmf)
print()
corpus_nmf.print_topics(2)


print(corpus_nmf.get_term_topics(word_id=0))
print()
print(corpus_nmf.get_document_topics([(0,1),(1,1),(2,1)]))

# --------------------------------------------------
import gensim
print()
print("texts", texts)
w2v = gensim.models.Word2Vec( texts, min_count=1, size=2)
print(w2v)

for i,word in enumerate(w2v.wv.vocab):
    if i==20:
        break
    print(word,":", w2v[word])

# --------------------------------------------------
print()
pairs =[
        ('human','system'),
        ('system','eps')
        ]

for w1,w2 in pairs:
    print('%r\t%r\t%.2f'%
          (w1,w2, w2v.wv.similarity(w1,w2)))
