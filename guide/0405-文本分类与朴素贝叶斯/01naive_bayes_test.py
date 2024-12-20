from nltk.classify import NaiveBayesClassifier

train = [
    (dict(a=1,b=1,c=1), 'y'),
    (dict(a=1,b=1,c=1), 'x'),
    (dict(a=1,b=1,c=0), 'y'),
    (dict(a=0,b=1,c=1), 'x'),
    (dict(a=0,b=1,c=1), 'y'),
    (dict(a=0,b=0,c=1), 'y'),
    (dict(a=0,b=1,c=0), 'x'),
    (dict(a=0,b=0,c=0), 'x'),
    (dict(a=0,b=1,c=1), 'y')    
    ]

test = [
    (dict(a=1,b=0,c=1)), #unseen
    (dict(a=1,b=0,c=0)), #unseen
    (dict(a=0,b=1,c=1)), #seen 3 times, labels=y,y,x
    (dict(a=0,b=1,c=0)), #seen 1times, label=x
]

#train
classifier = NaiveBayesClassifier.train(train)

#test
labels=classifier.classify_many(test)
print(labels)

#show probabilities
probs=classifier.prob_classify_many(test)
for pdist in probs:
    print('%.4f %.4f' % (pdist.prob('x'),pdist.prob('y')))
    
#how these features work
classifier.show_most_informative_features()
