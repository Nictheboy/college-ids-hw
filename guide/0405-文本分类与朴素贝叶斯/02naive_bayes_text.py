from nltk.classify import NaiveBayesClassifier
import nltk

def classify_eval(truth, pred):
    idx = 0
    (TP,FP,TN,FN) = (0,0,0,0)
    for truth_label in truth:
        pred_label=pred[idx]
        if (truth_label == True and pred_label==True):
            TP = TP+1
        elif (truth_label == False and pred_label==False):
            TN=TN+1
        elif (truth_label == True and pred_label==False):
            FN=FN+1
        elif (truth_label == False and pred_label==True):
            FP=FP+1
        idx = idx+1
    P=0 if TP==0 else TP/(TP+FP)
    R=0 if TP==0 else TP/(TP+FN)
    F=0 if (P==0 or R==0) else 2*P*R/(P+R)
    Acc=0 if(TP+TN==0)else (TP+TN)/(TP+TN+FP+FN)
    return (P,R,F,Acc)
            
def voc(data):
    voc={}
    for (sentence, val) in data:
        words = sentence.lower().split()
        for w in words:
            voc[w] = True
    return voc

def feature (data, v):
    ftr=[]
    for (sentence, label) in data:
        f=dict((w,0) for w in dict.keys(v))
        words = sentence.lower().split()
        for w in words:
            f[w] =1
        ftr.append((f,label))
    return ftr

train_corpus = [
    ('the team dominated the game', True),
    ('the game was intense', True),
    ('the ball went off the court', True),
    ('they had the ball for the whole game', True),
    ('the president did not comment', False),
    ('the show is over', False)
    ]
    
#build features
v = voc(train_corpus)

#train
train_feature_label = feature( train_corpus, v)
NBC = NaiveBayesClassifier.train(train_feature_label)

#test
test_corpus = [
    ('i lost the keys', False),
    ('the goalkeeper catched the ball', True),
    ('the other team controlled the ball', True),
    ('Sora has two kids', False),
    ('this is a book', True)
    ]

test_feature=[]
test_labels=[]
for(ftr,label)in feature(test_corpus, v):
    test_feature.append(ftr)
    test_labels.append(label)

pred_labels = NBC.classify_many(test_feature)
print(test_labels)
print(pred_labels)

print('precision=%.4f, recall=%.4f, f-score=%.4f,  accuracy=%.4f' %  classify_eval(test_labels, pred_labels))



#show probabilities
probs=NBC.prob_classify_many(test_feature)
for pdist in probs:
    print('%.4f %.4f'% ( pdist.prob(True), pdist.prob(False)))

    #informative features
NBC.show_most_informative_features(20)


from sklearn.svm import LinearSVC

SVMC = nltk.classify.SklearnClassifier(LinearSVC())
SVMC.train( train_feature_label)
pred_labels_SVM=[]
for f in test_feature:
    pred_labels_SVM.append(SVMC.classify(f))
print("------------------------------")
print("pred_labels_SVM",pred_labels_SVM )

print('precision=%.4f, recall=%.4f, f-score=%.4f,  accuracy=%.4f' %  classify_eval(test_labels, pred_labels_SVM))
