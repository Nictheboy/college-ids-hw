# Perform regular tasks


import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import jieba


# (1) 数据集装载、显示
def load_data(file_path: str, threshold):
    data = pd.read_csv(file_path)
    data = data[
        (
            (data["label"].astype(str).str.lower() == "positive")
            & (data["score"] >= threshold["positive"])
        )
        | (
            (data["label"].astype(str).str.lower() == "negative")
            & (data["score"] >= threshold["negative"])
        )
    ].reset_index(drop=True)
    print(data)
    return data


# (2) 分词
def vectorize_en(text):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(text), vectorizer


def vectorize_ch(text):
    text = [" ".join(jieba.lcut(string)) for string in text]
    vectorizer = CountVectorizer()
    tokens = vectorizer.fit_transform(text)
    return tokens, vectorizer


# (3) 全集、分类别子集的词云可视化
def generate_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", font_path="simhei.ttf"
    ).generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()


# (4) 分类模型（多种分类算法）
def train_models(X_train, y_train):
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model
    return models


# (5) 分类效果评价（accuracy, precision, recall）
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print()
        print(f"    Model: {name}")
        print(classification_report(y_test, y_pred))


# (6) 利用模型，对单一文档进行分类
def classify_document(model, vectorizer, document):
    tokens = vectorizer.transform([document])
    prediction = model.predict(tokens)
    return prediction


def analyze(filename, threshold, vectorizer):
    data = load_data(filename, threshold)
    X, vectorizer = vectorizer(data["text"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    generate_wordcloud(data["text"], "Full Dataset Wordcloud")
    generate_wordcloud(
        data[data["label"].astype(str).str.lower() == "positive"]["text"],
        "Positive Dataset Wordcloud",
    )
    generate_wordcloud(
        data[data["label"].astype(str).str.lower() == "negative"]["text"],
        "Negative Dataset Wordcloud",
    )

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    for i in range(20):
        print()
        print(f"Document {i + 1}: {data['text'][i]}")
        print(
            f"Predicted Label: {classify_document(models['Naive Bayes'], vectorizer, data['text'][i])[0]}"
        )
        print(f"Actual Label: {data['label'][i]}")


if __name__ == "__main__":
    analyze(
        "data/chinese/all_with_sentiment.csv", {"positive": 0.900, "negative": 0.900}, vectorize_ch
    )
    analyze(
        "data/english/all_with_sentiment.csv", {"positive": 0.999, "negative": 0.999}, vectorize_en
    )
