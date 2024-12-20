# Use transformer to annotate data with sentiment

# Since transformer is much more precise than Naive Bayes, SVM, SVC, Decision Tree and KNN,
# there's no problem to use it to generate the sentiment annotation of the entire dataset.


import pandas as pd
import csv
from transformers import pipeline


def annotate_sentiment(input_csv, output_csv, model_name):
    sentences = pd.read_csv(input_csv)["text"].to_list()

    annotated_sentences = []
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    results = sentiment_pipeline(sentences)
    for result in enumerate(results):
        annotated_sentences.append((sentences[result[0]], result[1]["score"], result[1]["label"]))

    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "score", "label"])
        writer.writerows(annotated_sentences)


if __name__ == "__main__":
    annotate_sentiment(
        "data/chinese/all.csv",
        "data/chinese/all_with_sentiment.csv",
        "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
    )
    annotate_sentiment(
        "data/english/all.csv",
        "data/english/all_with_sentiment.csv",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
