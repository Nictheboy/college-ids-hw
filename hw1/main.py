import pandas as pd
import shutil

print_score_percentage_for_each_topic = False


def print_header(header: str):
    print()
    print(f" < {header} > ".center(shutil.get_terminal_size().columns, "-"))


# Read the data
data = pd.read_csv("data.csv", sep=";", encoding="windows-1252")

# Print the first 5 rows of the data
print_header("Original Data (first 5 rows)")
print(data.head())

# Preprocess the data
data_dropped_duplicates = data.drop_duplicates()
lines_dropped = len(data) - len(data_dropped_duplicates)
percentage_dropped = int(lines_dropped / len(data) * 100)
dropped = data[data.duplicated(keep=False)]
data = data_dropped_duplicates
print_header("Preprocess Data")
print(f"Dropped {lines_dropped} ({percentage_dropped}%) duplicates, such as:")
print(dropped.head())
null_counts = data.isnull().sum()
print(
    f"Found {null_counts.sum()} null values in the data, so we do not need to drop any."
)

# Get question topics
data["Topic Name"] = data.apply(
    lambda row: (
        row["Topic"]
        if row["Topic"] == row["Subtopic"]
        else f"{row['Topic']} / {row['Subtopic']}"
    ),
    axis=1,
)
topics = data.groupby("Topic Name")[["Topic", "Subtopic"]].first().reset_index()
topics["Question Count"] = (
    data.drop_duplicates("Question ID")["Topic Name"].value_counts().sort_index().values
)

# Print the topics
print_header("Topics")
print(topics[["Topic", "Subtopic", "Topic Name", "Question Count"]])

# Assign score to questions
data["Total Basic Score"] = data["Question Level"].eq("Basic").astype(int)
data["Total Advanced Score"] = data["Question Level"].eq("Advanced").astype(int)
data["Total Score"] = data["Total Basic Score"] + data["Total Advanced Score"]
data["Personal Basic Score"] = data["Total Basic Score"] * data["Type of Answer"]
data["Personal Advanced Score"] = data["Total Advanced Score"] * data["Type of Answer"]
data["Personal Score"] = data["Personal Basic Score"] + data["Personal Advanced Score"]
for i, row in topics.iterrows():
    topic_name = row["Topic Name"]
    total_column_name = "Total Score of " + topic_name
    personal_column_name = "Personal Score of " + topic_name
    data[total_column_name] = data["Total Score"] * data["Topic Name"].eq(topic_name)
    data[personal_column_name] = data[total_column_name] * data["Type of Answer"]


# Calculate the total score of each student
scores = (
    data.groupby("Student ID")[
        [
            "Total Basic Score",
            "Personal Basic Score",
            "Total Advanced Score",
            "Personal Advanced Score",
            "Total Score",
            "Personal Score",
        ]
        + ["Total Score of " + topic_name for topic_name in topics["Topic Name"]]
        + ["Personal Score of " + topic_name for topic_name in topics["Topic Name"]]
    ]
    .sum()
    .reset_index()
)
scores["Basic Percentage"] = (
    scores["Personal Basic Score"] / scores["Total Basic Score"] * 100
)
scores["Advanced Percentage"] = (
    scores["Personal Advanced Score"] / scores["Total Advanced Score"] * 100
)
scores["Percentage"] = scores["Personal Score"] / scores["Total Score"] * 100
for i, row in topics.iterrows():
    topic_name = row["Topic Name"]
    total_column_name = "Total Score of " + topic_name
    personal_column_name = "Personal Score of " + topic_name
    percentage_column_name = "Percentage of " + topic_name
    scores[percentage_column_name] = (
        scores[personal_column_name] / scores[total_column_name] * 100
    )

# Print the total score of each student
print_header("Student Scores (first 5 rows)")
print(
    scores[
        ["Student ID", "Percentage", "Basic Percentage", "Advanced Percentage"]
    ].head()
)

# Print the score of each student in each topic
if print_score_percentage_for_each_topic:
    for i, row in topics.iterrows():
        topic_name = row["Topic Name"]
        percentage_column_name = "Percentage of " + topic_name
        print_header(f"Student Score Percentages in {topic_name} (first 5 rows)")
        topic_percentages = scores[["Student ID", percentage_column_name]].dropna()
        print(topic_percentages.head())


# Calculate max, min, quartiles, mean and std of the scores
def calculate_statistics(column: pd.Series):
    return pd.DataFrame(
        {
            "Max": column.max(),
            "Min": column.min(),
            "Q1": column.quantile(0.25),
            "Q2": column.quantile(0.5),
            "Q3": column.quantile(0.75),
            "Mean": column.mean(),
            "Std": column.std(),
        }
    )


print_header("Statistics")
percentage_column_names = ["Percentage", "Basic Percentage", "Advanced Percentage"] + [
    "Percentage of " + topic_name for topic_name in topics["Topic Name"]
]
print(calculate_statistics(scores[percentage_column_names]))


# Find outliers
def find_outliers(df: pd.DataFrame, column: str):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]


print_header("Outliers")
for column in percentage_column_names:
    outliers = find_outliers(scores, column)
    if len(outliers) > 0:
        print(
            "--------------------------------------------------------------------------------"
        )
        print(f"Outliers in {column}:")
        Q1 = scores[column].quantile(0.25)
        Q3 = scores[column].quantile(0.75)
        IQR = Q3 - Q1
        all = scores[["Student ID", column]].dropna().sort_values(column)
        print(f"Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}, {len(all)} students")
        if column.startswith("Percentage of "):
            topic_name = column[len("Percentage of ") :]
            question_count = topics[topics["Topic Name"] == topic_name][
                "Question Count"
            ].iloc[0]
            print(f"{question_count} questions in {topic_name}")
        print(outliers[["Student ID", column]].sort_values(column))
        print("All students:")
        print(all)
        print(
            "--------------------------------------------------------------------------------"
        )
    else:
        print(f"(No outliers in {column}.)")
