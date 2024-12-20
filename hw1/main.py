import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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


# Find relationship between country and score

# Merge country with scores
student_and_country = (
    data.groupby("Student ID")["Student Country"].first().reset_index()
)
score_and_country = pd.merge(scores, student_and_country, on="Student ID")
print_header("Student, Country and Score (first 5 rows)")
print(score_and_country.head())

# Encode country
label_encoder = LabelEncoder()
label_encoder.fit(score_and_country["Student Country"])
score_and_country["Country Code"] = label_encoder.transform(
    score_and_country["Student Country"]
)

# Fill NaN in percentages
for column in percentage_column_names:
    score_and_country[column] = score_and_country[column].fillna(
        scores[column].quantile(0.5)
    )
percentages = score_and_country[percentage_column_names]

# Do not need to scale the data because the percentages are already in the same range

# Use PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
pca.fit(percentages[percentage_column_names])
percentages_pca = pca.transform(percentages[percentage_column_names])

# Use T-SNE to reduce the dimensionality of the data
tsne = TSNE(n_components=2)
percentages_tsne = tsne.fit_transform(percentages[percentage_column_names])


# Plot the data
def plot_points_by_country(ndarr, x: str, y: str, title: str, filename: str):
    plt.figure(figsize=(10, 10))
    for i, country in enumerate(label_encoder.classes_):
        country_indices = score_and_country["Country Code"] == i
        plt.scatter(
            ndarr[country_indices, 0],
            ndarr[country_indices, 1],
            label=country,
        )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.savefig("images/" + filename)


def plot_curve_by_country(ndarr, y: str, title: str, filename: str):
    plt.figure(figsize=(10, 10))
    for i, country in enumerate(label_encoder.classes_):
        country_indices = score_and_country["Country Code"] == i
        sns.kdeplot(ndarr[country_indices], label=country)
    plt.xlabel("Density")
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.savefig("images/" + filename)


plot_points_by_country(
    scores[["Total Score", "Percentage"]].to_numpy(),
    "Total Score",
    "Percentage",
    "Student Scores by Total Score and country",
    "percentage_total.png",
)
plot_curve_by_country(
    percentages["Percentage"],
    "Percentage",
    "Density of Student Scores by Country",
    "percentage_density.png",
)
plot_points_by_country(
    percentages_pca,
    "Principle Component 1",
    "Principle Component 2",
    "PCA of Student Scores by Country",
    "percentage_pca.png",
)
plot_points_by_country(
    percentages_tsne,
    "T-SNE Component 1",
    "T-SNE Component 2",
    "T-SNE of Student Scores by Country",
    "percentage_tsne.png",
)


# Calculate correlations

print_header("Correlation of Scores")
correlation = score_and_country[
    [
        "Personal Score",
        "Personal Basic Score",
        "Personal Advanced Score",
        "Country Code",
    ]
].corr()
print(correlation)

print_header("Correlation with Percentage")
correlation = score_and_country[["Country Code"] + percentage_column_names].corr()
print(correlation["Percentage"].sort_values(ascending=False))


# Grouping students

# Standardize the data
scaler = StandardScaler()
scaler_score_and_country = scaler.fit_transform(
    score_and_country[["Country Code"] + percentage_column_names]
)

# Use KNN to group students
k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(scaler_score_and_country)
score_and_country["KNN Group"] = kmeans.labels_


# Use decision tree to group students
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(scaler_score_and_country, score_and_country["KNN Group"])
score_and_country["Decision Tree Group"] = decision_tree.predict(
    scaler_score_and_country
)

# Use SVC to group students
svc = SVC()
svc.fit(scaler_score_and_country, score_and_country["KNN Group"])
score_and_country["SVC Group"] = svc.predict(scaler_score_and_country)


# Plot groups
def plot_groups(
    ndarr, k: int, group_name: str, x: str, y: str, title: str, filename: str
):
    plt.figure(figsize=(10, 10))
    for i in range(k):
        group_indices = score_and_country[group_name] == i
        plt.scatter(
            ndarr[group_indices, 0],
            ndarr[group_indices, 1],
            label=f"Group {i}",
        )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.savefig("images/" + filename)


plot_groups(
    percentages_pca,
    k,
    "KNN Group",
    "Principle Component 1",
    "Principle Component 2",
    "PCA of Student Scores by KNN Group",
    "group_knn_pca.png",
)
plot_groups(
    percentages_tsne,
    k,
    "KNN Group",
    "T-SNE Component 1",
    "T-SNE Component 2",
    "T-SNE of Student Scores by KNN Group",
    "group_knn_tsne.png",
)
plot_groups(
    percentages_pca,
    decision_tree.classes_.size,
    "Decision Tree Group",
    "Principle Component 1",
    "Principle Component 2",
    "PCA of Student Scores by Decision Tree Group",
    "group_decision_tree_pca.png",
)
plot_groups(
    percentages_tsne,
    decision_tree.classes_.size,
    "Decision Tree Group",
    "T-SNE Component 1",
    "T-SNE Component 2",
    "T-SNE of Student Scores by Decision Tree Group",
    "group_decision_tree_tsne.png",
)
plot_groups(
    percentages_pca,
    svc.classes_.size,
    "SVC Group",
    "Principle Component 1",
    "Principle Component 2",
    "PCA of Student Scores by SVC Group",
    "group_svc_pca.png",
)
plot_groups(
    percentages_tsne,
    svc.classes_.size,
    "SVC Group",
    "T-SNE Component 1",
    "T-SNE Component 2",
    "T-SNE of Student Scores by SVC Group",
    "group_svc_tsne.png",
)
