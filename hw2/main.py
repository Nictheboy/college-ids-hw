import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Data Set: Wholesale customers data
# From: https://archive.ics.uci.edu/dataset/292/wholesale+customers

# Load the dataset
data = pd.read_csv('/home/nictheboy/Documents/college-ids-hw/hw2/data.csv')

# Define features (X) and target (y)
X = data[['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
y = data['Channel']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Support Vector Machine': SVC(class_weight='balanced')
}

# Train and evaluate classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Classifier: {name}")
    print(classification_report(y_test, y_pred))


# Initialize clustering algorithms
clustering_algorithms = {
    'K-Means': KMeans(n_clusters=2),
    'Gaussian Mixture': GaussianMixture(n_components=2)
}

# Cluster the data
for name, algo in clustering_algorithms.items():
    algo.fit(X)
    y_pred = algo.predict(X)
    print(f"Clustering Algorithm: {name}")
    print(f"Silhouette Score: {silhouette_score(X, y_pred)}")
