import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load the dataset
try:
    msg = pd.read_csv('document.csv', names=['message', 'label'])
except FileNotFoundError:
    print("The file 'document.csv' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("The file 'document.csv' is empty.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)

# Print the total instances of the dataset
print("Total Instances of Dataset: ", msg.shape[0])

# Map labels to numerical values
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Split data into features and labels
X = msg.message
y = msg.labelnum

# Split data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN or infinite values in ytrain and remove them
mask = ~pd.isnull(ytrain) & ~pd.isnull(Xtrain)
Xtrain = Xtrain[mask]
ytrain = ytrain[mask]

# Vectorize the text data
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Convert to DataFrame for display
df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
print("Sample of Vectorized Training Data:")
print(df.head())

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

# Display sample predictions
print('Sample Predictions:')
for doc, p in zip(Xtest, pred):
    p = 'pos' if p == 1 else 'neg'
    print(f"{doc} -> {p}")

# Display accuracy metrics
print('Accuracy Metrics:')
print('Accuracy:', accuracy_score(ytest, pred))
print('Recall:', recall_score(ytest, pred))
print('Precision:', precision_score(ytest, pred))
print('Confusion Matrix:\n', confusion_matrix(ytest, pred))
