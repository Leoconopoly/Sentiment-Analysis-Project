import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data to numerical data using TF-IDF
from sklearn.linear_model import LogisticRegression  # For the logistic regression model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score  # For evaluating the model
import re  # For regular expressions, used in text cleaning
import nltk  # For natural language processing tasks
from nltk.corpus import stopwords  # For accessing the list of stopwords
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For creating beautiful plots

# Download stopwords
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove HTML line breaks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

# Load the dataset
data = pd.read_csv('imdb_dataset.csv')

# Clean the reviews
data['review'] = data['review'].apply(clean_text)

# Define stopwords and convert set to list
stop_words = list(stopwords.words('english'))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.3, random_state=42)

# Initialize the TF-IDF vectorizer with the stopwords
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Fit the vectorizer on the training data and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the TF-IDF transformed training data
model.fit(X_train_tfidf, y_train)

# Make predictions on the TF-IDF transformed test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model and print the classification report
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Get the predicted probabilities for the positive class
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

# Compute the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test.map({'negative': 0, 'positive': 1}), y_prob)

# Compute the average precision score
average_precision = average_precision_score(y_test.map({'negative': 0, 'positive': 1}), y_prob)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'AP={average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test.map({'negative': 0, 'positive': 1}), y_prob)

# Compute the Area Under the Curve (AUC) score
roc_auc = roc_auc_score(y_test.map({'negative': 0, 'positive': 1}), y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'AUC={roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
