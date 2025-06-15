from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
X = ["buy cheap offer", "meeting schedule", "win cash prize", "project update"]
y = ["spam", "ham", "spam", "ham"]

# Convert text to numeric features
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_counts, y)

# Predict new data
test = vectorizer.transform(["Spam"])
print(model.predict(test))
