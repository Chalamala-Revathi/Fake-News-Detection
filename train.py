# ===============================
# STEP 1: Import Libraries
# ===============================
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===============================
# STEP 2: Load Dataset
# ===============================
df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\Fake-News-Detection\\FakeNewsNet.csv")

# ===============================
# STEP 3: Select Required Columns
# ===============================
df = df[['title', 'real']]

# Remove null values
df = df.dropna()

# ===============================
# STEP 4: Define X and y
# ===============================
X = df['title']
y = df['real']

# ===============================
# STEP 5: Convert Text to Numbers
# ===============================
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_vectorized = vectorizer.fit_transform(X)

# ===============================
# STEP 6: Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 7: Train Model
# ===============================
model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# STEP 8: Evaluate Model
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ===============================
# STEP 9: Save Model and Vectorizer
# ===============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and Vectorizer saved successfully!")