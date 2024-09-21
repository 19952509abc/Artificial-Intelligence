import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
column_names = ['Alternate', 'Bar', 'Fri/Sat', 'Hungry', 'Patrons', 'Price', 'Raining', 'Reservation', 'Type', 'WaitEstimate', 'Wait']
data = pd.read_csv(r'C:\Users\Muhammad Saad\Desktop\AI (Muhammad Saad)(BCS21258)\decisiontree\restaurant.csv', header=None, names=column_names)

# Preprocess data
X = pd.get_dummies(data.drop('Wait', axis=1))  # Convert categorical to numeric
y = data['Wait']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Wait', 'Wait'], yticklabels=['No Wait', 'Wait'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, filled=True, class_names=True)
plt.show()
