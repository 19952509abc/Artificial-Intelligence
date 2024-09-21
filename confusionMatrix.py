from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# True labels (actual values)
y_true = [1]*50 + [1]*10 + [0]*35 + [0]*5  # 50 actual positives, 10 false negatives, 35 actual negatives, 5 false positives

# Predicted labels (by the model)
y_pred = [1]*50 + [0]*10 + [0]*35 + [1]*5  # 50 true positives, 10 false negatives, 35 true negatives, 5 false positives

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision, Recall, F1 Score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
