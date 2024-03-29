import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('/Users/alfinpatel/Documents/projects/sign-language/data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Debugging: Check the length of each element in data
lengths = [len(elem) for elem in data]
max_length = max(lengths)
min_length = min(lengths)
print(f"Max length of elements: {max_length}")
print(f"Min length of elements: {min_length}")

# Pad or truncate elements to ensure they all have the same length
data_padded = [elem[:max_length] + [0] * (max_length - len(elem)) if len(elem) < max_length else elem for elem in data]

# Convert data to NumPy array
data_np = np.array(data_padded)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_np, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump(model, f)
