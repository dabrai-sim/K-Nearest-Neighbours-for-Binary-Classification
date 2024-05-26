import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
dz = pd.read_csv('data/heart_data.csv')

# Preprocessing
dz.drop(['index'], axis=1, inplace=True)
label = dz['cardio'].to_numpy()
features = dz[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

# Normalize features
scaler = MinMaxScaler()
Xtransformed = scaler.fit_transform(features)

# Split dataset
Xtrain = Xtransformed[:42000, :]
Xval = Xtransformed[42000:56000, :]
Xtest = Xtransformed[56000:, :]
trainLabel = label[:42000]
valLabel = label[42000:56000]
testLabel = label[56000:]

# Train KNN model and evaluate
k = np.arange(1, 21, 1)
train_score = []
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(Xtrain, trainLabel)
    train_score.append(model.score(Xtrain, trainLabel))

# Plot training accuracy
plt.plot(k, train_score)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy for Different k Values')
plt.legend(['train acc'])
plt.show()

# Evaluate on validation set
model = KNeighborsClassifier(n_neighbors=30)
model.fit(Xtrain, trainLabel)
yval_op = model.predict(Xval)
print('Validation score is {} %'.format(accuracy_score(valLabel, yval_op) * 100))

# Evaluate on test set
labelPredicted = model.predict(Xtest)
print(confusion_matrix(testLabel, labelPredicted))
print(classification_report(testLabel, labelPredicted))
