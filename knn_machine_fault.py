import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/predictive_maintenance.csv')

# Preprocessing
df.drop(['UDI', 'Type', 'Product ID'], axis=1, inplace=True)
label = df['Target'].to_numpy()
features = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]

# Normalize features
scaler = MinMaxScaler()
Xtransformed = scaler.fit_transform(features)

# Split dataset
Xtrain = Xtransformed[:6000, :]
Xval = Xtransformed[6000:8000, :]
Xtest = Xtransformed[8000:, :]
trainLabel = label[:6000]
valLabel = label[6000:8000]
testLabel = label[8000:]

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
model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xtrain, trainLabel)
yval_op = model.predict(Xval)
print('Validation score is {} %'.format(accuracy_score(valLabel, yval_op) * 100))

# Evaluate on test set
labelPredicted = model.predict(Xtest)
print(confusion_matrix(testLabel, labelPredicted))
print(classification_report(testLabel, labelPredicted))
