import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")
# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})

# Fill the nan values in the age column
passengers['Age'].fillna(inplace = True, value = round(passengers['Age'].mean()))

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)


# Scale the feature data so it has man = 0 and standard deviation = 1
scale = StandardScaler()
train_features = scale.fit_transform(train_features)
test_features = scale.fit_transform(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Score the model on the train data
model.score(train_features, train_labels)

# Score the model on the test data
model.score(test_features, test_labels)

# Analyze the coefficients
		# print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

# Sample passenger features
Evan = np.array([0.0, 15.0, 0.0, 1.0])
Teo = np.array([0.0, 14.0, 1.0, 0.0])
Henry = np.array([0.0, 15.0, 1.0, 0.0])
Jack = np.array([0.0, 15.0, 1.0, 0.0])
Aiden = np.array([0.0, 14.0, 0.0, 1.0])
Raphael = np.array([0.0, 15.0, 1.0, 0.0])
Rebecca = np.array([1.0, 15.0, 0.0, 1.0])
Merrick = np.array([0.0, 15.0, 1.0, 0.0])
Defne = np.array([1.0, 13.0, 1.0, 0.0])
Anna = np.array([1.0, 14.0, 0.0, 1.0])


# Combine passenger arrays
sample_passengers = np.array([Evan, Teo, Henry, Jack, Aiden, Raphael, Rebecca, Merrick, Defne, Anna, Chris])

# Scale the sample passenger features
sample_passengers = scale.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
  # print(model.predict_proba(sample_passengers))e
