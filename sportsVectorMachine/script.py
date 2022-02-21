import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

def find_strike_zone(player):

  # print(aaron_judge.columns)
  # print(aaron_judge.description.unique())
  # print(aaron_judge.type.unique())

  player['type'] = player['type'].map({'S': 1, 'B': 0})

  # print(aaron_judge['type'])
  # print(aaron_judge['plate_x'])

  player = player.dropna(subset = ['type', 'plate_x', 'plate_z'])

  plt.scatter(x = player['plate_x'], y = player['plate_z'], c = player['type'], cmap = plt.cm.coolwarm, alpha = 0.5)

  training_set, validation_set = train_test_split(player, random_state = 1)

  classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

  draw_boundary(ax, classifier)

  print(classifier.score(validation_set[['plate_x, 'plate_z']], validation_set.type))

  plt.show()

find_strike_zone(david_ortiz)
