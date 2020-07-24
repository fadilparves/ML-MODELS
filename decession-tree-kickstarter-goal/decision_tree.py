#!/usr/bin/env python3


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

def change_date_to_0(row):
    if "-" in row['goal']:
        return 0
    else:
        return row['goal']
    
def remove_str_from_goal(row):
    if row['goal_c'][0].isdigit() == True:
        return row['goal_c']
    else:
        return 0


dataset = pd.read_csv("data.csv")
dataset.columns = dataset.columns.str.replace(' ', '')
dataset['goal'] = dataset['goal'].str.strip()
d1 = dataset[dataset['state'] == 'failed']
d2 = dataset[dataset['state'] == 'successful']
frame = [d1,d2]
dataset = pd.concat(frame)
dataset['goal_c'] = dataset.apply(change_date_to_0, axis=1)
dataset['goal_c'] = dataset.apply(remove_str_from_goal, axis=1)
dataset['main_category'].unique()
dataset = dataset[dataset['goal_c'] != '0']
"""
array(['Publishing', 'Film & Video', 'Music', 'Food', 'Design', 'Crafts',
       'Games', 'Comics', 'Fashion', 'Theater', 'Art', 'Photography',
       'Technology', 'Dance', 'Journalism', 'Metal', 'Cookbooks', 'Web',
       'Shorts', 'Plays', 'Hardware', 'Playing Cards', 'World Music',
       'Mobile Games', 'Camera Equipment', 'Classical Music',
       'Conceptual Art', 'Nonfiction', 'Product Design', 'Documentary',
       'Video Games', ' 50 Years in the Making', 'Country & Folk',
       'Mixed Media', 'Comic Books', ' Retro Gaming art.', 'Places',
       'Events', 'Fiction', 'Tabletop Games', 'Video', 'Performance Art',
       'Small Batch', "Children's Books", 'Poetry', 'Public Art',
       'Art Books', 'Drama', 'Apparel', 'Sculpture', 'DIY', 'Hip-Hop',
       'Accessories', 'People', 'Webseries', 'Interactive Design',
       'Periodicals', 'Vegan', 'Indie Rock', 'Academic', 'Pop',
       ' M.ercury E.dition)', 'Faith', 'Jazz', 'Space Exploration',
       'Performances', 'Digital Art', 'Narrative Film', 'Apps',
       'Installations', 'Pet Fashion', 'Restaurants', 'Rock', 'Software',
       'Drinks', 'Architecture', 'Photobooks', 'Textiles', 'Fine Art',
       'Food Trucks', ' Learn', 'Wearables', 'Gaming Hardware',
       'Civic Design', ' Kingdom of Heaven.', 'Zines', 'Musical',
       'Graphic Design', 'Print', 'Horror', 'Animation', 'Flight',
       ' pants', 'Illustration', 'Festivals', 'Radio & Podcasts',
       'Action', 'Young Adult', 'DIY Electronics', 'Painting',
       'Webcomics', 'Television', '3D Printing', 'Audio', 'Jewelry',
       'Woodworking', 'Electronic Music', ' Good for your skin',
       ' Spirits', 'R&B', 'Robots', ' soccer', 'Nature', ' Demons ',
       'Literary Journals', ' Divine Wisdom', 'Farms', 'Ready-to-wear',
       ' Restore Pride', 'Kids'], dtype=object)
"""

X = dataset.iloc[:, [0,1]].values
print(X)
y = dataset['state'].values
print(y)

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
print(X)
y = labelencoder.fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)











