import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Data 
url = "https://data.wien.gv.at/daten/geo?service=WFS&request=GetFeature&version=1.1.0&typeName=ogdwien:BAUMKATOGD&srsName=EPSG:4326&outputFormat=csv"
df = pd.read_csv(url)

# preprocessing: filter top 3 most common species 
top_species = df['GATTUNG_ART'].value_counts().nlargest(3).index
df_filtered = df[df['GATTUNG_ART'].isin(top_species)].dropna(subset=['BAUMHOEHE', 'STAMMUMFANG', 'PFLANZJAHR'])

X = df_filtered[['BAUMHOEHE', 'STAMMUMFANG', 'PFLANZJAHR']] # features
y = df_filtered['GATTUNG_ART'] # label

# train/test data split 80/20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training 
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# generate outputs 

# a) histogram of tree heights
plt.figure(figsize=(10,6))
df_filtered['BAUMHOEHE'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of tree heights in Vienna')
plt.savefig('tree_height_histogram.png')

# b) confusion matrix

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=top_species)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_species)
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Species prediction confusion matrix')
# plt.savefig('confusion_matrix.png')


fig, ax = plt.subplots(figsize=(10, 10)) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_species)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title('Species prediction confusion matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("experiment finished. Files 'tree_height_histogram.png' and 'confusion_matrix.png' generated.")