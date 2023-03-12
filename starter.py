"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


STUDENTNUMMER = "1019433" # TODO: aanpassen aan je eigen studentnummer

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)

# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

#print(X)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# teken de punten
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k.') # k = zwart


# TODO: print deze punten uit en omcirkel de mogelijke clusters
# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
# k-means clustering
k = 2
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

# plt.figure()
plt.scatter(x, y, c=kmeans.labels_.astype(float))

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r', zorder=10)

# toon de plots
plt.show()




# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# Plot data
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# TODO: leer de classificaties
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

Y_predict_lr_train = lr.predict(X_train)
Y_predict_dt_train = dt.predict(X_train)
print("Accuracy logistic regression (train):", accuracy_score(y_train, Y_predict_lr_train))
print("Accuracy decision tree (train):", accuracy_score(y_train, Y_predict_dt_train))

# Plot results
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_predict_lr_train)
plt.title("Logistic Regression (train)")
plt.show()

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_predict_dt_train)
plt.title("Decision Tree (train)")
plt.show()

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
Y_predict_lr_test = lr.predict(X_test)
Y_predict_dt_test = dt.predict(X_test)
print("Accuracy logistic regression (comparison real y):", accuracy_score(y_test, Y_predict_lr_test))
print("Accuracy decision tree (comparison real y):", accuracy_score(y_test, Y_predict_dt_test))

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)



# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

# Voorspel
Y_predict_lr_test = lr.predict(X_test)
Y_predict_dt_test = dt.predict(X_test)

# classification_test = data.classification_test(Y_predict_lr_test.tolist())
# print("Logistic Regression classificatie accuratie (test): " + str(classification_test))
#
# classification_test = data.classification_test(Y_predict_dt_test.tolist())
# print("Decision Tree classificatie accuratie (test): " + str(classification_test))

# Stuur logistic regression of decision tree naar server
Z = Y_predict_lr_test
#Z = Y_predict_dt_test


Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))

