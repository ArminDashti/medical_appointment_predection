import numpy as np
import pandas as pd
import seaborn as sborn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing

#%%
df = pd.read_csv('c:/users/armin/desktop/KaggleV2-May-2016.csv')

df = df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)

df['No-show'] = df['No-show'].replace('Yes', 1)
df['No-show'] = df['No-show'].replace('No', 0)

df['Gender'] = df['Gender'].replace('F', 1)
df['Gender'] = df['Gender'].replace('M', 0)

df['scheduled_date'] = df['ScheduledDay'].str[0:10]
df['scheduled_hour'] = df['ScheduledDay'].str[11:13]
df['appointment_date']  = df['AppointmentDay'].str[0:10]
df['appointment_hour']  = df['AppointmentDay'].str[11:13]

df['Age'] = round(df['Age'] / 10) * 10

def convert_day_of_the_week(date):
    return datetime.strptime(str(date), '%Y-%m-%d').strftime('%A')

df['scheduled_dw'] = df['scheduled_date'].apply(convert_day_of_the_week)
df['appointment_dw'] = df['appointment_date'].apply(convert_day_of_the_week)

def convert_to_datetime(date):
    return datetime.strptime(date, '%Y-%m-%d').date()

df['scheduled_date'] = df['scheduled_date'].apply(convert_to_datetime)
df['appointment_date'] = df['appointment_date'].apply(convert_to_datetime)

df['wating'] = (df['appointment_date'] - df['scheduled_date']).dt.days

label_encoder = preprocessing.LabelEncoder()
df['scheduled_dw']= label_encoder.fit_transform(df['scheduled_dw'])
df['appointment_dw']= label_encoder.fit_transform(df['appointment_dw'])

df = df.drop(['ScheduledDay', 'AppointmentDay', 'scheduled_date', 'appointment_date'], axis=1)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

y = df[['No-show']]
x = df.drop(['No-show'], axis=1)

x = Normalizer().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)


clf = LogisticRegression(random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)

accuracy_score(y_test, pred)
#%%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, x, y, cv=5)
scores
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

df_i = df.copy()
df_i_zero = df_i.loc[df_i['No-show'] == 0]
df_i_one = df_i.loc[df_i['No-show'] == 1]
df_i_zero = df_i_zero.sample(frac=0.3)
print(len(df_i_zero))
print(len(df_i_one))
result = pd.concat([df_i_zero, df_i_one])
result = result.loc[result['wating'] >= 0]
y = result[['No-show']]
x = result.drop(['No-show'], axis=1)
# x = result[['wating', 'Age', 'appointment_hour', 'appointment_dw', 'SMS_received']]
# pca = PCA(n_components=6)
# x = pca.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

x = StandardScaler().fit_transform(x)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
pred = clf.predict(X_test)

# clf = SVC(gamma='auto').fit(X_train,y_train)
# pred = clf.predict(X_test)
accuracy_score(y_test, pred)

# scores = cross_val_score(clf, x, y, cv=5)
# scores
#%%
from sklearn.neighbors import KNeighborsClassifier

nbrs = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
pred = nbrs.predict(X_test)
accuracy_score(y_test, pred)
#%%
from sklearn.linear_model import SGDClassifier

cdlf = SGDClassifier(max_iter=5000).fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(y_test, pred)
#%%
from sklearn.naive_bayes import GaussianNB

clf = SVC(gamma='scale').fit(X_train,y_train)
pred = clf.predict(X_test)
accuracy_score(y_test, pred)
#%%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
kmeans.set_output()
#%%
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=10, max_iter=3000, solver='sgd').fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(y_test, pred)