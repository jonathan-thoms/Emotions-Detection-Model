import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click.core import augment_usage_errors
from matplotlib.pyplot import figure

keyfacial_df=pd.read_csv('data/data.csv')

keyfacial_df['Image']=keyfacial_df["Image"].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96,96))
'''
fig=plt.figure(figsize=(20,20))

for k in range(64):
    ax=plt.subplot(8,8,k+1)
    i=random.randint(0, len(keyfacial_df["Image"]))
    plt.imshow(keyfacial_df["Image"][i], cmap="grey")
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], "rx")
plt.show()
'''


#data augmentation
#flipping images horizontally
import copy
keyfacial_df_copy=copy.copy(keyfacial_df)
columns=keyfacial_df_copy.columns[:-1]
keyfacial_df_copy["Image"]=keyfacial_df_copy["Image"].apply(lambda x: np.flip(x, 1))
for i in range(len(columns)):
    if i%2 == 0:
        keyfacial_df_copy[columns[i]]=keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))
augemented_df=np.concatenate((keyfacial_df_copy, keyfacial_df))

#flipping images vertically
keyfacial_df_copy=copy.copy(keyfacial_df)
columns=keyfacial_df_copy.columns[:-1]
keyfacial_df_copy["Image"]=keyfacial_df_copy["Image"].apply(lambda x: np.flip(x, 0))
for i in range(len(columns)):
    if i%2 == 1:
        keyfacial_df_copy[columns[i]]=keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))
augemented_df=np.concatenate((keyfacial_df_copy, augemented_df))

'''
plt.imshow(keyfacial_df["Image"][0], cmap="grey")
for j in range(1,31,2):
    plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], "rx")
plt.show()

plt.imshow(keyfacial_df_copy["Image"][0], cmap="grey")
for j in range(1, 31, 2):
    plt.plot(keyfacial_df_copy.loc[0][j - 1], keyfacial_df_copy.loc[0][j], "rx")
plt.show()
'''

keyfacial_df_copy=copy.copy(keyfacial_df)
keyfacial_df_copy["Image"]=keyfacial_df_copy["Image"].apply(lambda x: np.clip(random.uniform(1.5,2)*x, 0, 255.0))
augemented_df=np.concatenate((augemented_df, keyfacial_df_copy))
#print(augemented_df.shape)

#Data normalization

img=augemented_df[:,30]
img=img/255.

X=np.empty((len(img), 96, 96, 1))

for i in range(len(img)):
    X[i]=np.expand_dims(img[i], axis=2)
X=np.asarray(X).astype(np.float32)
#print(X.shape)

y=augemented_df[:,:30]
y=np.asarray(y).astype(np.float32)
#print(y.shape)

#splitting
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)
