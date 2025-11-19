import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure

keyfacial_df=pd.read_csv('data/data.csv')

keyfacial_df['Image']=keyfacial_df["Image"].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96,96))

fig=plt.figure(figsize=(20,20))

for k in range(64):
    ax=plt.subplot(8,8,k+1)
    i=random.randint(0, len(keyfacial_df["Image"]))
    plt.imshow(keyfacial_df["Image"][i], cmap="grey")
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], "rx")
plt.show()