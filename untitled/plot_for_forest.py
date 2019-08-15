import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as cm
csv_file_location = "/home/earl/Thesis/shopfacade_plots/final_result_small.csv"
dataframe= pd.read_csv(csv_file_location)
data = dataframe.sort_values(by=['accuracy','time'],ascending=[False,True])
data['rank'] = np.arange(len(data))
print(data)
plt.figure(figsize=(10,10))
#sns.scatterplot(x='time',y='accuracy',data=data, hue='accuracy')
plt.title('Time Vs Accuracy - Shop facade')
#plt.scatter(x=data['time'],y=data['accuracy'],cmap='viridis')
#plt.show()

for j,i in enumerate(data.values):

    if j<=5:
        plt.scatter(i[-2],i[-3],c='r',s=40,facecolors= 'none')
        plt.scatter(i[-2], i[-3], c='b', s=20)
    else:
        plt.scatter(i[-2], i[-3], c='b', s=20)

plt.xlabel('Time For Prediction in seconds')
plt.ylabel('Accuracy of the classifier')
plt.show()