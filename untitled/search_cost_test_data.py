import shutil
import os
import random
import numpy as np
import matplotlib.pyplot as plt
"""test_location = "/home/earl/Thesis/GreatCourt/test"
images_location = "/home/earl/Thesis/GreatCourt"
images_test_file_location = "/home/earl/Thesis/GreatCourt/dataset_test.txt"
point_3d_bin_location = "/home/earl/Thesis/local_tmp/points3D.bin"
test_images=[]
with open(images_test_file_location,'r') as data:
    for line in data:
        if line.startswith("se"):
            split_data = line.split(' ')
            #actual_location = images_location+'/'+split_data[0]
            test_images.append(split_data[0])


           # test_file_local_location = split_data[0].split('/')
           # new_name =images_location+'/'+test_file_local_location[0]+'/'+test_file_local_location[0]+test_file_local_location[1]
            #print(new_name)
            #os.rename(actual_location,new_name)
            #location_to_move =test_location+'/'
            ##shutil.move(actual_location,location_to_move)
            #print(location_to_move)
            #actual_location = new_name
            #shutil.move(actual_location,location_to_move)
print(test_images)"""

prob=[]
label =[]
for  i in range (100):
    prob.append(random.uniform(0, 1))
    label.append(random.randint(0,1))
print(prob)
print(label)
positive = np.zeros(10)
negative = np.zeros(10)
positive_truth = np.zeros(10)
negative_truth = np.zeros(10)
for i,j in enumerate(prob):
    classified = label[i]
    truth = random.randint(0,1)
    if truth == 1:
        index = int(j*10)

        positive_truth[index]+=1
        if truth==classified:
            positive[index]+=1
    if truth ==0:
        index = int(j*10)
        negative_truth[index]+=1
        if truth == classified:
            negative[index]+=1
print(positive,positive_truth)
accuracy_positve = np.divide(positive,positive_truth)
accuracy_negative = np.divide(negative,negative_truth)
x_axis= np.arange(10)
x_axis =x_axis/10
fig ,ax = plt.subplots()
line = ax.plot(x_axis,accuracy_positve,label = 'positve')
line2 = ax.plot(x_axis,accuracy_negative,label ='negative')
ax.legend()
plt.xlabel('probability from random forest')
plt.ylabel('percentage of matches')
plt.title('Random_forest n_estimator =1000 , max_features = number of features')
plt.show()





