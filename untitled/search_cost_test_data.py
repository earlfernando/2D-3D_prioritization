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


positive =[     0,      0,      0,     0,      0, 332447,133521,   5081,      0,
      0]
negative = [     0,     0,      0,      0,      0, 773928, 308695, 135575,  44742,
   1766]
positive_truth =[1.00000e+00, 2.01000e+02 ,3.11600e+03, 2.45400e+04 ,1.43622e+05, 3.32447e+05,
                  1.33521e+05, 5.08100e+03 ,0.00000e+00, 0.00000e+00]
negative_truth= [     0,   0,   3310, 171964, 862628, 773928, 308695, 135575,  44742,
   1766]
accuracy_negative=[]
accuracy_positve=[]
for i in range(10):
    if positive[i]:
        accuracy_positve.append(positive[i]/positive_truth[i])
    else :
        accuracy_positve.append(0)
    if negative[i] >0:
        accuracy_negative.append(negative[i]/negative_truth[i])
    else :
        accuracy_negative.append(0)

x_axis= np.arange(10)
x_axis =x_axis/10
fig ,ax = plt.subplots()
line = ax.plot(x_axis,accuracy_positve,label = 'positve')
line2 = ax.plot(x_axis,accuracy_negative,label ='negative')
ax.legend()
plt.xlabel('probability from random forest')
plt.ylabel('percentage of matches')
plt.title('Random_forest n_estimator =1,added_estimator = 1 , max_features = number of features')
plt.show()





