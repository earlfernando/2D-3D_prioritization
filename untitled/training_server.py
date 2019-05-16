import numpy as np
import sqlite3
import sys
from read_model import read_images_binary, read_points3d_binary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn import metrics
import csv
import pickle
from sklearn.cluster import KMeans,MiniBatchKMeans
from fptas import  FPTAS
from operator import itemgetter
import random
import matplotlib.pyplot as plt

#database_locatiom = "/home/earl/Thesis/GreatCourt/greatCourt_database.db"
image_bin_location = "/home/earl/Thesis/GreatCourt/images.bin"
csv_file_location = "/home/earlfernando/training/training_Data_RandomForest.csv"

file_name_random_forest = "/home/earlfernando/training/test_model_random_forest_2.sav"
file_name_kmeans = "/home/earlfernando/training//test_model_kmeans.sav"
feature_length =128
csv_file_location_kmeans= "/home/earlfernando/training/train_kmeans.csv"
number_of_clusters = 10000
#database_location_overall = "/home/earl/Thesis/GreatCourt/greatCourt_database.db"
#image_bin_location_overall = "/home/earl/Thesis/GreatCourt/images.bin"
#point3D_location_overall = "/home/earl/Thesis/GreatCourt/points3D.bin"
csv_file_location_kmeans_test = "/home/earlfernando/training/test_kmeans.csv"
max_cost = 200
capacity =200

def blob_to_array(blob, dtype, shape=(-1,)):
    IS_PYTHON3 = sys.version_info[0] >= 3
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def check(array_local, x, y):
    return_value = False

    b= np.array(array_local[:,0] ==x)
    list_of_row_indices= np.where(b)
    for i in list_of_row_indices:
         if array_local[i] == [x, y]:
                return_value = True
                break
    return  return_value

class image :
    def __init__(self,name,train_test):
        self.name = name
        self.feature_location= []
        self.descriptor =[]
        self.poistive_descriptor =[]
        self.poistive_descriptor_location =[]
        self.negative_descriptor = []
        self.negative_descriptor_location = []
        self.positive_index=[]
        self.train_test = train_test


    def add_positve(self,x,y):
        #if not check(self.poistive_descriptor_location,x,y):
            list_of_row_indices =  np.where(np.array(self.feature_location[:,0])==x)

            b= self.feature_location
            c = np.array(self.feature_location[:,0])==x
            f= np.where(c)
            d= b[c]
            t= 0

            for i in list_of_row_indices[0]:
                t+=1


                truth = self.feature_location[i]
                if truth[1]==y:
                    self.positive_index.append(i)
                    self.poistive_descriptor_location.append([x, y])
                    start = i * 128

                    end = start + 128
                    t=np.arange(start,end)
                    descriptor =self.descriptor[t]

                    self.poistive_descriptor.append(descriptor)
                    break



    def add_negative(self):
        mask = np.ones(np.shape(self.feature_location)[0],dtype=bool)
        mask[self.positive_index]=False
        self.negative_descriptor_location = self.feature_location[mask]
        for number, i in enumerate(mask):
            if i :
                start = number *128
                end = start+128
                t= np.arange(start,end)
                descriptor = self.descriptor[t]

                self.negative_descriptor.append(descriptor)



def add_feature_location(database_location):
    database = sqlite3.connect(database_location)
    cursor =database.cursor()
    test_images_id =[]
    training_images_id =[]


    test_images_str = test_images_string()
    cursor.execute('''SELECT image_id,name FROM IMAGES ''')

    for row in cursor:
        """image_name = row[1].split('/')
        if image_name[0]=='test':
            test_images_id.append(row[0])"""
        if row[1] in test_images_str:
            test_images_id.append(row[0])
        else:
            training_images_id.append(row[0])
    database.close()
    training_images_id = np.array(training_images_id)
    test_images_id = np.array(test_images_id)

    class_array =[]

    database = sqlite3.connect(database_location)
    cursor =database.cursor()
    second_curosr = database.cursor()
    #cursor.execute('''SELECT image_id,data,rows FROM  descriptors''')

    cursor.execute('''SELECT image_id,data,rows FROM descriptors''')
    second_curosr.execute('''SELECT image_id, data, rows FROM keypoints''')
    counter =0
    for first,row in zip(second_curosr,cursor):
        """if str(row[0])== str(first[0]):

            print(str(row[0]),np.shape(blob_to_array(row[1],dtype=np.uint8)),str(row[2]),blob_to_array(row[1],dtype=np.uint8))
        else:
            return print("Error image_id doesnt match")
        print(blob_to_array(first[1],dtype= np.float32), str(first[2]),np.shape(blob_to_array(first[1],dtype= np.float32)))
        #features_location_local = [blob_to_array(first[1],dtype= np.float32)[::6],blob_to_array(first[1],dtype= np.float32)[1::6]]
        #new_location = np.column_stack((blob_to_array(first[1],dtype= np.float32)[::6],blob_to_array(first[1],dtype= np.float32)[1::6]))"""

        mask_test = test_images_id == row[0]


        mask_train = training_images_id== row[0]

        if len(test_images_id[mask_test])==1:
            test_train = 0

        else :
             test_train =1



        class_array.append(image(str(row[0]),train_test= test_train))

        class_array[counter].feature_location = np.column_stack((blob_to_array(first[1],dtype= np.float32)[::6],blob_to_array(first[1],dtype= np.float32)[1::6]))
        class_array[counter].descriptor = np.array(blob_to_array(row[1],dtype=np.uint8))
        counter+=1


    database.close()
    return class_array

def make_training_data(cameras,image_array):
    for rand, cam in enumerate(cameras):


        for h, k in enumerate(cameras[cam].point3D_ids):

            if k >= 0:

                id = cameras[cam].xys[h]
                image_array[cam - 1].add_positve(id[0], id[1])

        image_array[cam - 1].add_negative()


    postive_samples = []
    negative_samples = []
    for image in image_array:
        if image.train_test == 1:
            for descriptor in image.poistive_descriptor:
                postive_samples.append(descriptor)

            for descriptor in image.negative_descriptor:
                negative_samples.append(descriptor)
    return postive_samples,negative_samples


def make_testing_data(cameras,image_array):
    for rand, cam in enumerate(cameras):

        o = cameras[cam].point3D_ids
        local_image = image_array[cam-1]
        counter = 0
        for h, k in enumerate(cameras[cam].point3D_ids):
            counter +=1
            if k >=0:
                id = cameras[cam].xys[h]
                image_array[cam - 1].add_positve(id[0], id[1])
        image_array[cam - 1].add_negative()
        if len(image_array[cam-1].poistive_descriptor) == 0:
            print('error')

    return image_array




def create_headers(feature_length):
    columns=[]
    for i in range(feature_length):
        columns.append(str(i))

    return columns
def handle_data(positive,negative,feature_length,csv_file_location):
    print('data_handling')
    headers = create_headers(feature_length)
    headers.append('label')
    print(np.shape(positive)[0],np.shape(negative)[0])

    positive_label = np.ones((np.shape(positive)[0], 1))
    negative_label = np.zeros((np.shape(negative)[0], 1))
    training_samples = np.vstack((positive, negative))
    training_labels = np.vstack((positive_label, negative_label))
    shuffling_array = np.arange(np.shape(positive)[0]+np.shape(negative)[0])
    np.random.shuffle(shuffling_array)
    print(np.shape(positive)[0]+np.shape(negative)[0])


    #shape_training_labels = np.shape(training_labels[0])
    #samples = np.append(training_samples, training_labels, axis=1)

    with open (csv_file_location,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for i in shuffling_array:
            label = training_labels[i]
            descriptor = training_samples[i]
            k= np.append(descriptor,label)
            #values.append(tuple(k))
            writer.writerow(k)



    #for i in samples:
        #values.append(tuple(i))
    csvfile.close()

    return headers


def handle_data_for_test(positive,negative,feature_length,csv_file_location_kmeans):
    print('data_handling')
    headers = create_headers(feature_length)
    headers.append('label')
    training_samples = np.vstack((positive, negative))







    #shape_training_labels = np.shape(training_labels[0])
    #samples = np.append(training_samples, training_labels, axis=1)
    #samples_kmeans = random.sample(samples,100000)
    with open (csv_file_location_kmeans,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for j,i in enumerate(training_samples):


            #values.append(tuple(k))
            writer.writerow(i)



    #for i in samples:
        #values.append(tuple(i))
    csvfile.close()

    return headers




def handle_data_for_kmeans(positive,negative,feature_length,csv_file_location_kmeans):
    print('data_handling')
    headers = create_headers(feature_length)
    headers.append('label')
    positive = random.sample(positive,100000)







    #shape_training_labels = np.shape(training_labels[0])
    #samples = np.append(training_samples, training_labels, axis=1)
    #samples_kmeans = random.sample(samples,100000)
    with open (csv_file_location_kmeans,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for j,i in enumerate(positive):


            #values.append(tuple(k))
            writer.writerow(i)



    #for i in samples:
        #values.append(tuple(i))
    csvfile.close()

    return headers


def random_forest(headers,feature_length,csv_file_location,file_name):
    #df = pd.DataFrame.from_records(values, columns=headers)

    df = pd.read_csv(csv_file_location,names= headers)

    print("csv_read")
    X = df[create_headers(feature_length)]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=1000,max_features= None)
    print('training')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    pickle.dump(clf,open(file_name,'wb'))


def random_forest_chunks(headers, feature_length, csv_file_location, file_name):
    # df = pd.DataFrame.from_records(values, columns=headers)
    chunk_size = 10**4
    counter = 0
    clf = RandomForestClassifier(n_estimators=10000,warm_start=True,max_features= None,n_jobs=-1,min_samples_leaf=100,oob_score= True)
    for i,chunk in enumerate(pd.read_csv(csv_file_location, header= 0, chunksize= chunk_size)):
        X =chunk[create_headers(feature_length)]
        print(i)
        y = chunk['label']
        clf.fit(X, y)
        if i <=30:
            trees = 1000
        else:
            trees = 200

        print(clf.oob_score_)
        clf.n_estimators += trees
    return clf
    #pickle.dump(clf, open(file_name, 'wb'))




def k_means(headers,feature_length,csv_file_location,file_name,number_of_clusters):
    chunk_size = 10 **3
    kmeans = KMeans(n_clusters=10000, max_iter=10)
    c=0
    for chunk in pd.read_csv(csv_file_location, header=0, chunksize=chunk_size):
        X = chunk[create_headers(feature_length)]
        kmeans.fit(X)


def k_means_broken_samples(headers,feature_length,csv_file_location_kmeans,file_name,number_of_clusters):
    chunk_size = 10 **4
    kmeans = MiniBatchKMeans(n_clusters=number_of_clusters,batch_size = 100, max_iter=100,random_state= 42,verbose=True)
    print("entering loop")

    for i, chunk in enumerate(pd.read_csv(csv_file_location_kmeans, header = 0, chunksize=chunk_size)):
        X = chunk[create_headers(feature_length)]
        kmeans.partial_fit(X)
        print(i)
    pickle.dump(kmeans, open(file_name, 'wb'))

def search_cost_calculation(headers,feature_length,csv_file_location_kmeans,file_name,number_of_clusters):
    chunk_size = 10**4
    loaded_model = pickle.load(open(file_name, 'rb'))
    result = []
    search_cost =[]
    for chunk in pd.read_csv(csv_file_location_kmeans, header =0, chunksize=chunk_size):
        X = chunk[create_headers(feature_length)]
        local_result=loaded_model.predict(X)
        np.append(result,local_result)
    for i in range(number_of_clusters):
        mask = result==i
        cost = np.sum(mask)
        search_cost.append(cost)


def test_images_string():
    images_test_file_location = "/home/earl/Thesis/GreatCourt/dataset_test.txt"
    test_images = []
    with open(images_test_file_location, 'r') as data:
        for line in data:
            if line.startswith("se"):
                split_data = line.split(' ')
                # actual_location = images_location+'/'+split_data[0]
                test_images.append(split_data[0])
    return test_images


def make_test_data(points3D_location,database_location):
    points3D =read_points3d_binary(points3D_location)
    database = sqlite3.connect(database_location)
    cursor =database.cursor()
    test_images_id =[]
    training_images_id =[]


    test_images_str = test_images_string()
    cursor.execute('''SELECT image_id,name FROM IMAGES ''')
    for row in cursor:
        """image_name = row[1].split('/')
        if image_name[0]=='test':
            test_images_id.append(row[0])"""
        if row[1] in test_images_str:
            test_images_id.append(row[0])
        else:
            training_images_id.append(row[0])
    database.close()
    print('test images obtrained')
    test_images_id = np.array(test_images_id)
    training_images_id = np.array(training_images_id)
    local_image_array = add_feature_location(database_location)
    cameras = read_images_binary(image_bin_location)
    image_array = make_testing_data(cameras,local_image_array)
    print('image_array made')
    test_data_positive=[]
    test_data_negative=[]

    for cam in points3D:
        images_with_3d = points3D[cam].image_ids
        images_with_3d_2d = points3D[cam].point2D_idxs
        test_common = np.intersect1d(images_with_3d,test_images_id)
        if len(test_common)>=1:
            train_common = np.intersect1d(images_with_3d,training_images_id)
            if len(train_common)>=2:

                    for common in test_common:
                        location= np.where(images_with_3d==common)
                        counter =0
                        id_2d = images_with_3d_2d[location][0]
                        image_camera = cameras[common].point3D_ids
                        for g,i in enumerate(image_camera):
                            if i>0:
                                counter +=1
                                if cam == i:
                                    index = counter

                        image_class = image_array[common- 1]
                        descriptor =image_class.poistive_descriptor[index-1]
                        test_data_positive.append(descriptor)
            else:


                location = test_common[0]
                counter =0
                image_camera = cameras[location].point3D_ids
                for g, i in enumerate(image_camera):
                    if i > 0:
                        counter += 1
                        if cam == i:
                            index = counter


                image_class = image_array[location - 1]
                descriptor = image_class.poistive_descriptor[index-1]
                test_data_negative.append(descriptor)
    """
        else :
            for location in range(1,len(images_with_3d)+1):
                id_2d = images_with_3d_2d[location]
                image_class = image_array[location-1]
                descriptor =image_class.poistive_descriptor[id_2d]
                test_data_negative.append(descriptor)"""
    print('adding negative')
    for image in image_array:
        if image.train_test ==1:
            for descriptor in image.negative_descriptor:
                test_data_negative.append(descriptor)
    return test_data_positive,test_data_negative







    return test_data

def prediction (headers,feature_length,csv_file_location_test,file_name_random_forest,file_name_kmeans,number_of_clusters,search_cost,capacity):
    chunk_size = 10**3
    forest_model = pickle.load(open(file_name_random_forest, 'rb'))
    kmeans_model = pickle.load(open(file_name_kmeans,'rb'))
    result_forest= []
    result_kmeans =[]
    prediction_accracy = 0
    for chunk in pd.read_csv(csv_file_location_test, names=0, chunksize=chunk_size):
        X = np.array(chunk[create_headers(feature_length)])
        print(np.shape(X))
        y = np.array(chunk['label'])

        kmeans_result_local = kmeans_model.predict(X)
        forest_result_local = forest_model.predict_proba(X)
        np.append(result_kmeans,kmeans_result_local)
        np.append(result_forest,forest_result_local)


    actual_cost =[]
    for i in result_kmeans:
        np.append(actual_cost,search_cost[i])
    list_for_prioritization = [(prob,cost)for prob,cost in zip(result_forest,actual_cost)]
    number_of_items = len(result_kmeans)
    best_cost,best_combination=FPTAS(number_of_items,capacity=capacity,weight_cost=list_for_prioritization,scaling_factor=number_of_items)
    return  best_cost

def prediction_forest (headers,feature_length,csv_file_location_test,file_name_random_forest,clf):
    chunk_size = 10**4
    forest_model = clf
    #forest_model = pickle.load(open(file_name_random_forest, 'rb'))
    result_forest= []
    positive = np.zeros(10)
    negative = np.zeros(10)
    positive_truth = np.zeros(10)
    negative_truth = np.zeros(10)
    total = 0
    model_accuracy = 0
    chunk_accuracy = 0
    chunk_total = 0
    for chunk in pd.read_csv(csv_file_location_test, header=0, chunksize=chunk_size):
        X = np.array(chunk[create_headers(feature_length)])
        y = np.array(chunk[['label']])
        forest_result_class = forest_model.predict(X)


        forest_result_local = forest_model.predict_proba(X)
        numpy_local = np.array(forest_result_local)
        y= np.transpose(y)
        array =forest_result_class ==y
        chunk_accuracy += np.count_nonzero(forest_result_class ==y)
        chunk_total += np.shape(forest_result_class)[0]
        print(chunk_accuracy/chunk_total,'size',np.shape(forest_result_class)[0],'chunk acc',chunk_accuracy)
"""

        for number, prob in enumerate(forest_result_local):
                total +=1
                truth = y[0][number]
                classified = forest_result_class[number]
                prob_positive = prob[1]
                prob_negative = prob[0]
                positve_index = int(prob_positive*10)
                negative_index = int(prob_negative*10)
                if truth == 1:
                    positive_truth[positve_index]+=1
                    if truth == classified:
                        positive[positve_index]+=1
                        model_accuracy+=1
                if truth == 0:
                    negative_truth[negative_index]+=1
                    if truth == classified:
                        negative[negative_index]+=1
                        model_accuracy+=1
    print(model_accuracy/total)
    accuracy_positve = np.divide(positive, positive_truth)
    accuracy_negative = np.divide(negative, negative_truth)
    x_axis = np.arange(10)
    x_axis = x_axis / 10

    fig, ax = plt.subplots()
    line = ax.plot(x_axis, accuracy_positve, label='positve')
    line2 = ax.plot(x_axis, accuracy_negative, label='negative')
    ax.legend()
    plt.xlabel('probability from random forest')
    plt.ylabel('percentage of matches')
    plt.title('Random_forest n_estimator =1000 , max_features = number of features')
    plt.show()
"""




def pareto_optimal(result_forest,actual_cost,max_cost):


    pareto_optimal_solution = []
    pareto_optimal_old = []
    pareto_optimal_dub = []
    points = [[prob, cost] for prob, cost in zip(result_forest, actual_cost)]
    number_of_points = len(points)

    for i in range(number_of_points):

        cost = points[i][1]
        prob = points[i][0]
        if prob == 0.0 or cost >= 200:
            print(cost)

            continue
        else:
            pareto_optimal_old = pareto_optimal_solution
            pareto_optimal_dub = []
            old_number = len(pareto_optimal_solution)
            pareto_optimal_dub.append([prob, cost])
            for k in range(old_number):
                if pareto_optimal_solution[k][1] <= max_cost:
                    old_prob = prob + pareto_optimal_solution[k][0]
                    old_cost = cost + pareto_optimal_solution[k][1]
                    pareto_optimal_dub.append([old_prob, old_cost])
            pareto_optimal_solution = []
            sol_temp = pareto_optimal_dub + pareto_optimal_old
            sol_temp = sorted(sol_temp, key=itemgetter(1))
            last_index = 0
            if len(sol_temp) >= 1:
                for i in range(1, len(sol_temp)):
                    last_item = sol_temp[last_index]
                    present_item = sol_temp[i]
                    if last_item[0] < present_item[0]:
                        if last_item[1] < present_item[1]:
                            pareto_optimal_solution.append(last_item)
                            last_index = i
            if last_index < len(sol_temp):
                pareto_optimal_solution.append(sol_temp[last_index])
    print(pareto_optimal_solution)
    return pareto_optimal_solution


#cameras =read_images_binary(image_bin_location)

#image_array = get_details_from_database()
#image_array =add_feature_location(database_locatiom)

print('task1 complete')
#positive, negative = make_training_data(cameras, image_array)
print('task2 complete')

#headers=handle_data(positive,negative,feature_length,csv_file_location)
print('3')
#headers=handle_data_for_kmeans(positive,negative,feature_length,csv_file_location_kmeans)
print('4')
#test_data_positve,test_data_negative = make_test_data(point3D_location_overall,database_locatiom)
#headers = handle_data(test_data_positve,test_data_negative,feature_length,csv_file_location_kmeans_test)
print('all the csv files are ready')

###remove this
headers = create_headers(feature_length)
headers.append('label')
###
#clf=random_forest_chunks(headers,feature_length,csv_file_location,file_name_random_forest )
#k_means(headers,feature_length,csv_file_location,file_name_kmeans)
print("kmeans")
print("random forest saved")
k_means_broken_samples(headers,feature_length,csv_file_location_kmeans,file_name_kmeans,number_of_clusters)
#search_cost = search_cost_calculation(headers,feature_length,csv_file_location_kmeans,file_name_kmeans,number_of_clusters)
#prediction_forest(headers,feature_length,csv_file_location_kmeans_test,file_name_random_forest,clf)


#best_cost =prediction (headers,feature_length,csv_file_location_kmeans_test,file_name_random_forest,file_name_kmeans,number_of_clusters,search_cost,capacity)
















