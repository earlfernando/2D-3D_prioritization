#Librariers to be imported
import  os
import numpy as np
import sys
import sqlite3
import numpy.matlib
import math
from quaternion import quat2mat, inner_product_calculation, mat_quat_mul,quaterinion_multiplication,mult,inverse

#nvm file location
nvm_file_location = "/home/earl/Thesis/GreatCourt/reconstruction.nvm"
image_file_text_location ="/home/earl/Thesis/local_tmp/image_input.txt"
image_file_text_location1 ="/home/earl/Thesis/GreatCourt/image_input.txt"
dir_for_counting= "/home/earl/Thesis/GreatCourt/"
database_locatiom = "/home/earl/Thesis/GreatCourt/Sequence_Database_greatCourt.db"
camera_write_text_file_location = "/home/earl/Thesis/GreatCourt/cameras_input_for_triangulate.txt"
image_write_text_file_location = "/home/earl/Thesis/GreatCourt/image_input_for_triangulate.txt"
new_nvm_file_location =  "/home/earl/Thesis/GreatCourt/reconstruction_new.nvm"

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


#Methods
def image_dir_counting(dir,sequence,image_data,database_location):
    r = []
    file_count =0
    sequence = np.array(sequence)
    image_data = np.array(image_data)
    camera_id_array =[]
    sorted_sequence = []
    camera_id =0
    sorted_image_data =[]
    database = sqlite3.connect(database_location)
    cursor = database.cursor()
    test_images_id = []
    training_images_id = []

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

    for root, dirs, files in os.walk(dir):
        for name in sorted(dirs):
            if name == 'videos':
                continue
            local_path =os.path.join(root, name)
            path, directories, images = next(os.walk(local_path))
            for image in sorted(images):
                camera_id += 1
                local_image_name = local_image_name = name+'/'+image
                if local_image_name in test_images_str:
                    print(image)
                else:

                    local_image_name = name+'/'+image
                    image_index = np.where(sequence==local_image_name)
                    if len(image_index)!=1:
                        print("error")
                    sorted_image_data.append(image_data[image_index])
                    camera_id_array.append(camera_id)
                    sorted_sequence.append(local_image_name)
            file_count += len(images)
    return file_count, sorted_image_data,sorted_sequence,camera_id_array

def write_image_seqence(image_file_text, sequence_location, image_name,seqence):

    seqence_length = len(seqence)
    if sequence_location <= 10:
        image_start = 0
    else:
        image_start = sequence_location-10

    if sequence_location >= seqence_length-10:
        image_end = seqence_length
    else:
        image_end = sequence_location+10
    #writes the images before
    for i in range (image_start,sequence_location):
        local_string = image_name+' '+ seqence[i]+'\n'
        image_file_text.write(local_string)
    #writes the images after
    for i in range (sequence_location+1,image_end):
        local_string = image_name + ' ' + seqence[i] + '\n'
        image_file_text.write(local_string)

def convert_list_str_to_int(image_data):
    for number, data in enumerate(image_data):
        test_number = list(map(float, data))
        image_data[number] = test_number

    return  np.array(image_data)

def split(write_nvm=False):
        seqence =[]
        image_data =[]
        with open(nvm_file_location,'r') as nvm_file:
            for line in nvm_file:
                if line.startswith("se"):

                    line = line.replace('.jpg','.png')
                    split_data =line.replace('\t', ' ').rstrip().split(' ')
                    seqence.append(split_data[0])
                    image_data.append(split_data[1:len(split_data)])
                    #seqence.append(split_data[0].replace('.jpg','.png'))
        with open(image_file_text_location1,'w') as image_file_text :
            for sequence_location,image_name in enumerate(seqence):
                write_image_seqence(image_file_text, sequence_location, image_name,seqence)
        if write_nvm:
            write_line =''
            with open(nvm_file_location, 'r') as nvm_file:
                with open(new_nvm_file_location,'w') as new_nvm_file:
                    for line in nvm_file:
                      if line.startswith("se"):
                         line = line.replace('.jpg', '.png')
                         write_line = write_line+ line

                      else:
                          write_line = write_line+line

                    new_nvm_file.write(write_line)



        return seqence, convert_list_str_to_int(image_data)



def blob_to_array(blob, dtype, shape=(-1,)):
    IS_PYTHON3 = sys.version_info[0] >= 3
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def array_to_blob(array):
    IS_PYTHON3 = sys.version_info[0] >= 3
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)




"""def database_change(sorted_image_data, sorted_sequence, camera_id_array,database_location):

    ### camera id sequence is faulty check get_details from camera, there is a order mismatch between the camera_id
    ### and image_id (see the database)
    database = sqlite3.connect(database_locatiom)
    cursor =database.cursor()
    params_storage =[]

    cursor.execute('''SELECT camera_id,params, prior_focal_length FROM cameras''')
    for row in cursor:
        # row[0] returns the first column in the query (name), row[1] returns email column.
        #print(str(row[2]),type(row[1]))
        params = blob_to_array(row[1], np.float64)
        params_storage.append(params)
        #print('{0} : {1} :{2}'.format(row[0], row[1],row[2]))


    for id in camera_id_array:
        local= sorted_image_data[id-1]
        focal_length = local[0][0]
        ####focal_length = '{0:f}'.format(focal_length)
        #distortion = local[0][len(local[0])-2]
        #local_params =params_storage[id-1]
        #camera_center_x = local[0][len(local[0]) - 4]
        #camera_center_y= local[0][len(local[0]) - 3]
        #params_to_update = np.array([focal_length, camera_center_x, camera_center_y, distortion])

        #params_to_update = np.array([focal_length, local_params[1],local_params[2],distortion])

        #blob = array_to_blob(params_to_update)
        cursor.execute('''UPDATE Cameras SET prior_focal_length = ? WHERE camera_id =?''', (focal_length, id))


    database.commit()
    database.close()"""


def get_details_from_database():

    database = sqlite3.connect(database_locatiom)
    cursor =database.cursor()

    cursor.execute('''SELECT image_id,camera_id,name FROM images''')
    test_images_str = test_images_string()

    with open(image_write_text_file_location,'w+') as image_write:
            with open(camera_write_text_file_location,'w+') as camera_write:
                for row in cursor:
                    if row[2] in test_images_str:

                        continue
                    else:
                        camera_write.write(str( row[2])+'\n')
                        if row[1] != row[0]:
                           print(row[1])
                        image_write.write(str(row[2]
                                              )+' '+str(row[0])+' '+str(row[1])+'\n')
    database.close()


def get_details_from_database_1():

    database = sqlite3.connect(database_locatiom)
    cursor =database.cursor()

    cursor.execute('''SELECT image_id,data,rows FROM  descriptors''')

    with open(image_write_text_file_location,'w+') as image_write:
            with open(camera_write_text_file_location,'w+') as camera_write:
                for row in cursor:
                   print(str(row[0]),np.shape(blob_to_array(row[1],dtype=np.uint8)),str(row[2]),blob_to_array(row[1],dtype=np.uint8))
                   # camera_write.write(str( row[2])+'\n')
                if row[1] != row[0]:
                      print(row[1])
                    #image_write.write(str(row[2]
                                         # )+' '+str(row[0])+' '+str(row[1])+'\n')
    database.close()

"""
def get_details_from_database(sorted_image_data, sorted_sequence, camera_id_array,database_location):
    database = sqlite3.connect(database_locatiom)
    cursor =database.cursor()
    camera_id_array =[]
    image_id_array=[]
    image_id_seq =[]

    cursor.execute('''SELECT image_id,camera_id,name FROM images''')
    for row in cursor:
        # row[0] returns the first column in the query (name), row[1] returns email column.

        camera_id_array.append(row[1])
        image_id_array.append(row[0])
        image_id_seq.append(row[2])
        print('{0} : {1} :{2}'.format(row[0], row[1],row[2]))


    with open(image_write_text_file_location,'w') as image_write:
            with open(camera_write_text_file_location,'w') as camera_write:
                for number,camera_id in enumerate(camera_id_array):
                    camera_write.write(str(camera_id)+'\n')
                    image_write.write(str(image_id_seq[number])+' '+str(image_id_array[number])+'\n')



    database.commit()
    database.close()
"""



def writing_as_per_distance_and_angle(sequence,image_data):
    overall_quaterion = image_data[:,1:5]
    overall_rotation =[quat2mat(i)for i in overall_quaterion]
    overall_camera_center = image_data[:,5:8]
    with open(image_file_text_location1, 'w') as image_file_text:
     for id, image_name in enumerate(sequence):
        local_image_rotation = overall_rotation[id]
        local_image_camera_center = overall_camera_center[id]
        image_distances = calculate_distance(local_image_camera_center,overall_camera_center)
       # angles = calculate_angle(local_image_rotation,overall_rotation)
        angles_new = quaterion_angle(overall_quaterion[id],overall_quaterion,id)


        angles_mask = angles_new<=30
        distance_mask = image_distances<=10

        mask = np.logical_and(angles_mask,distance_mask)
        image_location= np.where(mask)[0]
        image_location= list(image_location)

        image_location.remove(id)
        for i in image_location:
            local_string = sequence[id] + ' ' + sequence[i] + '\n'
            image_file_text.write(local_string)

def writing_as_per_location(sequence,image_data):

    with open(image_file_text_location1, 'w') as image_file_text:
     for id, image_name in enumerate(sequence):
            print(image_name,id)
            write_image_seqence(image_file_text,id,image_name,sequence)









def calculate_distance(point, array):

    point_array = np.matlib.repmat(np.asmatrix(point),np.size(array,0),1)
    distance = array-point_array
    distance = np.square(distance)
    distance= np.sum(distance,axis=1)
    distance = np.sqrt(distance)

    return np.array(distance).flatten()

def stacking_matrices(matrix,length):
    return_matrix=[]
    for i in range(length):
        return_matrix.append(matrix)

    return return_matrix

def calculate_angle(local_image_rotation, overall_rotation):
    local_roation_array = stacking_matrices(local_image_rotation,len(overall_rotation))

    theta_1 = mat_quat_mul(local_roation_array,overall_rotation)
    print('done')
    theta_2 = [quaterinion_multiplication(i,j)for i,j in zip(local_roation_array,overall_rotation)]
    for i in range(len(theta_2)):
        print(theta_1[i],theta_2[i])





def quaterion_angle(loca_quater,overall_quaterion,id):
    return_array =[]
    for j,i in enumerate(overall_quaterion):
        diff = mult(loca_quater,inverse(i))
        if j ==id:

            w= 1
        else :
            w=diff[0]

        theta = 2*math.acos(w)
        return_array.append(theta)

    return  np.degrees(return_array)






#Main loop
sequence, image_data =split(write_nvm=True)
file_count, sorted_image_data, sorted_sequence, camera_id_array = image_dir_counting(dir_for_counting,
                                                                                     sequence,image_data,database_locatiom)
writing_as_per_location(sequence,image_data)
#writing_as_per_distance_and_angle(sequence,image_data)
#database_change(sorted_image_data,sorted_sequence,camera_id_array,database_locatiom)
get_details_from_database()

