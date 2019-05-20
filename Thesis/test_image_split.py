import  shutil
dir_for_counting= "/home/earl/Thesis/GreatCourt/"
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

files = test_images_string()
print(test_images_string())
for f in files:
    image = f.split('/')
    destination_path ="/home/earl/Thesis/GreatCourt/test"+'/'+image[0]+'/'
    shutil.move(dir_for_counting+f,destination_path )