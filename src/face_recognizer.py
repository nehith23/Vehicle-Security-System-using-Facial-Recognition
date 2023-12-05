# class to handle face training and recognition requests from the server

print("=== import started (face_recognizer) ===")
import face_recognition
import os
import numpy
import pickle

# the following two lines are used to solve truncated file problem
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("=== import complete (face_recognizer) ===")

def mydebug(msg):
    print("DEBUG: "+str(msg))

class FaceRecognizer:


    def __init__(self, path_to_training_data):

        path_to_training_data = os.path.abspath(path_to_training_data)
        if os.path.exists(path_to_training_data) and os.path.isdir(path_to_training_data):
            self.is_valid = True
            self.path_to_training_data = path_to_training_data
        else:
            self.is_valid = False
            self.path_to_training_data = ''


    def get_person_index(self, person_name):

        for i in range(len(self.known_faces_encoded)):
            if self.known_faces_encoded[i][0] == person_name:
                return i
        self.known_faces_encoded.append(tuple([person_name, []]))
        return len(self.known_faces_encoded)


    def train_on_folder_tree(self, delete_image_without_faces = True):
    
        folder_path = self.path_to_training_data
        if not os.path.exists(folder_path): return

        known_faces = []
        self.known_faces_encoded = []
        for i in os.listdir(folder_path):
            res = self.__train_on_folder(folder_path + "/" + i, delete_image_without_faces)
            if len(res[1]) == 0: continue
            self.known_faces_encoded.append(res)


    def retrain_on_folder(self, person_name, delete_image_without_faces = True):

        # person_name = folder_path[folder_path.rfind("/")+1:]
        folder_path = self.path_to_training_data + "/" + person_name

        index_to_remove = self.get_person_index(person_name)
        self.known_faces_encoded.pop(index_to_remove)

        res = self.__train_on_folder(folder_path, delete_image_without_faces)
        if len(res[1]) != 0: self.known_faces_encoded.append(res)


    def __train_on_folder(self, folder_path, delete_image_without_faces = True):
        

        folder_path = os.path.abspath(folder_path)
        folder_path_exists = os.path.isdir(folder_path)
        pic_list = []
        pic_list_encoded = []

        if folder_path_exists:
            pic_list = os.listdir(folder_path)

        if (not folder_path_exists) or (len(pic_list) == 0):
            return ('',[])

        person_name = folder_path[folder_path.rfind("/")+1:]
        for i in range(len(pic_list)):
            file_path = folder_path + "/" + pic_list[i]
            # mydebug("loading image: "+ str(file_path))
            known_picture = face_recognition.load_image_file(file_path)
            # convert to 128 floating point representation
            known_picture_encoded = face_recognition.face_encodings(known_picture)
            if len(known_picture_encoded) == 0:
                # no face found
                print("WARNING: unable to add the file \"" + file_path + "\"")
                if delete_image_without_faces:
                    os.remove(file_path)
                    print("WARNING: file deleted \"" + file_path + "\"")
            else:
                # the following line will append all the items of "known_picture_encoded" to "pic_list_encoded"
                print("message: added new image file \"" + file_path + "\"")
                pic_list_encoded.extend(known_picture_encoded)
        return (person_name, pic_list_encoded)


    def train_on_image(self, person_name, image_path, delete_image_without_faces = True):
       

        index_to_insert = self.get_person_index(person_name)

        known_picture = face_recognition.load_image_file(image_path)
        known_picture_encoded = face_recognition.face_encodings(known_picture)
        if len(known_picture_encoded) == 0:
            # no face found
            print("WARNING: unable to add the file \"" + image_path + "\"")
            if delete_image_without_faces:
                os.remove(image_path)
                print("WARNING: file deleted \"" + image_path + "\"")
            return False
        else:
            # the following line will append all the items of "known_picture_encoded" to "pic_list_encoded"
            print("message: added new image file \"" + image_path + "\"")
            self.known_faces_encoded[index_to_insert][1].extend(known_picture_encoded)
            return True


    def remove_person(self, person_name, delete_all_images = False):
      
        index_to_remove = self.get_person_index(person_name)
        self.known_faces_encoded.pop(index_to_remove)
        if delete_all_images:
            os.system("rm -r " + self.path_to_training_data + "/" + str(person_name))


    def face_detection(self, picture_path, verify_all_faces = True, success_percentage = 0.6, distance_tolerance = 0.6):

        if success_percentage > 1: success_percentage = 1
        if (not os.path.exists(picture_path)) or (not os.path.isfile(picture_path)):
            return [(False, 'file does not exists')]

        unknown_picture = face_recognition.load_image_file(picture_path)
        unknown_picture_encoded = face_recognition.face_encodings(unknown_picture)
        if len(unknown_picture_encoded) == 0:
            # no face found
            return [(False, 'no face found')]

        return_value = []
        for i in self.known_faces_encoded:
            for j in unknown_picture_encoded:
            
                results = face_recognition.compare_faces(i[1], j, distance_tolerance)
                results = numpy.array(results)
                if results.sum() > (len(i[1]) * success_percentage):

                    if verify_all_faces:
                        return_value.append((True, i[0], results.sum() / len(i[1]), results.sum(), len(i[1])))
                    else:
                        return [(True, i[0], results.sum() / len(i[1]), results.sum(), len(i[1]))]

        if len(return_value) == 0:
            return [(False, 'unauthorized person')]
        else:
            return return_value


    def single_face_detection(self, picture_path, success_percentage = 0.6, distance_tolerance = 0.6):
    
        return self.face_detection(picture_path, False, success_percentage, distance_tolerance)


    def save_to_file(self, file_path = "trained_faces.pkl"):
 

        # open the file for writing
        file_object = open(file_path,'wb') 
        # this writes the object to the file named stored in the variable file_path
        pickle.dump((self.is_valid, self.path_to_training_data, self.known_faces_encoded), file_object)
        # here we close the file object
        file_object.close()


    def load_from_file(self, file_path = "trained_faces.pkl"):


        # Check if file exists and it contains data or not. If file not found or file is empty, then return False. Else read the content and load the object
        if os.path.isfile("./" + file_path) and os.path.getsize(file_path) != 0:
            # open the file in read mode
            file_object = open(file_path, 'rb')  
            # load the object from the file into class data members
            (self.is_valid, self.path_to_training_data, self.known_faces_encoded) = pickle.load(file_object)  
            return True
        return False


if __name__ == "__main__":
    a = FaceRecognizer('./z_face_testing/pictures_of_people_i_know')

    print("\n=== Training started ===")
    if not a.load_from_file():
        a.train_on_folder_tree()
        a.save_to_file()
    print("=== Training complete ===")
    # a.save_to_file()

    print("\n=== Face detection started ===")
    path_to_new_pics = os.path.abspath('./z_face_testing/unknown_pictures/')
    for i in os.listdir(path_to_new_pics):
        print("face recognition:", i, a.face_detection(path_to_new_pics + '/' + str(i), True, 0.0, 0.4))
        print("face recognition:", i, a.single_face_detection(path_to_new_pics + '/' + str(i), 0.0, 0.4))
    print("=== Face detection complete ===")

 