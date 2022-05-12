# %config IPCompleter.greedy=True
# импортируем библиотеки для работы с картинками и файловой системой
from unittest.main import MODULE_EXAMPLES 
import face_recognition
import os
import cv2

#папки с фото
KNOWN_FACES_DIR = "known_face"
UNKNOWN_FACES_DIR = "unknown_face"
TOLERANCE = 0.6 #порог отсечения
MODEL_LOCATION = "cnn" #название модели, использующиеся для распознавания лиц
MODEL_ENCODE = "large"

#обработка фото людей, которых мы будем искать
print("Loading known faces list")

known_names_list = {} # списки имен 
known_faces_list = {} # список найденных лиц

for name in os.listdir(KNOWN_FACES_DIR):
    for file_name in os.listdir(KNOWN_FACES_DIR + "/" + name):
        print(KNOWN_FACES_DIR + "/" + name + "/" + file_name)
        image = face_recognition.load_image_file(KNOWN_FACES_DIR+"/" + name + "/" + file_name)
        list_of_faces = face_recognition.face_encoding(image, model=MODEL_ENCODE)
        if list_of_faces != None and len(list_of_faces) > 0:
            encoded_face = list_of_faces[0]
        else:
            print("cant find face in this file ", KNOWN_FACES_DIR + "/" + name + "/" + file_name)
        # encoded_face = face_recognition.face_encodings(image)[0]
        known_faces_list.append(encoded_face)
        known_names_list.append(name)

# обработка незнакомых людей
print("Processing unknown faces")

for file_name in os.listdir(UNKNOWN_FACES_DIR):
    print("Unknown file name: ", file_name)

    image = face_recognition.load_image_file(UNKNOWN_FACES_DIR + "/" + file_name)
    face_locations_list = face_recognition.face_locations(image, model = MODEL_LOCATION)
    encodings_faces_list = face_recognition.face_encodings(image, face_locations_list, model = MODEL_ENCODE)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print("Found faces = ", len(encodings_faces_list))

    for face_encoding, face_location in zip(encodings_faces_list, face_locations_list):
        result = face_recognition.compare_faces(known_faces_list, face_encoding, TOLERANCE)
        match = None

        # если лица знакомы, товыделяем их синей рамкой, иначе красной
        if True in result:
            match = known_names_list[result.index(True)]

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [255, 0, 0]

            cv2.rectangle(image, top_left, bottom_right, color, 2)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        else:
            match = known_names_list[result.index(True)]

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 0, 255]

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, "Unknown", (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # вывод фото
        cv2.imgshow(file_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(file_name)