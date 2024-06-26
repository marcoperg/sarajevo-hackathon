from flask import Flask

import face_recognition
import cv2
import numpy as np

from transformers import CLIPProcessor, CLIPModel

#from multiprocessing import Process, Value, Lock, Manager
#from multiprocessing.shared_memory import ShareableList

from threading import Thread, Lock


font = cv2.FONT_HERSHEY_DUPLEX

def read_class_labels(filepath):
    with open(filepath, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels


clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
age_classes_file = "age_classes.txt"  # Replace with the actual path
age_labels = read_class_labels(age_classes_file)

emotion_classes_file = "emotion_classes.txt"  # Replace with the actual path
emotion_labels = read_class_labels(emotion_classes_file)

def get_emotion_and_age(img):
    inputs = clip_processor(text=age_labels, images=img, return_tensors="pt", padding=True)
    outputs = clip(**inputs)
    age_logits = outputs.logits_per_image

    inputs = clip_processor(text=emotion_labels, images=img, return_tensors="pt", padding=True)
    outputs = clip(**inputs)
    emotion_logits = outputs.logits_per_image
    return (age_labels[age_logits.softmax(dim=1).argmax().item()], emotion_labels[emotion_logits.softmax(dim=1).argmax().item()])


def detection_loop(face_names, emotions, ages, lock):
    # Get a reference to webcam #0 (the default one)

    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("image_training/marco.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("image_training/javi.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Load a second sample picture and learn how to recognize it.
    sanchez_image = face_recognition.load_image_file("image_training/david.jpg")
    sanchez_face_encoding = face_recognition.face_encodings(sanchez_image)[0]


    # Load a second sample picture and learn how to recognize it.
    macron_image = face_recognition.load_image_file("image_training/julian.png")
    macron_face_encoding = face_recognition.face_encodings(macron_image)[0]
    # Function to read class labels from a text file

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding,
        sanchez_face_encoding,
        macron_face_encoding
    ]
    known_face_names = [
        "Marco",
        "Jose Javier",
        "David",
        "Julian"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        _, frame = video_capture.read()
        

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame#small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            age, emotion = None, None
            if len(face_locations) > 0:
                first_face = face_locations[0]
                age, emotion = get_emotion_and_age(small_frame[first_face[0]:first_face[2],first_face[3]:first_face[1]])

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            with lock:
                face_names.clear()
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(np.array(known_face_encodings), np.array(face_encoding))
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(np.array(known_face_encodings), np.array(face_encoding))
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

        process_this_frame = not process_this_frame
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, str(age), (30, 30), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, str(emotion), (60, 60), font, 1.0, (255, 255, 255), 1)
        emotions.clear()
        ages.clear()
        emotions.append(emotion)
        ages.append(age)


        # Display the results
        with lock:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def run(face_names, emotions, ages, lock):
    app = Flask(__name__)
    @app.route("/users")
    def users_recog():
        with lock:
            return str(face_names[0])
    
    @app.route("/emotion")
    def emotion():
        with lock:
            return str(''.join(emotions))

    @app.route("/age")
    def age():
        with lock:
            return ages[0]



    app.run(host="0.0.0.0", port=5005)

if __name__ == "__main__":
    face_names = []
    emotions = []
    ages = []
    lock = Lock()

    p2 = Thread(target=run, args=(face_names, emotions, ages, lock))
    p2.start()
    detection_loop(face_names, emotions, ages, lock)
    p2.join()