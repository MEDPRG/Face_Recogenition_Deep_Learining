import uuid

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import sys

# PREPARING THE MODEL

# Initializing MTCNN and InceptionResnetV1 (keep_all = False) it means that if 1 image contain multiple faces then it
# will keep just 1 face from that which means keep all these false that's means keep only 1 face
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
# (keep_all = True) it will keep all the images in the form of list
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
# Initializing the class, and we're passing pretrained model which is vggface2 and if this model isn't already
# downloaded on your computer then when you first run it. it will download automatically
resnet = InceptionResnetV1(pretrained='vggface2').eval()

while True:
    while True:
        i = int(input('1- Add a person\n2- Training Model \n3- Live camera\n'))
        if i == 1 or i == 2 or i == 3:
            break

    match i:
        case 1:
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            if not (os.path.exists('photos')):
                # Make the directories
                os.makedirs('photos')
            else:
                print('exists')

            name = str(input("Enter The Name of the person that you want to capture: ")).lower()
            Knowing_images_path = 'photos/' + name
            path = Knowing_images_path
            if not (os.path.exists(Knowing_images_path)):
                # Make the directories
                os.makedirs(Knowing_images_path)
            else:
                print('already exist')

            count = 1
            while cap.isOpened():
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                capt_frame = frame.copy()
                for (x, y, w, h) in face_rect:
                    if count <= 90:
                        count += 1
                        img_name = os.path.join(path, str(int(uuid.uuid1())) + '.jpg')
                        cv2.imwrite(img_name, capt_frame)
                        print(count)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0XFF == ord('q') or count == 90:
                    break

            print('finished')
            # Release the webcam
            cap.release()
            # Close the image show frame
            cv2.destroyAllWindows()
        case 2:
            # Read data from folder
            # Taking all the images from the photos folder into dataset
            dataset = datasets.ImageFolder('photos')  # photos folder path
            # taking the name of the folders images.
            idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names


            def collate_fn(x):
                return x[0]


            # get the pictures in the pil image format(faces)
            loader = DataLoader(dataset, collate_fn=collate_fn)

            name_list = []  # list of names corresponding to crapped photos
            embedding_list = []  # list of embedding matrix after conversion from cropped face to embedding matrix using resnet

            for img, idx in loader:
                # cropped the face form the image. keep only 1 face from img that we got from the loader
                face, prob = mtcnn0(img, return_prob=True)
                # if there is a face and its probability is over 0.92 we pass the face in the resnet model
                if face is not None and prob > 0.92:
                    # we pass face.unsqueeze(0) because resnet is expecting an embed dimensional
                    emb = resnet(face.unsqueeze(0))  # provide an embedding
                    # appending each time an embed into the embedding list using detach because it's just release
                    # some memory as embedding which the required grid will set to false and this will release some
                    # memory
                    embedding_list.append(emb.detach())
                    # set the names which is saved in the idx_to_class list depending on the idx
                    name_list.append(idx_to_class[idx])

            # Save the data
            data = [embedding_list, name_list]
            torch.save(data, 'data.pt')  # Saving data.pt file
            # THE END OF PREPARING THE MODEL
        case 3:
            # Using webcam recognize face

            # Loading data.pt file inside a load list(matrix)
            load_data = torch.load('data.pt')
            embedding_list = load_data[0]
            name_list = load_data[1]
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cam = cv2.VideoCapture(0)

            while True:
                ret, frame = cam.read()
                if not ret:
                    print('fail to grab frame, try again')
                    break

                img = Image.fromarray(frame)  # Convert the frame from NumPyarray to an image
                # mtcnn(img, return_prob=True) will return multiple faces if the image contain multiple faces with
                # their probability ("img_cropped_list") list of multiple faces and  ("prob_list") list of their
                # probabilities

                # Uses MTCNN to detect faces in the input image (img).
                # img_cropped_list contains the cropped face images.
                # prob_list contains the corresponding probabilities/confidences for each detected face.
                img_cropped_list, prob_list = mtcnn(img, return_prob=True)

                if img_cropped_list is not None:
                    # Checks if there is at least one face detected.
                    # If faces are detected, retrieves the bounding boxes (`boxes`) for the detected faces.
                    boxes, _ = mtcnn.detect(img)  # get the list of different boxes that we'll draw on the faces

                    # we'll loop through all the probabilities in ("prob_list")
                    # Loops through the detected faces and their corresponding probabilities.
                    for i, prob in enumerate(prob_list):
                        # Filters faces based on a probability threshold (90% confidence).
                        if prob > 0.90:
                            # Uses a ResNet model (resnet) to extract embeddings from the cropped face.
                            emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                            dist_list = []  # list of matched distances, minimum distance is used to identify the person

                            for idx, emb_db in enumerate(embedding_list):
                                # Compares the extracted embedding with embeddings in the database (embedding_list)
                                # using Euclidean distance.
                                dist = torch.dist(emb, emb_db).item()
                                dist_list.append(dist)

                            # Identifies the person based on the minimum distance.
                            min_dist = min(dist_list)  # get the minimum dist value
                            min_dist_idx = dist_list.index(min_dist)  # get the minimum dist index
                            name = name_list[min_dist_idx]  # get the name of the corresponding to minimum dist

                            # Draw Box on the Face:
                            # Retrieves the bounding box for the detected face.
                            box = boxes[i]  # get the box of the face detected from the list of boxes
                            pt1 = (box[0], box[1])
                            pt2 = (box[2], box[3])

                            # Store Original Frame:
                            # Creates a copy of the original frame for further processing or visualization.
                            original_frame = frame.copy()
                            if min_dist < 0.90:
                                cv2.flip(frame, 180)
                                # show image back to screen
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                # Detect the face in the image
                                face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

                                for (x, y, w, h) in face_rect:
                                    cv2.putText(frame, name + ' ' + str(int(min_dist*100))+'%', (x, y - 10),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), thickness=2)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                cv2.imshow("IMG", frame)
                k = cv2.waitKey(1)
                if (k % 256) == 27:  # ESC
                    print('Esc pressed, closing... ')
                    break
                elif (k % 256) == 32:  # space to save image
                    print('Enter your name: ')
                    name = input()

                    # Create directory if not exists
                    if not os.path.exists('photos/' + name):
                        os.mkdir('photos/' + name)

                    img_name = f'photos/{name}/{int(time.time())}.jpg'
                    cv2.imwrite(img_name, original_frame)
                    print(f'saved: {img_name}')

            cam.release()
            cv2.destroyAllWindows()

    while True:
        C = int(input('Do you want to go back to the menu:\n1- Yes\n2- No\n'))
        if C == 1 or C == 2:
            break

    if C == 2:
        print('exit')
        break

sys.exit(0)
