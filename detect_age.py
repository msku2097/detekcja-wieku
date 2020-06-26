#użycie
# python3 detect_age.py --image 1/costam.jpg --face face_detector --age age_detector

# importujemy paczki
import numpy as np
import argparse
import cv2
import os

#konstruktor argumentów
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# definiujemy zakresy wieku
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]

# wczytujemy nasz zserializowany model detekcji twarzy
print("[INFO] wczytuję model detekcji twarzy...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# wczytujemy zserializowany model detekcji wieku
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# wczytujemy obraz i budujemy bloba
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# wrzucamy bloba w siec neuronowa
print("[INFO] sprawdzam detekcje...")
faceNet.setInput(blob)
detections = faceNet.forward()

# loopujemy po detekcjach
for i in range(0, detections.shape[2]):
	# wyciągamy prawdopodobienstwo detekcji
	confidence = detections[0, 0, i, 2]

	# odfiltrowywujemy slabe detekcje
	if confidence > args["confidence"]:
		# okreslamy koordynaty do narysowania obrysu
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# wyciągamy ROI i budujemy obraz
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
			(78.4263377603, 87.7689143744, 114.895847746),
			swapRB=False)

		# tworzymy predykcje wieku i klasyfikujemy w odpowiedni zakres razem z pradopodobienstwem
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]

		# wyrzuc na konsole znaleziony przedzial wieku
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))

		# narysuj ramke
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# wyswietl obraz
cv2.imshow("Image", image)
cv2.waitKey(0)