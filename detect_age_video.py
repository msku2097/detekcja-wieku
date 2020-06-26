# uzycie
# python3 detect_age_video.py --face face_detector --age age_detector

# import paczek
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
	# definiujemy zakresy wieku
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# initializujemy liste
	results = []

	# pobieramy wymiary ramki a nastepnie budujemy bloba
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# wrzucamy bloba w siec neuronowa i odbieramy predykcje
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loopujemy po iteracjach
	for i in range(0, detections.shape[2]):
		# wyciagnij prawdopodobientwo
		confidence = detections[0, 0, i, 2]

		# odfiltruj slabe detekcje
		if confidence > minConf:
			# okreslamy koordynaty do narysowania obrysu
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# wyciągnij ROI 
			face = frame[startY:endY, startX:endX]

			# upewnij się, ze ROI jest odpowiednio wielkie
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# zbuduj bloba z ROI twarzy
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# tworzymy predykcje wieku i klasyfikujemy w odpowiedni zakres razem z pradopodobienstwem
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]

			# zbudujemy slownik z detekcji twarzy oraz obrysu z detekcja wieku i zaktualizujemy liste
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence)
			}
			results.append(d)

	# zwroc 
	return results

# konstruktor arugmentow
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#wczytujemy nasz zserializowany model detekcji twarzy
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# wczytujemy zserializowany model detekcji wieku
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initializuj strumien wideo 
print("[INFO] uruchamian wideo...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loopuj po ramkach wideo
while True:
	# zlap ramke i zmien jej rozmiar do rozmiaru 400 px
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# rozpoznaj twarze w ramkach. dla kazdej ramki dopisz wiek
	results = detect_and_predict_age(frame, faceNet, ageNet,
		minConf=args["confidence"])

	# loopuj po wynikach
	for r in results:
		# narysuj obrys wokol znalezionej twarzy
		text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# pokaz ramke na strumieniu
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# wcisnij q zeby przerwac
	if key == ord("q"):
		break
		
# czyscimy
cv2.destroyAllWindows()
vs.stop()