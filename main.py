import argparse
import time
import dlib
import cv2
import imutils
from imutils.video import VideoStream
from imutils import face_utils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to facial landmark model")
args = vars(ap.parse_args())

print("loading facial landmark model...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("loading camera...")
vstream = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the video stream and convert it to grayscale
	frame = vstream.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	
    # loop over the face detections
	for rect in rects:
		# use our model to predict the location of our landmark coordinates, then convert the prediction to an easily parsable NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates and draw them on the image
		for (sX, sY) in shape:
			cv2.circle(frame, (sX, sY), 1, (0, 255, 0), -1)
	
    # show the frame
	cv2.imshow("camera", cv2.flip(frame,1))
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vstream.stop()