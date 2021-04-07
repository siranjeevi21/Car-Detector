import cv2




video = cv2.VideoCapture('p.mp4')


classifier_file = 'cars.xml'

pedestrian_tracker_file = 'pedestrian_tracking_datasets.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
	(read_succesful, frame) = video.read()
	if read_succesful:
		grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		break	
	pedestrians = car_tracker.detectMultiScale(grayscaled_frame)
	cars = car_tracker.detectMultiScale(grayscaled_frame)

	
	for (x, y, w, h) in cars:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0, 255),2)	

	for(x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


	cv2.imshow('output_video', frame)
	cv2.waitKey(1)	



















































print("Code Comleted")