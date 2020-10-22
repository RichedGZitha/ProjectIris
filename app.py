import cv2, time
#import pandas as pd

# params: Front face classifier and an Image (Frame).
# @return: An image with rectangles drawn around the faces of people.
def detectFaces(classifer, oriImage):
	# detct all the faces in image.
	faces = classifer.detectMultiScale(oriImage, scaleFactor = 1.05, minNeighbors = 5)

	# draw rectangle around each face in frame.
	for x,y,w,h in faces:
		oriImage = cv2.rectangle(oriImage, (x,y), (x + w, y + h), (0,255,0),3)
 
	return oriImage


# Calculates the frame difference of two frames
def calculateFrameDifference(ref_Image, oriImage):
	frame_Diff = cv2.absdiff(ref_Image, oriImage)

	return frame_Diff

# Calculates Threshold to remove shadows and white noise.
def thresholdCalculator(frame_Diff, threshold_value,maxVal,thresholdingMode):
	frame_Threshold = cv2.threshold(frame_Diff, threshold_value, maxVal, cv2.THRESH_BINARY)[1]

	frame_Threshold = cv2.dilate(frame_Threshold, None, iterations = 0)

	return frame_Threshold


# Determines the counters of an object.
def calculateContours(frame_threshold):
	contours, taxonomy = cv2.findContours(frame_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	return (contours, taxonomy)

def main():

	# Face classifer initialisation.
	face_cascadeClass = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

	ref_Image = None

	# Time  and Pandas objects.
	status_list = [None, None]
	times = []

	# Pandas Data Frame Object. Stores time when object is in front of the camera.
	#finalData = pd.DataFrame({"Start":[1]}, {"End":[1]})

	# Capture video from video source.
	video   = cv2.VideoCapture(0) #("videos/AIDrone.mp4")

	while(True):

		read_success, frame = video.read()
		status = 0

		if read_success == False or (cv2.waitKey(1) & 0xFF == ord('p')):
			
			break

		# Color Image to be used later.
		frame_color = frame

		# Convert Image to Grascale.
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Convert frame to a Gaussian Blob. This increases the efficiency of the motion detector.
		frame = cv2.GaussianBlur(frame,(25,25),0)

		# Setting the initial state Image Immediately when the camera is switched. 
		if ref_Image is None:
			ref_Image = frame
			continue

		# Detect faces and draw rectangles around the face.
		#img = detectFaces(face_cascadeClass, frame) # Slow as hell. WHY ?

		# calculate the differnce:
		frame_Diff = calculateFrameDifference(ref_Image, frame)

		# Define a threshold to remove the shadows and other noise.
		frame_Threshold = thresholdCalculator(frame_Diff, 30, 255, "TEMP")
		
		# Define the contour area. Add borders to object.
		contours, taxonomy = calculateContours(frame_Threshold)
		
		# Append contours to the original grey scale Image.
		for cont in contours:
			if cv2.contourArea(cont) < 500:
				continue

			status = 1

			(x,y,w,h) = cv2.boundingRect(cont)
			cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0),3) 

			'''status_list.append(status)

			status_list[-1] = status_list[-2:]

			if status_list[-1] == 1 and status_list[-2] == 0:
				times.append(datetime.now())

			if status_list[-1] == 0 and status_list[-2] == 1:
				times.append(datetime.now())'''

		cv2.imshow("Kiing On High", frame)
		
	video.release()
	cv2.destroyAllWindows()

	# Create a pandas data frame object:
	'''for i in range(0,len(times),2):
		finalData = finalData.append({"Start":times[i]}, {"End":times[i + 1]}, ignore_index = True)

	# Save to CSV File.
	finalData.to_csv("data/Times.csv")'''


if __name__ == '__main__': 
	main()