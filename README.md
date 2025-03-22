# Sleep_Detection
 
This project is a computer vision-based sleep detection system that monitors eye closure and triggers an alarm if the user appears to be drowsy. It uses OpenCV, dlib, and pygame to detect facial landmarks and generate alerts.  

## Features  
	Real-time eye tracking using OpenCV and dlib  
	Calculates the Eye Aspect Ratio (EAR) to determine drowsiness  
	Triggers an alarm sound when eyes are closed for a prolonged period  
	Works with laptop/PC cameras  

## Requirements  
	Python 3.x  
	OpenCV  
	dlib  
	numpy  
	pygame  

## How It Works  
	The camera captures the user's face.  
	Eye Aspect Ratio (EAR) is calculated using facial landmarks.  
	If EAR remains low for too long (indicating closed eyes), an alarm sound is played.  
