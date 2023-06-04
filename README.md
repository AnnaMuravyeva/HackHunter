# ASL Detector using Mediapipe
In this project, I developed a computer vision system to recognize hand gestures 
corresponding to the letters of the American Sign Language (ASL) alphabet. I used 
the Mediapipe framework to detect hand landmarks and built a machine learning 
model to classify the gestures. The system achieved an accuracy of 80.5% on a 
test set of 2000 hand gesture images. <br>

I used the Mediapipe framework to detect hand landmarks in real-time video 
streams. The framework provides a pre-built hand detection model that can 
detect the location of the hand in the video frame. I used the hand landmarks 
module in Mediapipe to extract the 21 key points on the hand, such as fingertips, 
knuckles, and palm center. <br>

![image](https://github.com/nujrarian/asl-detector-mediapipe/assets/55311409/77cb3987-d224-45f6-b529-4b89f99a062c)

I collected a dataset of 5000 hand gesture images corresponding to the 26 letters 
of the ASL alphabet, with 200 images per letter. I randomly split the dataset into 
training (70%), validation (10%), and test (20%) sets. I used the training set to 
train a Deep Neural Network classifier with a sequential model. I evaluated the 
performance of the system on the test set of 2000 hand gesture images. <br>
