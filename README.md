Author : Arnaud TECHER
This small project is a learning project for Computer Vision application. 
The goal of this project is to manipulate a Deep Learning model for computer vision apply to autonomous cars.
The data used in this project are public data available on Kaggle web site. 

The data contains a video of a dash cam in the car, the goal of this project is to develop a Deep Learning Model that can identify 
key elements around the vehicule based on the image of the dashcam. The key elements are :
    - car
    - truck
    - person 
    - bicycle 
    - traffic light
The base model used here is the YOLO model from Ultralytics, which is a deep learning model pre-trained for computer vision application. 
Two python has been developed. The first one is use to run an inference of the YOLO model in order to identify the key elements on the road.
The second python file use the database in order to fine-tune the model using a training base in the data and a test base. The goal is to learn 
how to train a base model on a specific dataset. 


![Demo](yolo_inference.gif)