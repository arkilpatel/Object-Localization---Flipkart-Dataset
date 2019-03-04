Object Localization Problem. Draw bounding boxes around the objects.

Resized input training data: https://drive.google.com/file/d/1EcX3lKwMckKyYM4SFZx1f5ZwPATWDje6/view?usp=sharing
test data: https://drive.google.com/file/d/1KIlIpHJ5vHq3p41uA2yedq9m4GYXc0J_/view?usp=sharing

training.csv: https://drive.google.com/file/d/1KIlIpHJ5vHq3p41uA2yedq9m4GYXc0J_/view?usp=sharing
test.csv: https://drive.google.com/file/d/1KIlIpHJ5vHq3p41uA2yedq9m4GYXc0J_/view?usp=sharing

Approach:
1) Resized Image to 96x128x3.
2) Augmented the images by rotating the images by 180 degrees and taking their mirror images.
3) Split the train data into train and val data with a val split of 0.1.
4) Trained the model from scratch (No pretrained model was used) using the specified model placeholder whose summary is provided in the source code.
5) Custom IoU metric was used to evaluate the validation data during training.

Feature Selection:
1) No feature selection was done.

Tools:
1) The code was written on Google Colaboratory platform using the Keras framework.
2) Standard python libraries like Numpy, Pandas, SKlearn, etc were used.
