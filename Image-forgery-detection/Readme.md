# Image-Forgery-Detection

Detecting whether image is digitally forged or not. Returns percentage Real or Fake

### Note: for better accuracy use dataset equally or nearly equally divided in negative and positive images

## Getting Started

### Generating dataset

* Put all the negaive and positive images in the "/dataset/Training_data/" directory

* Rename Image: 	

	* for positive - positive.{index no. eg. 1,2,3..}.jpg or jpeg
	
	* for negative - negative.{index no. eg. 1,2,3..}.jpg or jpeg
	
	* if they are in "PNG" format convert them in jpg format using "png_to_jpg.py" code and set the path accoringly

* Run "extraction_multiple.py" from object-detection/object_detection/ directory providing path of normal images dataset and save the extracted images in a new directory

* Train model using the new extracted images dataset

### Training 

* Train CNN model on the generated dataset using "trainModel.py" code(Set the no. of epochs based on total_loss value using hit and trial)
	
* Trained model will be saved in "Image-forgery-detection/model/" directory

### Testing

* save all the test Images in "/Image-forgery-detection/dataset/Testdataset" directory

* Run "runTest.py" code

