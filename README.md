# Image-Anomaly-Detection

Detects whether image is digitally morphed or not.

### Note

runTest.py is complete script of with object detection embedded in it.
To Run:
	* Just provide path to the trained model(Both obeject detection and image forgery model) and provide path of test image

## Getting Started

### pre-requisite

* pip3 install numpy sklearn opencv-python PIL tensorflow matplotlib utils tqdm tflearn pdf2image --user

* git clone https://github.com/tensorflow/models.git

* save all the object detection files in the "models/research/object_detection" directory

* (object-detection) For dataset xml file creation use "git clone https://github.com/tzutalin/labelImg.git" then inside labelimg directory run "python3 labelImg.py"

### Image-forgery-detection

* Functionality: Detect whether input image is digitally morphed or not i.e. it is Real or Fake

* Method Used: CNN trained on ELA Images

* Logic: - Jpeg or jpg format images pixel density decreases upon resaving

	* Images are resaved with particular loss percentage
	
	* Areas of anomoly have higher pixel density as compared to the other part of image
	
	* Hence, CNN is trained using these positive as well as negative image dataset 
	
* For working check Readme file inside the folder

### Object-detection

* Functionality: Detect and extract the object in the input image creating bounding box around it

* Method Used: Faster-rcnn-inception-v2-coco pre-trained model

* Logic: - Model is trained on our own custom annotated image dataset after implementing some changes

	 * It extract features of the annotated image(object) based on which it detects objects and labels them in the output image
	 
* For working check Readme file inside the folder 
