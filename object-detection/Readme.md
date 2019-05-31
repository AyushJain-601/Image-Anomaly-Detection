# Object-Detection

Detects trained object in the input image.

### NOTE

* "object_detection" directory mentioned below is the directory from https://github.com/tensorflow/models.git. So, do clone and copy that directory(already mentioned in pre-requisite).
* Extraction_single and Extraction_multiple scripts are inside temp folder

## Getting Started

### Generating Dataset

* Save all the training images in the "object-detection/object_detection/Images/train" directory

* Save all the validation images in the "object-detection/object_detection/Images/test" directory

* NOTE: To train object detection model all the training and validation images has to be annotated before i.e. generate Bounding box and label them based on their class using "labelImg tool" save all the "XML" in their respected image directory.

* Convert all the "XML" file to "CSV" format using "xml_to_csv.py" code

* Now generate ".Tfrecord" format of these "CSV" files using "generate_tf.py" code

	* python generate_tfRecord.py --csv_input=/object_detection/Images/train_labels.csv  --output_path=/object_detection/Images/test_labels.csv
	
* This .tfrecord files are the final dataset containing annotated image path and bounding box coordinates

### Generate labelmap.pbtxt file

* Generate labelmap.pbtxt file containing no. of categories check "object-detection/object_detection/training/labelmap.pbtxt"

### Configure Model
* Save model to be used in "/home/ashish/Anomoly-detection/object-detection/object_detection/training/" directory in this case we have used "faster_rcnn_inception_v2_coco_2018_01_28" pre-made RCNN-tensorflow model

* copy config file of the model used from "/home/ashish/Anomoly-detection/object-detection/object_detection/samples/" save in the same "training" directory

* Configure ".config" file of the model by making following changes: 

	* "num_classes:" equal to no. of classes or category of object
	
	* "fine_tune_checkpoint:" path to model in our case "object-detection/object_detection/training/faster_rcnn_inception_v2_coco"
	
	* inside "train_input_reader:" function
		* change "input_path:" path to "train.record" file in our case "object-detection/object_detection/Images/train.record"
		* change "label_map_path:" path to "labelmap.pbtxt" file in our case "object-detection/object_detection/training/labelmap.pbtxt"
		
	* inside "eval_input_reader:" function 
		* change "input_path:" path to "test.record" file in our case "object-detection/object_detection/Images/test.record"
		* change "label_map_path:" path to "labelmap.pbtxt" file in our case "object-detection/object_detection/training/labelmap.pbtxt"
		
	* save the changes

### Train Model
* Inside "/object_detection/" directory: 

	* Run python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
	
	* Run tensorboard --logdir=/Images in a seperate terminal inside same /object_detection directory
	
	* Stop training the function when total loss graph saturates
	
	* trained model will be saved in /object_detection/Images folder

### Generate Inference Graph
* Inside "/object_detection/" directory: 

	* run and change path according to the path where trained model is saved python3 export_inference_graph.py --input-type image_tensor --pipeline_config_path /training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix Images/model.ckpt-11579 --output_directory Images


### Test Dataset
* Save all the images inside "object_detection/Images" directory

* Inside "object_detection" directory:

	* For multiple images run python3 extraction_multiple.py providing path of directory
	
	* For single image run python3 extraction_single.py providing path of single file
	
	* Extracted images are saved in /object-detection/image_extracted/ directory
