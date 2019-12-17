### TLT Demo Step by Step (Pascal VOC Dataset)

A more updated and detailed guide is available here: https://medium.com/@Smartcow_ai/nvidia-transfer-learning-toolkit-a-comprehensive-guide-75148d1ac1b

Before beginning we need to assure that we have a working ``API_KEY``, login to [ngc.nvidia.com](https://ngc.nvidia.com/) and get a key if you don't have one. And create a Directory Structure as shown below.

````bash
./workspace
	|--dataset
	|--pretrained_model
    	|--trained_model
	|--scripts
	|--spec_files
	|--tf_records
````



**Pull the NVIDIA Transfer Learning Toolkit Container.**

````bash
docker pull nvcr.io/nvidia/tlt-streamanalytics:v1.0_py2
````

To Run the **NV_TLT Container** Execute below command. (remember to mount the directory we just created inside the container).

````bash
docker run --runtime=nvidia -it -v '/home/user_name/workspace/:/workspace/' \
nvcr.io/nvidia/tlt-streamanalytics:v1.0_py2 /bin/bash
````

By now you must have entered inside the bash terminal of the container. Let's quickly download a pretrained model for our Transfer Learning Training.

To see a **list of available models** play with below command.

````bash
ngc registry model list nvidia/iva/tlt*
````

Now that we have a list of models let's quickly **download pretrained model**.

````bash
ngc registry model download-version nvidia/iva/tlt_resnet18_detectnet_v2:1 \
-d /workspace/pretrained_model
````

Well if you have reached to this step congrats!, it means you are able to run NV_TLT Container  so far.

Let's **download our Dataset**.

````bash
cd /workspace/dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
chmod 777 VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
rm -rf VOCtrainval_11-May-2012.tar
````

No that we have downloaded our dataset it is required to convert them to kitti format, for doing the same we need a script. So let's quickly create ``/workspace/scripts/voc_to_kitti.py`` [source code here. ](workspace/scripts/voc_to_kitti.py)

After creating the script let's quickly do our VOC to kitti conversion.

````bash
python /workspace/scripts/voc_to_kitty.py -i /workspace/dataset/VOCdevkit/VOC2012
````

This will create additional directories inside ``/workspace/dataset/VOCdevkit/VOC2012`` where our kitti format dataset is stored now.

Well if everything is fine till now then we can move on to generate ``tf-records`` for training our model, but before that we need to create a spec_file for the conversion.

````bash
#/workspace/spec_files/det_tfrecords_pascal_voc_trainval.txt
kitti_config {
  root_directory_path: "/workspace/dataset/VOCdevkit/VOC2012"
  image_dir_name: "JPEGImages_kitti/trainval"
  label_dir_name: "Annotations_kitti/trainval"
  image_extension: ".jpg"
  partition_mode: "random"
  num_partitions:2
  val_split: 20
  num_shards: 10
}
image_directory_path: "/workspace/dataset/VOCdevkit/VOC2012"
````

We are good to go with out ``kitti to tf-records`` conversion. Execute below commands to start the **conversion** process.

````bash
tlt-dataset-convert \
    -d /workspace/spec_files/det_tfrecords_pascal_voc_trainval.txt \
    -o /workspace/tf_records/
````

you will notice several new files will be generated inside ``/workspace/tf_records`` directory.

Now we are just one step behind training the model, we need to create a spec_file for training as well. So quickly create ``/workspace/spec_files/det_train_resnet18_pascal_voc.txt`` [source code here.](workspace/spec_files/det_train_resnet18_pascal_voc.txt)

Finally, to **Begin Training** Run below command (replace with your own *API_KEY*)

````bash
API_KEY="------your-api-key-here------"
````

````bash
tlt-train detection -e /workspace/spec_files/det_train_resnet18_pascal_voc.txt \
                     -r /workspace/trained_model \
                     -k $API_KEY \
                     -n resnet18_detector
````
