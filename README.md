#Tensorbox with Inceptionv_3 Model
## Description
Implementing tensorbox using google inceptionv3 mode. Original [Tensorbox](https://github.com/Russell91/TensorBox) Thanks @[bernardopires](https://github.com/Russell91/TensorBox/issues/18)

Basically using overhaeat algorithm with inceptionv_3 model. 
> The inceptionv3 pb file is stored under **/model/** foler.


##----- using codes for street number for test ---------- 
> * git clone https://github.com/s9hangliy/tensorbox_inception3.git  _//Download related data_ 
> * ./download_data.sh 
> * cd utils && make && cd .. **//if you got error here with Cython, try sudo pip install --upgrade cython* *** 
> * python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output

> Using tensorboard for Visualization of the training tensorboard --logdir output (ps aux | grep tensorboard)

##------- Using codes for your own trainig ------------

* Prepare your dataset ./images #containing all the images ./labels #containig the txt files containing labels for corresponding image

* Generate json file using jason_yolo_test.py
* Make sure inceptionv3 model exist
* python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output

##------- Strip the grah for andorid usage ------------
* Get Tensorflow source'
* bazel build tensorflow/python/tools:strip_unused
* bazel-bin/tensorflow/python/tools/strip_unused \ --input_graph=/home/shangliy/models/inception3_10_13_1050.pb \ --output_graph=/home/shangliy/models/inception3_stripped_graph.pb \ --input_node_names=sub --output_node_names=output
