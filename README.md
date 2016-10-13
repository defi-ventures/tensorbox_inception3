# tensorbox_inception3

Implementing tensorbox using google inceptionv3 mode.
Original Code: https://github.com/Russell91/TensorBox
Thanks @bernardopires  https://github.com/Russell91/TensorBox/issues/18

//Details come later

----- using codes for street number for test ----------
git clone https://github.com/shangliy/tensorbox_inception3.git
//Download related data
./download_data.sh
cd utils && make && cd ..
//if you got error here with Cython, try sudo pip install --upgrade cython
python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output

// Using tensorboard for Visualization of the training
 tensorboard --logdir output



 ------- Using codes for your own trainig ------------
 1. Prepare your dataset
 ./images #containing all the images
 ./labels #containig the txt files containing labels for corresponding image

 2. Generate jason file
