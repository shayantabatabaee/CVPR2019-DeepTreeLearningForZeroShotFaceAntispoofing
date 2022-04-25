# Deep Tree Learning for Zero-shot Face Anti-Spoofing
## Changes
Some features were added compared to original repository.
<br>
#### Setup
For installing dependencies please run: 
<pre><code>pip install -r requirements.txt</code></pre>
#### Dataset
To start training you should just modified `__load_files` method in `dataset.py` file.
For better memory optimization and performance of training, all data loads into memory and feeds to model using a generator.
<br>
There is no need to have depth map since i assume dataset consist of replay attack, photo attack and live.
Note that the spoof label is `1` and live label is `0`.
#### Evaluate
To evaluate model run:
<pre><code>.evaluate(validation_dataset)</code></pre>
#### Save
There are two methods for saving model, use `save` to save graph with weights. Also you can use `save_weight` to only save weights of model without graph
#### Load
Both `load` and `load_weight` can be used for loading model
##
Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu

![alt text](https://yaojieliu.github.io/images/cvpr19.png)

## Setup
Install the Tensorflow 2.0.

## Training
To run the training code:
python train.py


## Acknowledge
Please cite the paper:

    @inproceedings{cvpr19yaojie,
        title={Deep Tree Learning for Zero-shot Face Anti-Spoofing},
        author={Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu},
        booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2019)},
        address={Long Beach, CA},
        year={2019}
    }
    
If you have any question, please contact: [Yaojie Liu](liuyaoj1@msu.edu) 
   
