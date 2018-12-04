# Novelty Detection for VIRAT with ALOCC using Keras


This repository is a fork of the Keras implementation of paper https://arxiv.org/abs/1802.09088

The author provides a tutorial to justify and teach how to use the Adversarially Learned One-Class Classifier (ALOCC) model:

* [Tutorial part 1](https://www.dlology.com/blog/how-to-do-novelty-detection-in-keras-with-generative-adversarial-network/)
* [Tutorial part 2](https://www.dlology.com/blog/how-to-do-novelty-detection-in-keras-with-generative-adversarial-network-part-2/)



## Objective

The main objective of this work is to adapt ALOCC to be used with the [VIRAT dataset](http://www.viratdata.org/). In more detail, I am exploring the idea of making it work to detect if a frame belongs to a camera or not from a GAN perspective.



### Approach

First, I sampled frames from each of the 12 camera views found in the [VIRAT dataset](http://www.viratdata.org/). Then the model was trained per view, taking advantage of the unsupervised learning that the model provides. Finally, the adversarial model is fine-tuned with scene samples as 1's and out-of-scene samples as 0's.