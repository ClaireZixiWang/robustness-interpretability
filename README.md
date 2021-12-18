# Neural Network Robustness vs Interpretability
## Project Description
### Problem Motivation
Interpretable Machine Learning and Robust Machine Learning are two important fields of studies that aims to make Machine Learning models more reliable and trustworthy, especially in complex high-stake applications such as medicine, automatic vehicle, finance, law, etc. 

While our Machine Learning models have reached very high accuracy, it is still brittle under carefully-constructed adversarial attacks. With some carefully constructed noise added on top, an image that looks almost identical to us human eyes, will suddenly be classified as another picture by a Neural Network, even with very high confidence. This issue has boomed an era of study in Robust Machine Learning, including [Robust Optimization](https://arxiv.org/pdf/1706.06083.pdf), [Robust Regularizartion](https://arxiv.org/abs/1807.10272),  [Online Defense](https://arxiv.org/abs/2103.14222), all aiming to improve the models' robustness under these adversarial examples. You can read more about this topic in our [Midterm Seminar Blog](https://medium.com/@zixi-wang/comparing-adversarial-attacks-and-defenses-vision-vs-speech-5e2e0f9f753b).

![image](https://user-images.githubusercontent.com/59561588/146614009-dc0914f5-b52a-4e82-9067-8d0c89902ec1.png)

The Neural Network's vulnerability under adversarial attacks brings up another problem - in fact, we don't know what our models are doing when they are making decisions. We don't know why adversaries exists and what our models are learning and seeing when facing these adversaries. This brings up the other topic of our project - Interpretable Machine Learning. We would like to know what the models are doing for the sake of debugging, explaining to stakeholders, and taking accountabilities. Moreover, we just want to make sure our model is reliable, right?

We look at these two intrinsincally intertwined sub-fields of machine learning, and we want to answer the following questions: if robust models are "immune" to the noise in the pictures, then are they learning the "semantically meaningful" features that we see as human eyes? If so, are robust model more interpretable than naturally trained models?
### Approach and Experiment Design
In our project, we use saliency maps to interprete Neural Networks, and compare the saliency maps of natural images and adversarial images under naturally trained and adversarially trained models. We aim to investigate whether Robust Models gives a cleaner saliency map, and whether that means a more interpretable model.
#### Interpreting Neural Network
![image](https://user-images.githubusercontent.com/59561588/146616863-67730e9e-0d61-45e7-8139-975587fdd6e9.png)

Saliency map is a commonly used technique in Interpretable ML. It computes and shows the gradient of the output category with respect to the input images. This gives us an indicator as to how classification changes with respect to small changes in each input image pixels. Essentially it shows us which pixels are used for the model to make a decision. 

We slightly adated the original Saliency Map method in our projects - we used the guided-backpropogation to get the gradients of the input image. Essentially we only look at the absolute values of the gradients, and disregard any differences between positive gradients and negative gradients. This is mainly because in our project, we only want to investigate *which features* are useful in predictions, and the direction of those features' contribustion is less relative.

Moreover, we quantify the "cleaness" of a feature map by computing its inner product with the original picture. This is because in MNIST dataset, both the original images and the saliency maps are gray-scale pictures with black background and white "strokes". So the more a feature map is similar to the original image through inner product, the more we consider it as a "clean" interpretation. Or in other word, the more "interpretable" that model should be.
#### Robust Models
We adopted a straight forward Robust Optimization method to train a Robust Model. Specifically, we used images resulting from 7-step PGD attack to adversarially train a Robust Model. We use this Adversarially triained Robust Model to do the following experiments.
#### Experiment Set Up
Our Implementation details are as follows:
* Model: LeNet with 50 epochs
* Robust model: 7-step PGD with 50 epochs
* Framework: PyTorch
* Dataset: MNIST
* Adversarial Attack: FGSM Attack
* Platform: Google Colab with GPU
* Libraries: 
  * [Adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) for adversarial training
  * [Pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) for guided backpropagation

Our experiment design flow is as follows:
* Train the model and evaluate on test images, observe the saliency maps
* Generate adversarial examples and see how the saliency maps change
* Adversarially train a robust model
* Test the robust model on original and adversarial images
* Compare the four saliency maps and calculate the similarities with input images


## Repo Description
The main code of our project is the following two files: `Robustness_Interpretability_MNIST_Final.ipynb` and `guided_backprop.py`. `guided_backprop.py` is the helper code that we borrowed from [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations) to compute guided backpropagation gradients. `Robustness_Interpretability_MNIST_Final.ipynb` is the main experimentation notebook, where we train two models (naturally and adversarially), generate adversarial images for testing, and compare the saliency maps for different images under different models.

`results` stores the resulting picture maps that we plot during the experiment.

`lenet-non-val.pth` and `lenet-pgd-trained.pth` are the pretrianed Natural LeNet and Adversarially Trained LeNet that we included here in the repo. But the training themselves actually will not take too much time. This inclusion is mostly for the sake of completeness.

`Resnet18_naturally_train.ipynb` and `Robustness_interpretability_cifar10_resent18.ipynb` are notebooks for a next-step experiment on CIFAR10 and ResNet18. It is still very much a next step and is beyond the scope of this project per se. However, we just want to include them here for the convinence of future work.

## Running the demo
## Results
### Naturally Trained Model
![image](https://user-images.githubusercontent.com/59561588/146620884-b86df0bf-c82e-4b4b-ad1e-59cc6bb83100.png)
![image](https://user-images.githubusercontent.com/59561588/146621757-0bee7b2f-4d14-4fd4-bba8-7bb43de1ffb8.png)


### Adversarially Trained Model
![image](https://user-images.githubusercontent.com/59561588/146620871-c7ba5057-ee5b-4690-91f3-c62ab27086a0.png)
![image](https://user-images.githubusercontent.com/59561588/146621764-168d6a83-fabe-4d39-b88d-a04f97c68aa6.png)


We can see that the saliency maps for robust model is a lot cleaner than those for natural models. Specifically, for natural models, the saliency maps of natural images looks somewhat like the numbers in the original images, whereas those of adversarial images make almost no sense. On the contrary, for robust models, both natural and adversarial images' saliency maps looks very much like the original numbers. 

We can conclude that there is some degree of correlation between a model's robustness and interpretability. For a stronger support of this correlation, we could quantify the "interpretability" of a model by the calculating the average similarity score between original images and saliency maps among the entire MNIST dataset, and quantitatively analyze the correlation between robustness and interpretability. We could also look at bigger datasets like CIFAR10 and ImageNet and see if a similar result would still hold. For the time limit of this project, we will leave these experiments for future works. 
