# Dog-Classification
A simple dog breed classifier using tensor flow in a Jupyter notebook.

This machine learning model takes as input a photo and identifies the breed of the dog in the photo.

It was trained with teh Stanford dogs dataset, which includes 20,580 images of dogs of 120 different breeds.

For this project I used transfer learning, where you take a pretrained model and use it as a 'base' for the Neural Network(NN). This base was
trained previously by researchers on many classes of images, this means that it picks out feautures like circles and straight lines. But also
more complex features like heads and bodies of animals and people.

This means that when a 'head' is added to the neural network and it is trained it will start looking for whatever images you train it on.

I have tested models for the base of the NN, and decided to use Resnet34.

