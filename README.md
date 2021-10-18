# Idea
Using autoencoders to force ANNs to generalize.

Artificial Neural Networks are tremendously good at learning correct classifications from large datasets.
But my suspicion is that this is largely due to blunt memorization of the dataset, which becomes apparent if a network
is trained on the same dataset for too long and starts overfitting. My intuition is that ANNs basically start overfitting right
from the beginning - it just so happens that if the images of the test set are taken from the same distribution as the
training set, the separator function learned by the network is <i>correct enough</i> (i.e. images from the same class are sufficiently
similar to each other, at least in high-D space, so that you get the correct output classification.)
Expressed differently: A network learns images by heart, it "recognizes" images that it has seen before and remembers their 
classification, and its vision is blurry enough that it also recognizes images that it hasn't actually seen before.

This is not what we want, even if it seems to be working well.

What we actually want is for networks to form meaningful abstractions of seen images, extract properties from the training images
and learn class assignments based on these abstractions. <b>We don't want the inner layers of ANNs to represent a well-compressed 
version of the example that is being fed through the network. We want <i>lossy</i> compression that throws out all information that's not relevant
for the classification in the last layer. </b>

We now need to find architectural choices that enforce this behaviour: Correctly classifying the images while throwing out irrelevant information.
One example of such a choice is "fewer neurons in a later layer than in an early one". Another is CNNs, which reuse weights (meaning that 
the weights of the filter have to work well everywhere instead of "only on this image patch", which enforces generalization).

<b>My proposition is to introduce a new loss signal into the training process that tries to measure and penalize the tendency of the network to
memorize what it sees.</b> To this end, an autoencoder is set up and trained to recreate the input image from the activations of a late network layer
which I will call the "representation-layer" here.

todo: image of a CNN with representation-layer and attached autoencoder.

Every few training steps, the autoencoder is trained using backprop, but only the neurons after the representation-layer are changed. So it learns to
recreate the original image as well as possible from only the activations of the representation layer.
The loss of the autoencoder is integrated into the loss function of the CNN, but with a negative sign: For the classifier, the classification needs to
be good, while the recreation quality needs to be as bad as possible.
This means that the network is forced to learn an intermediate representation of the image that still contains all the information necessary for a correct
classification, while all other information is discarded. In particular, if this works, the representation layer does not contain a mere compressed version
of the input image (because from that, a perfect recreation would be possible).


# Setup 
To activate the environment: source ./env/bin/activate
