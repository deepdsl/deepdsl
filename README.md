# deepdsl

DeepDSL is a domain specific language embedded in Scala for writing deep convolutional neural network applications.
- DeepDSL program compiles into plain Java source program
- The compiled Java source program uses JCuda to run on Nvidia GPU

## Run DeepDSL compiled programs
- There are several compiled Java source program located at "deepdsl-java/src/main/java/deepdsl/gen". 
- These programs train several well-known deep networks: Lenet, Alexnet, Overfeat, Googlenet, Vgg, and ResNet. 

### Adjust learning parameters
- At the start of each file, there are some parameters you can adjust such as learn_rate and moment, as well as training iterations and test iterations. 
- The batch size for Lenet is set at 500; for Alexnet, Overfeat, and Googlenet is 128; for Vgg and ResNet is set at 64.  
- At this time, if you want to change batch size, you may want to regenerate the Java source file. Directly editing the Java source might easily miss a few places.

### Default location for trained parameters
Each program will save trained parameters (as serialized Java objects) into a default directory. 
- It will try to load saved parameters (if exist) from the same directory as well. 
- For example, Lenet.java will try to use the directory "src/main/java/deepdsl/gen/lenet" and Alexnet.java will try to use the directory "src/main/java/deepdsl/gen/alexnet" 
- You can customize this in the source file directly.

### Default location for training and testing data
Each program assumes a location for the training and test data. 
- Lenet.java uses Mnist, which is assumed to be located at "dataset/mnist"
- Programs such as Alexnet.java uses imagenet (as Lmdb database) located at "database/imagenet224/ilsvrc12_train_lmdb" for training data and "dataset/imagenet224/ilsvrc12_val_lmdb" for testing data, where the image sizes are cropped to 224 x 224. Other image sizes should work since we would randomly cropped it to the right size. (But testing images will also be randomly cropped -- this might be fixed later).
- The training and testing all use the same batch size. 

## Generate Java source
You can generate Java source for a particular network by running a Scala test program located at "deepdsl-java/src/test/java/NetworkTest.scala". While this is a Scala program, you can run it as a JUnit test to generate Java source code, which will be written to "src/main/java/deepdsl/gen/".
