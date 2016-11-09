# DeepDSL

DeepDSL is a domain specific language embedded in Scala for writing deep convolutional neural network applications.
- DeepDSL program compiles into plain Java source program
- The compiled Java source program uses JCuda to run on Nvidia GPU

## Run DeepDSL compiled programs
- There are several compiled Java source program located at [src/main/java/deepdsl/gen/]. 
- These programs train several well-known deep networks: Lenet, Alexnet, Overfeat, Googlenet, Vgg, and ResNet. 

### Maven
Currently the Maven configuration only supports the native libraries for Linux; support for Windows are coming up soon ...

### Adjust learning parameters
- At the start of each file, there are some parameters you can adjust such as learn_rate and moment, as well as training iterations and test iterations. 
- The batch size for Lenet is set at 500; for Alexnet, Overfeat, and Googlenet is 128; for Vgg and ResNet is set at 64.  
- At this time, if you want to change batch size, you may want to regenerate the Java source file. Directly editing the Java source might easily miss a few places.

### Default location for trained parameters
Each program will save trained parameters (as serialized Java objects) into a default directory. 
- It will try to load saved parameters (if exist) from the same directory as well. 
- For example, [Lenet.java] will try to use the directory "[src/main/java/deepdsl/gen/]lenet" and [Alexnet.java] will try to use the directory "[src/main/java/deepdsl/gen/]alexnet" 
- You can customize this in the source file directly.

### Default location for training and testing data
Each program assumes a location for the training and test data. 
- [Lenet.java] uses Mnist, which is assumed to be located at [dataset/mnist]
- Programs such as [Alexnet.java] uses imagenet (as Lmdb database), which is assumed to be located at "[dataset/imagenet224/]ilsvrc12_train_lmdb" for training data and "[dataset/imagenet224/]ilsvrc12_val_lmdb" for testing data, where the image sizes are cropped to 224 x 224. Other image sizes should work since we would randomly cropped it to the right size. (But testing images will also be randomly cropped -- this might be fixed later).
- The training and testing all use the same batch size. 

## Generate Java source
You can generate Java source for a particular network by running a Scala test program [NetworkTest.scala]. While this is a Scala program, you can run it as a JUnit test to generate Java source code, which will be written to [src/main/java/deepdsl/gen/].

### Example: generate Lenet

```scala
    val K = 10 // # of classes 
    val N = 500; val C = 1; val N1 = 28; val N2 = 28 // batch size, channel, and x/y size
 
    // Specifying train dataSet. (code gen will also use this to find test dataSet)
    val y = Vec._new(Mnist, "label", "Y", N)              
    val x = Vec._new(Mnist, "image", "X", N, C, N1, N2)  
       
    // followings are tensor functions
    val cv1 = CudaLayer.convolv("cv1", 5, 20)       // convolution layer with kernel 5, stride 1, padding 0, and output channel 20
    val cv2 = CudaLayer.convolv("cv2", 5, 50)
    val mp = CudaLayer.max_pool(2)                  // max pooling with kernel 2 and stride 2
    val flat = Layer.flatten(4, 1)                  // flatten a 4-D tensor to 2-D: axis 0 - 3 becomes axis 0 and  axis 1-3
    val f = Layer.full("fc1", 500)                  // fully connected layer with output dimension 500
    val f2 = Layer.full("fc2", K)                   
    val softmax = CudaLayer.softmax                 
    val relu = CudaLayer.relu(2)                    // ReLU activation function (2-D)
      
    // o is a left-associative function composition operator: f o g o h == (f o g) o h  
    val network = f2 o relu o f o flat o mp o cv2 o mp o cv1 

    println(typeof(network))                        // typecheck the network and print out the tensor function type
    
    val x1 = x.asCuda                               // load x (images) to GPU memory
    val y1 = y.asIndicator(K).asCuda                // convert y (labels) to indicator vectors and load into GPU memory
    val c = (Layer.log_loss(y1) o softmax o network) (x1) // represent the log-loss of the training data
    val p = (Layer.precision(y1) o network) (x1)    // represent the accuracy of the test data
   
    val param = c.freeVar.toList                    // discover the list of training parameters
    
    // parameters: name, training iterations, test iterations, learn rate, momentum, weight decay, cropping (0 means none)
    val solver = Train("lenet", 100, 10, 0.01f, 0.9f, 0.0005f, 0)
    
    val loop = Loop(c, p, (x, y), param, solver)    // represent the training and testing loop
 
    runtimeMemory(loop.train)                       // print out the detailed memory consumption for one training loop
    parameterMemory(loop)                           // print out the parameter memory use
    workspaceMemory(loop.train)                     // print out the GPU (convolution) workspace use (only if you has Nvidia GPU)
    cudnn_gen.print(loop)                           // generate Java source code
```

[NetworkTest.scala]: <https://github.com/deepdsl/deepdsl/blob/master/deepdsl-java/src/test/java/NetworkTest.scala>

[src/main/java/deepdsl/gen/]: <https://github.com/deepdsl/deepdsl/tree/master/deepdsl-java/src/main/java/deepdsl/gen>

[Lenet.java]: <https://github.com/deepdsl/deepdsl/tree/master/deepdsl-java/src/main/java/deepdsl/gen/Lenet.java>
[Alexnet.java]: <https://github.com/deepdsl/deepdsl/tree/master/deepdsl-java/src/main/java/deepdsl/gen/Alexnet.java>

[dataset/mnist]: <https://github.com/deepdsl/deepdsl/tree/master/dataset/mnist>
[dataset/imagenet224/]: <https://github.com/deepdsl/deepdsl/tree/master/dataset/imagenet224/>
