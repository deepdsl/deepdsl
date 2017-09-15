#!/bin/bash 

rm deepdsl-java/*txt
rm deepdsl-java/*log
rm deepdsl-java/src/main/java/deepdsl/gen/*_infer.java
git checkout deepdsl-java/src/main/java/deepdsl/gen/*java
git checkout deepdsl-java/src/main/java/deepdsl/gen/*_infer.java
git checkout deepdsl-java/src/main/java/deepdsl/gen/Lenet.java
git checkout deepdsl-java/src/main/java/deepdsl/gen/Lenet_tanh.java
rm deepdsl-java/src/main/java/deepdsl/gen/Alexnet128.java
rm deepdsl-java/src/main/java/deepdsl/gen/Alexnet256.java
rm deepdsl-java/src/main/java/deepdsl/gen/Googlenet128.java
rm deepdsl-java/src/main/java/deepdsl/gen/Googlenet256.java
rm deepdsl-java/src/main/java/deepdsl/gen/Overfeat128.java
rm deepdsl-java/src/main/java/deepdsl/gen/Overfeat256.java
rm deepdsl-java/src/main/java/deepdsl/gen/Resnet32.java
rm deepdsl-java/src/main/java/deepdsl/gen/Resnet64.java
rm deepdsl-java/src/main/java/deepdsl/gen/Vgg64.java
