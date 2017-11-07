#!/bin/bash

#frequency in milliseconds
freq=100
result_file=result.txt
package=deepdsl.gen

do_test() {
        test_num=$3
        for network in Alexnet128 Alexnet256 Googlenet128 Googlenet256 Overfeat128 Overfeat256 Vgg64 Resnet32 Resnet64
        do
        #test one network at a time
        nvidia-smi --query-gpu=memory.used --format=csv >> $network-$test_num.txt -lms $1 &
        { time mvn -Plinux64 exec:java -Dexec.mainClass="$2.$network" ; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "$network execution time: " >> $result_file
        
        #to use both $network and $1, need 1. use double quate and put backslash in front of $1 to allow
        #variable interpolation
        #kill $(jobs -p)
        ps -ef | grep [n]vidia-smi | awk '{print $2}' | xargs kill
        sort -n $network-$test_num.txt | sed -n '$p' | awk "{print \"$network-$test_num: \" \$1 \"mb\"}"  >> $result_file
        sleep 3
        done
}

clean() {
       for network in Alexnet128 Alexnet256 Googlenet128 Googlenet256 Overfeat128 Overfeat256 Vgg64 Resnet32 Resnet64
       do
        rm deepdsl-java/src/main/java/deepdsl/gen/$network.java
        rm -fr deepdsl-java/src/main/java/deepdsl/gen/`echo $network | tr '[:upper:]' '[:lower:]'`/
        rm deepdsl-java/$network-test1.txt
        rm deepdsl-java/$network-test2.txt
        rm deepdsl-java/$network-test3.txt
       done
       git checkout deepdsl-java/src/main/java/deepdsl/*
}
echo "Start testing"
echo "Start test time $(date)" >> deepdsl-java/$result_file
#assuming start from deepdsl root folder
#Step 1: generate the code and compile
{ time mvn -Plinux64 clean install -DskipTests; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "maven build time: " >> deepdsl-java/$result_file
#cd deepdsl-java
#mvn -Plinux64 test -Dtest=TestNetwork#testAll
{ time mvn -Plinux64 test -Dtest=TestNetwork; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "codegen time: " >> deepdsl-java/$result_file
#cd ..
{ time mvn -Plinux64 clean install -DskipTests; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "maven re-build(after codegen) time: " >> deepdsl-java/$result_file
cd deepdsl-java

#Step 2: Run test suite 1
echo "-------------------------------------------------------"
echo "test set 1: with cache enabled" >> $result_file
do_test $freq $package test1
sleep 10

#Step 3: Comment out one cache line and run test suite 2
echo "-------------------------------------------------------"
echo "test set 2: with only enableMemoryCache disabled" >> $result_file
sed -i -- 's/JCudaTensor.enableMemoryCache();/\/\/JCudaTensor.enableMemoryCache();/g' src/main/java/deepdsl/util/CudaRun.java
cd ..
{ time mvn -Plinux64 clean install -DskipTests; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "maven re-build(after enableMemoryCache is disabled) time: " >> deepdsl-java/$result_file
cd deepdsl-java
do_test $freq $package test2
sleep 10

#Step 4: Comment out both cache lines and run test suite 3
echo "-------------------------------------------------------"
echo "test set 3: with both enableMemoryCache and enableWorkspaceCache disabled" >> $result_file
sed -i -- 's/JCudaFunction.enableWorkspaceCache();/\/\/JCudaFunction.enableWorkspaceCache();/g' src/main/java/deepdsl/util/CudaRun.java
cd ..
{ time mvn -Plinux64 clean install -DskipTests; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "maven re-build(after both enableMemoryCache and enableWorkspaceCache are disabled) time: " >> deepdsl-java/$result_file
cd deepdsl-java
do_test $freq $package test3

#restore cache
{ time sed -i -- 's,//JCudaTensor,JCudaTensor,g' src/main/java/deepdsl/gen/*java; } 2>&1 | grep real | sed -En "s/real\s*(.+)/\1/p" | xargs echo "code restore(re-enabled both enableMemoryCache and enableWorkspaceCache)  time: " >> $result_file
cd ..
echo "Finish testing"
echo "End test time $(date)" >> deepdsl-java/$result_file
clean
