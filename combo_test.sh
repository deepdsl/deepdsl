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
        start=`date +%s.%N`
        mvn -Plinux64 exec:java -Dexec.mainClass="$2.$network"
        end=`date +%s.%N`
        runtime=$((end-start))
        echo "$network execution time: $runtime" >> deepdsl-java/$result_file
        
        #to use both $network and $1, need 1. use double quate and put backslash in front of $1 to allow
        #variable interpolation
        #kill $(jobs -p)
        ps -ef | grep nvidia-smi | awk '{print $2}' | xargs kill
        sort -n $network-$test_num.txt | sed -n '$p' | awk "{print \"$network-$test_num: \" \$1 \"mb\"}"  >> $result_file
        sleep 3
        done
}

clean() {
       for network in Alexnet128 Alexnet256 Googlenet128 Googlenet256 Overfeat128 Overfeat256 Vgg64 Resnet32 Resnet64
       do
       	rm deepdsl-java/src/main/java/deepdsl/gen/$network.java 
       	rm deepdsl-java/src/main/java/deepdsl/gen/$network\_infer.java 
       	rm deepdsl-java/$network-test1.txt
       	rm deepdsl-java/$network-test2.txt
       	rm deepdsl-java/$network-test3.txt
       done
}
echo "Start testing"
echo "Start test time $(date)" >> deepdsl-java/$result_file
#assuming start from deepdsl root folder
#Step 1: generate the code and compile
start=`date +%s.%N`
mvn -Plinux64 clean install -DskipTests 2>&1 >/dev/null
end=`date +%s.%N`
runtime=$((end-start))
echo "maven build time: $runtime" >> deepdsl-java/$result_file
#cd deepdsl-java
#mvn -Plinux64 test -Dtest=TestNetwork#testAll
start=`date +%s.%N`
mvn -Plinux64 test -Dtest=TestNetwork 2>&1 >/dev/null
end=`date +%s.%N`
runtime=$((end-start))
echo "codegen time: $runtime" >> deepdsl-java/$result_file
#cd ..
start=`date +%s.%N`
mvn -Plinux64 clean install -DskipTests 2>&1 >/dev/null
end=`date +%s.%N`
runtime=$((end-start))
echo "maven re-build(after codegen) time: $runtime" >> deepdsl-java/$result_file
cd deepdsl-java

#Step 2: Run test suite 1
echo "-------------------------------------------------------"
echo "test set 1: with cache enabled" >> $result_file
do_test $freq $package test1
sleep 10

#Step 3: Comment out one cache line and run test suite 2
echo "-------------------------------------------------------"
echo "test set 2: with only enableMemoryCache disabled" >> $result_file
sed -i -- 's/JCudaTensor.enableMemoryCache();/\/\/JCudaTensor.enableMemoryCache();/g' src/main/java/deepdsl/gen/*java
cd ..
start=`date +%s.%N`
mvn -Plinux64 clean install -DskipTests 2>&1 >/dev/null
end=`date +%s.%N`
runtime=$((end-start))
echo "maven re-build(after enableMemoryCache is disabled) time: $runtime" >> deepdsl-java/$result_file
cd deepdsl-java
do_test $freq $package test2
sleep 10

#Step 4: Comment out both cache lines and run test suite 3
echo "-------------------------------------------------------"
echo "test set 3: with both enableMemoryCache and enableWorkspaceCache disabled" >> $result_file
sed -i -- 's/JCudaTensor.enableWorkspaceCache();/\/\/JCudaTensor.enableWorkspaceCache();/g' src/main/java/deepdsl/gen/*java
cd ..
start=`date +%s.%N`
mvn -Plinux64 clean install -DskipTests 2>&1 >/dev/null
end=`date +%s.%N`
runtime=$((end-start))
echo "maven re-build(after both enableMemoryCache and enableWorkspaceCache are disabled) time: $runtime" >> deepdsl-java/$result_file
cd deepdsl-java
do_test $freq $package test3

#restore cache
start=`date +%s.%N`
sed -i -- 's,//JCudaTensor,JCudaTensor,g' src/main/java/deepdsl/gen/*java
end=`date +%s.%N`
runtime=$((end-start))
echo "code restore(re-enabled both enableMemoryCache and enableWorkspaceCache)  time: $runtime" >> deepdsl-java/$result_file

cd ..
echo "Finish testing"
echo "End test time $(date)" >> deepdsl-java/$result_file
clean
