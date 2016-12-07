package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

// This file is for inference only, which needs trained parameters.
public class Overfeat_infer {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/overfeat";
// test_data_path
static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// test_size
static int test_size = 10000;

// (Convolv(1,1),List(List(128, 1024, 13, 13), List(1024, 1024, 3, 3), List(1024)))
static JCudnnConvolution x65 = new JCudnnConvolution(new int[]{128,1024,13,13},new int[]{1024,1024,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,1),List(List(128, 256, 13, 13), List(512, 256, 3, 3), List(512)))
static JCudnnConvolution x43 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{512,256,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(128, 512, 13, 13), List(1024, 512, 3, 3), List(1024)))
static JCudnnConvolution x54 = new JCudnnConvolution(new int[]{128,512,13,13},new int[]{1024,512,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,2),List(List(128, 96, 27, 27), List(256, 96, 5, 5), List(256)))
static JCudnnConvolution x28 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
// (Convolv(4,0),List(List(128, 3, 224, 224), List(96, 3, 11, 11), List(96)))
static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 0);
// (Imagenet,false)
static ImagenetFactory x1 = ImagenetFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
// (Pooling(2,2,0,true),List(List(128, 1024, 13, 13)))
static JCudnnPooling x71 = new JCudnnPooling(new int[]{128,1024,13,13}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(128, 256, 27, 27)))
static JCudnnPooling x35 = new JCudnnPooling(new int[]{128,256,27,27}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(128, 96, 54, 54)))
static JCudnnPooling x20 = new JCudnnPooling(new int[]{128,96,54,54}, 2, 2, 0, PoolingType.MAX);
// (ReLU(),List(List(128, 1024, 13, 13)))
static JCudnnActivation x58 = new JCudnnActivation(new int[]{128,1024,13,13}, ActivationMode.RELU);
// (ReLU(),List(List(128, 256, 27, 27)))
static JCudnnActivation x32 = new JCudnnActivation(new int[]{128,256,27,27}, ActivationMode.RELU);
// (ReLU(),List(List(128, 512, 13, 13)))
static JCudnnActivation x47 = new JCudnnActivation(new int[]{128,512,13,13}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 54, 54)))
static JCudnnActivation x17 = new JCudnnActivation(new int[]{128,96,54,54}, ActivationMode.RELU);
// X
static JTensorFloat x2;
// cv1_B
static JCudaTensor x12 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
// cv1_W
static JCudaTensor x11 = JTensor.randomFloat(-0.07422696f, 0.07422696f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
// cv2_B
static JCudaTensor x27 = JTensor.constFloat(0.2f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x26 = JTensor.randomFloat(-0.028867513f, 0.028867513f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
// cv3_B
static JCudaTensor x42 = JTensor.constFloat(0.2f, 512).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x41 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv4_B
static JCudaTensor x53 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv4_B").asJCudaTensor();
// cv4_W
static JCudaTensor x52 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 1024, 512, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
// cv5_B
static JCudaTensor x64 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv5_B").asJCudaTensor();
// cv5_W
static JCudaTensor x63 = JTensor.randomFloat(-0.014731391f, 0.014731391f, 1024, 1024, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
// fc6_B
static JCudaTensor x84 = JTensor.constFloat(0.0f, 3072).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x79 = JTensor.randomFloat(-0.0073656957f, 0.0073656957f, 3072, 36864).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x95 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x90 = JTensor.randomFloat(-0.02551552f, 0.02551552f, 4096, 3072).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x106 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x101 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
test();
x53.free();
x64.free();
x84.free();
x106.free();
x63.free();
x27.free();
x26.free();
x90.free();
x12.free();
x41.free();
x52.free();
x79.free();
x101.free();
x95.free();
x42.free();
x11.free();
x35.free();
x58.free();
x71.free();
x54.free();
x65.free();
x32.free();
x20.free();
x17.free();
x13.free();
x28.free();
x43.free();
x47.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void test() {
 for(int x3=0; x3<test_itr; x3++) {
JTensorFloatTuple x4 =  x1.nextFloat();
x2 = x4.image;

// val X464 = Cuda(X)
JCudaTensor x5;
JTensorFloat x6;
x6 = x2;
x5 = x6.asJCudaTensor();

// val X465 = Convolv(4,0)(X464,cv1_W,cv1_B)
JCudaTensor x7;
JCudaTensor x8, x9, x10;
x8 = x5;
x9 = x11;
x10 = x12;
x7 = x13.forward(x8, x9, x10);

// Dealloc(X464)
JCudaTensor x14;
x14 = x5;
x14.free();

// val X466 = ReLU()(X465)
JCudaTensor x15;
JCudaTensor x16;
x16 = x7;
x15 = x17.forward(x16);

// val X467 = Pooling(2,2,0,true)(X466)
JCudaTensor x18;
JCudaTensor x19;
x19 = x15;
x18 = x20.forward(x19);

// Dealloc(X466)
JCudaTensor x21;
x21 = x15;
x21.free();

// val X468 = Convolv(1,2)(X467,cv2_W,cv2_B)
JCudaTensor x22;
JCudaTensor x23, x24, x25;
x23 = x18;
x24 = x26;
x25 = x27;
x22 = x28.forward(x23, x24, x25);

// Dealloc(X467)
JCudaTensor x29;
x29 = x18;
x29.free();

// val X469 = ReLU()(X468)
JCudaTensor x30;
JCudaTensor x31;
x31 = x22;
x30 = x32.forward(x31);

// val X470 = Pooling(2,2,0,true)(X469)
JCudaTensor x33;
JCudaTensor x34;
x34 = x30;
x33 = x35.forward(x34);

// Dealloc(X469)
JCudaTensor x36;
x36 = x30;
x36.free();

// val X471 = Convolv(1,1)(X470,cv3_W,cv3_B)
JCudaTensor x37;
JCudaTensor x38, x39, x40;
x38 = x33;
x39 = x41;
x40 = x42;
x37 = x43.forward(x38, x39, x40);

// Dealloc(X470)
JCudaTensor x44;
x44 = x33;
x44.free();

// val X472 = ReLU()(X471)
JCudaTensor x45;
JCudaTensor x46;
x46 = x37;
x45 = x47.forward(x46);

// val X473 = Convolv(1,1)(X472,cv4_W,cv4_B)
JCudaTensor x48;
JCudaTensor x49, x50, x51;
x49 = x45;
x50 = x52;
x51 = x53;
x48 = x54.forward(x49, x50, x51);

// Dealloc(X472)
JCudaTensor x55;
x55 = x45;
x55.free();

// val X474 = ReLU()(X473)
JCudaTensor x56;
JCudaTensor x57;
x57 = x48;
x56 = x58.forward(x57);

// val X475 = Convolv(1,1)(X474,cv5_W,cv5_B)
JCudaTensor x59;
JCudaTensor x60, x61, x62;
x60 = x56;
x61 = x63;
x62 = x64;
x59 = x65.forward(x60, x61, x62);

// Dealloc(X474)
JCudaTensor x66;
x66 = x56;
x66.free();

// val X476 = ReLU()(X475)
JCudaTensor x67;
JCudaTensor x68;
x68 = x59;
x67 = x58.forward(x68);

// val X477 = Pooling(2,2,0,true)(X476)
JCudaTensor x69;
JCudaTensor x70;
x70 = x67;
x69 = x71.forward(x70);

// Dealloc(X476)
JCudaTensor x72;
x72 = x67;
x72.free();

// val X478 = (X477[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor x73;
JCudaMatrix x74;
JCudaMatrix x75;
JCudaTensor x76;
JCudaTensor x77;
x77 = x69;
x76 = x77.flatten(1, new int[]{1024, 6, 6});
x74 = x76.asMatrix(1, true);
JCudaTensor x78;
x78 = x79;
x75 = x78.asMatrix(1, true);
x73 = x74.times(x75);

// Dealloc(X477)
JCudaTensor x80;
x80 = x69;
x80.free();

// val X480 = (X478 + (i1) => fc6_B)
JCudaTensor x81;
JCudaTensor x82, x83;
x82 = x73;
x83 = x84;
x81 = x83.copy(128, x82);

// val X481 = (X480)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor x85;
JCudaMatrix x86;
JCudaMatrix x87;
JCudaTensor x88;
x88 = x81;
x86 = x88.asMatrix(1, true);
JCudaTensor x89;
x89 = x90;
x87 = x89.asMatrix(1, true);
x85 = x86.times(x87);

// Dealloc(X480)
JCudaTensor x91;
x91 = x81;
x91.free();

// val X483 = (X481 + (i4) => fc7_B)
JCudaTensor x92;
JCudaTensor x93, x94;
x93 = x85;
x94 = x95;
x92 = x94.copy(128, x93);

// val X484 = (X483)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor x96;
JCudaMatrix x97;
JCudaMatrix x98;
JCudaTensor x99;
x99 = x92;
x97 = x99.asMatrix(1, true);
JCudaTensor x100;
x100 = x101;
x98 = x100.asMatrix(1, true);
x96 = x97.times(x98);

// Dealloc(X483)
JCudaTensor x102;
x102 = x92;
x102.free();

// val X486 = (X484 + (i7) => fc8_B)
JCudaTensor x103;
JCudaTensor x104, x105;
x104 = x96;
x105 = x106;
x103 = x105.copy(128, x104);

// Prediction(X486)
JCudaTensor x107;
x107 = x103;
System.out.println(x3 + " inference " + java.util.Arrays.toString(x107.prediction()));

// Dealloc(X486)
JCudaTensor x108;
x108 = x103;
x108.free();

}
 
}

}