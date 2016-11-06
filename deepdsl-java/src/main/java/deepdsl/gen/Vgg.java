package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.tensor.*;
import deepdsl.util.*;


public class Vgg {
static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.999995f;
// lrn_rate_1
static float lrn_rate_1 = -0.01f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/vgg";
// test_data_path
static String test_data_path = "dataset/imagenet224/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 40;
// train_data_path
static String train_data_path = "dataset/imagenet224/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 2000;

// (Convolv(1,1),List(64, 128, 112, 112))
static JCudnnConvolution x53 = new JCudnnConvolution(new int[]{64,128,112,112},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 128, 112, 112))
static JCudnnConvolution x43 = new JCudnnConvolution(new int[]{64,64,112,112},new int[]{128,64,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 256, 56, 56))
static JCudnnConvolution x86 = new JCudnnConvolution(new int[]{64,256,56,56},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 56, 56))
static JCudnnConvolution x76 = new JCudnnConvolution(new int[]{64,256,56,56},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 56, 56))
static JCudnnConvolution x66 = new JCudnnConvolution(new int[]{64,128,56,56},new int[]{256,128,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 512, 14, 14))
static JCudnnConvolution x152 = new JCudnnConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 14, 14))
static JCudnnConvolution x142 = new JCudnnConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 14, 14))
static JCudnnConvolution x132 = new JCudnnConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 28, 28))
static JCudnnConvolution x119 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 28, 28))
static JCudnnConvolution x109 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 28, 28))
static JCudnnConvolution x99 = new JCudnnConvolution(new int[]{64,256,28,28},new int[]{512,256,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 64, 224, 224))
static JCudnnConvolution x30 = new JCudnnConvolution(new int[]{64,64,224,224},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(1,1),List(64, 64, 224, 224))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,3,3},new int[]{64}, 1, 1);
// (Dropout(0.5),List(64, 4096))
static JCudnnDropout x175 = new JCudnnDropout(new int[]{64,4096}, 0.5f);
// (LMDB,false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, 640, new int[]{64, 3, 224, 224});
// (LMDB,true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, 6400, new int[]{64, 3, 224, 224});
// (LogSoftmax(),List(64, 1000))
static JCudnnSoftmax x202 = new JCudnnSoftmax(new int[]{64,1000}, 2);
// (Pooling(2,2,0,true),List(64, 128, 56, 56))
static JCudnnPooling x59 = new JCudnnPooling(new int[]{64,128,112,112}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(64, 256, 28, 28))
static JCudnnPooling x92 = new JCudnnPooling(new int[]{64,256,56,56}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(64, 512, 7, 7))
static JCudnnPooling x158 = new JCudnnPooling(new int[]{64,512,14,14}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(64, 512, 14, 14))
static JCudnnPooling x125 = new JCudnnPooling(new int[]{64,512,28,28}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(64, 64, 112, 112))
static JCudnnPooling x36 = new JCudnnPooling(new int[]{64,64,224,224}, 2, 2, 0, 0);
// (ReLU(),List(64, 128, 112, 112))
static JCudnnActivation x56 = new JCudnnActivation(new int[]{64,128,112,112}, 1);
// (ReLU(),List(64, 128, 112, 112))
static JCudnnActivation x46 = new JCudnnActivation(new int[]{64,128,112,112}, 1);
// (ReLU(),List(64, 256, 56, 56))
static JCudnnActivation x89 = new JCudnnActivation(new int[]{64,256,56,56}, 1);
// (ReLU(),List(64, 256, 56, 56))
static JCudnnActivation x79 = new JCudnnActivation(new int[]{64,256,56,56}, 1);
// (ReLU(),List(64, 256, 56, 56))
static JCudnnActivation x69 = new JCudnnActivation(new int[]{64,256,56,56}, 1);
// (ReLU(),List(64, 4096))
static JCudnnActivation x172 = new JCudnnActivation(new int[]{64,4096}, 1);
// (ReLU(),List(64, 512, 14, 14))
static JCudnnActivation x155 = new JCudnnActivation(new int[]{64,512,14,14}, 1);
// (ReLU(),List(64, 512, 14, 14))
static JCudnnActivation x145 = new JCudnnActivation(new int[]{64,512,14,14}, 1);
// (ReLU(),List(64, 512, 14, 14))
static JCudnnActivation x135 = new JCudnnActivation(new int[]{64,512,14,14}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x122 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x112 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x102 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 64, 224, 224))
static JCudnnActivation x33 = new JCudnnActivation(new int[]{64,64,224,224}, 1);
// (ReLU(),List(64, 64, 224, 224))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{64,64,224,224}, 1);
// V_cv11_B
static JCudaTensor x626 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv11_W
static JCudaTensor x630 = JTensor.constFloat(0.0f, 64, 3, 3, 3).asJCudaTensor();
// V_cv12_B
static JCudaTensor x611 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv12_W
static JCudaTensor x606 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_cv21_B
static JCudaTensor x577 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv21_W
static JCudaTensor x581 = JTensor.constFloat(0.0f, 128, 64, 3, 3).asJCudaTensor();
// V_cv22_B
static JCudaTensor x559 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv22_W
static JCudaTensor x554 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_cv31_B
static JCudaTensor x530 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv31_W
static JCudaTensor x525 = JTensor.constFloat(0.0f, 256, 128, 3, 3).asJCudaTensor();
// V_cv32_B
static JCudaTensor x507 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv32_W
static JCudaTensor x502 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_cv33_B
static JCudaTensor x484 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv33_W
static JCudaTensor x479 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_cv41_B
static JCudaTensor x450 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv41_W
static JCudaTensor x454 = JTensor.constFloat(0.0f, 512, 256, 3, 3).asJCudaTensor();
// V_cv42_B
static JCudaTensor x432 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv42_W
static JCudaTensor x427 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_cv43_B
static JCudaTensor x409 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv43_W
static JCudaTensor x404 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_cv51_B
static JCudaTensor x380 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv51_W
static JCudaTensor x375 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_cv52_B
static JCudaTensor x349 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv52_W
static JCudaTensor x356 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_cv53_B
static JCudaTensor x334 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv53_W
static JCudaTensor x329 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_fc6_B
static JCudaTensor x311 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc6_W
static JCudaTensor x305 = JTensor.constFloat(0.0f, 4096, 25088).asJCudaTensor();
// V_fc7_B
static JCudaTensor x264 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc7_W
static JCudaTensor x268 = JTensor.constFloat(0.0f, 4096, 4096).asJCudaTensor();
// V_fc8_B
static JCudaTensor x230 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc8_W
static JCudaTensor x237 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// cv11_B
static JCudaTensor x16 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
// cv11_W
static JCudaTensor x15 = JTensor.randomFloat(-0.27216554f, 0.27216554f, 64, 3, 3, 3).load(network_dir + "/cv11_W").asJCudaTensor();
// cv12_B
static JCudaTensor x29 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv12_B").asJCudaTensor();
// cv12_W
static JCudaTensor x28 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/cv12_W").asJCudaTensor();
// cv21_B
static JCudaTensor x42 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv21_B").asJCudaTensor();
// cv21_W
static JCudaTensor x41 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 128, 64, 3, 3).load(network_dir + "/cv21_W").asJCudaTensor();
// cv22_B
static JCudaTensor x52 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv22_B").asJCudaTensor();
// cv22_W
static JCudaTensor x51 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/cv22_W").asJCudaTensor();
// cv31_B
static JCudaTensor x65 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv31_B").asJCudaTensor();
// cv31_W
static JCudaTensor x64 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 256, 128, 3, 3).load(network_dir + "/cv31_W").asJCudaTensor();
// cv32_B
static JCudaTensor x75 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv32_B").asJCudaTensor();
// cv32_W
static JCudaTensor x74 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv32_W").asJCudaTensor();
// cv33_B
static JCudaTensor x85 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv33_B").asJCudaTensor();
// cv33_W
static JCudaTensor x84 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
// cv41_B
static JCudaTensor x98 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv41_B").asJCudaTensor();
// cv41_W
static JCudaTensor x97 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv41_W").asJCudaTensor();
// cv42_B
static JCudaTensor x108 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv42_B").asJCudaTensor();
// cv42_W
static JCudaTensor x107 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv42_W").asJCudaTensor();
// cv43_B
static JCudaTensor x118 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv43_B").asJCudaTensor();
// cv43_W
static JCudaTensor x117 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
// cv51_B
static JCudaTensor x131 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv51_B").asJCudaTensor();
// cv51_W
static JCudaTensor x130 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv51_W").asJCudaTensor();
// cv52_B
static JCudaTensor x141 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv52_B").asJCudaTensor();
// cv52_W
static JCudaTensor x140 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv52_W").asJCudaTensor();
// cv53_B
static JCudaTensor x151 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv53_B").asJCudaTensor();
// cv53_W
static JCudaTensor x150 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
// fc6_B
static JCudaTensor x169 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x165 = JTensor.randomFloat(-0.008928572f, 0.008928572f, 4096, 25088).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x185 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x181 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x199 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x195 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
ArithStats.isStats = false;
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
System.out.println(ArithStats.outStats());
test();
x16.save(network_dir + "/cv11_B");
x15.save(network_dir + "/cv11_W");
x29.save(network_dir + "/cv12_B");
x28.save(network_dir + "/cv12_W");
x42.save(network_dir + "/cv21_B");
x41.save(network_dir + "/cv21_W");
x52.save(network_dir + "/cv22_B");
x51.save(network_dir + "/cv22_W");
x65.save(network_dir + "/cv31_B");
x64.save(network_dir + "/cv31_W");
x75.save(network_dir + "/cv32_B");
x74.save(network_dir + "/cv32_W");
x85.save(network_dir + "/cv33_B");
x84.save(network_dir + "/cv33_W");
x98.save(network_dir + "/cv41_B");
x97.save(network_dir + "/cv41_W");
x108.save(network_dir + "/cv42_B");
x107.save(network_dir + "/cv42_W");
x118.save(network_dir + "/cv43_B");
x117.save(network_dir + "/cv43_W");
x131.save(network_dir + "/cv51_B");
x130.save(network_dir + "/cv51_W");
x141.save(network_dir + "/cv52_B");
x140.save(network_dir + "/cv52_W");
x151.save(network_dir + "/cv53_B");
x150.save(network_dir + "/cv53_W");
x169.save(network_dir + "/fc6_B");
x165.save(network_dir + "/fc6_W");
x185.save(network_dir + "/fc7_B");
x181.save(network_dir + "/fc7_W");
x199.save(network_dir + "/fc8_B");
x195.save(network_dir + "/fc8_W");
x131.free();
x15.free();
x305.free();
x404.free();
x356.free();
x140.free();
x507.free();
x97.free();
x75.free();
x51.free();
x169.free();
x334.free();
x64.free();
x84.free();
x29.free();
x237.free();
x130.free();
x98.free();
x230.free();
x554.free();
x181.free();
x107.free();
x311.free();
x484.free();
x375.free();
x141.free();
x380.free();
x195.free();
x432.free();
x165.free();
x199.free();
x65.free();
x74.free();
x85.free();
x479.free();
x581.free();
x108.free();
x349.free();
x41.free();
x52.free();
x151.free();
x630.free();
x329.free();
x268.free();
x577.free();
x28.free();
x264.free();
x409.free();
x118.free();
x16.free();
x626.free();
x502.free();
x454.free();
x606.free();
x559.free();
x450.free();
x150.free();
x427.free();
x42.free();
x525.free();
x185.free();
x117.free();
x611.free();
x530.free();
x53.free();
x119.free();
x175.free();
x142.free();
x125.free();
x112.free();
x59.free();
x152.free();
x36.free();
x145.free();
x99.free();
x172.free();
x23.free();
x89.free();
x86.free();
x92.free();
x79.free();
x46.free();
x17.free();
x122.free();
x69.free();
x155.free();
x202.free();
x30.free();
x109.free();
x66.free();
x33.free();
x132.free();
x76.free();
x43.free();
x135.free();
x102.free();
x56.free();
x158.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X20 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X66 = Cuda(Indicator(Y, 1000))
JCudaTensor x9;
JTensorFloat x10;
x10 = x4.asIndicator(1000);
x9 = x10.asJCudaTensor();

// val X21 = Convolv(1,1)(X20,cv11_W,cv11_B)
JCudaTensor x11;
JCudaTensor x12, x13, x14;
x12 = x7;
x13 = x15;
x14 = x16;
x11 = x17.forward(x12,x13,x14);

// val X194 = - X66.copy
JCudaTensor x18;
JCudaTensor x19;
float x20;
x19 = x9;
x19 = x19.clone();
x20 = -1;
x18 = x19.times_i(x20);

// val X22 = ReLU()(X21)
JCudaTensor x21;
JCudaTensor x22;
x22 = x11;
x21 = x23.forward(x22);

// val X23 = Convolv(1,1)(X22,cv12_W,cv12_B)
JCudaTensor x24;
JCudaTensor x25, x26, x27;
x25 = x21;
x26 = x28;
x27 = x29;
x24 = x30.forward(x25,x26,x27);

// val X24 = ReLU()(X23)
JCudaTensor x31;
JCudaTensor x32;
x32 = x24;
x31 = x33.forward(x32);

// val X25 = Pooling(2,2,0,true)(X24)
JCudaTensor x34;
JCudaTensor x35;
x35 = x31;
x34 = x36.forward(x35);

// val X26 = Convolv(1,1)(X25,cv21_W,cv21_B)
JCudaTensor x37;
JCudaTensor x38, x39, x40;
x38 = x34;
x39 = x41;
x40 = x42;
x37 = x43.forward(x38,x39,x40);

// val X27 = ReLU()(X26)
JCudaTensor x44;
JCudaTensor x45;
x45 = x37;
x44 = x46.forward(x45);

// val X28 = Convolv(1,1)(X27,cv22_W,cv22_B)
JCudaTensor x47;
JCudaTensor x48, x49, x50;
x48 = x44;
x49 = x51;
x50 = x52;
x47 = x53.forward(x48,x49,x50);

// val X29 = ReLU()(X28)
JCudaTensor x54;
JCudaTensor x55;
x55 = x47;
x54 = x56.forward(x55);

// val X30 = Pooling(2,2,0,true)(X29)
JCudaTensor x57;
JCudaTensor x58;
x58 = x54;
x57 = x59.forward(x58);

// val X31 = Convolv(1,1)(X30,cv31_W,cv31_B)
JCudaTensor x60;
JCudaTensor x61, x62, x63;
x61 = x57;
x62 = x64;
x63 = x65;
x60 = x66.forward(x61,x62,x63);

// val X32 = ReLU()(X31)
JCudaTensor x67;
JCudaTensor x68;
x68 = x60;
x67 = x69.forward(x68);

// val X33 = Convolv(1,1)(X32,cv32_W,cv32_B)
JCudaTensor x70;
JCudaTensor x71, x72, x73;
x71 = x67;
x72 = x74;
x73 = x75;
x70 = x76.forward(x71,x72,x73);

// val X34 = ReLU()(X33)
JCudaTensor x77;
JCudaTensor x78;
x78 = x70;
x77 = x79.forward(x78);

// val X35 = Convolv(1,1)(X34,cv33_W,cv33_B)
JCudaTensor x80;
JCudaTensor x81, x82, x83;
x81 = x77;
x82 = x84;
x83 = x85;
x80 = x86.forward(x81,x82,x83);

// val X36 = ReLU()(X35)
JCudaTensor x87;
JCudaTensor x88;
x88 = x80;
x87 = x89.forward(x88);

// val X37 = Pooling(2,2,0,true)(X36)
JCudaTensor x90;
JCudaTensor x91;
x91 = x87;
x90 = x92.forward(x91);

// val X38 = Convolv(1,1)(X37,cv41_W,cv41_B)
JCudaTensor x93;
JCudaTensor x94, x95, x96;
x94 = x90;
x95 = x97;
x96 = x98;
x93 = x99.forward(x94,x95,x96);

// val X39 = ReLU()(X38)
JCudaTensor x100;
JCudaTensor x101;
x101 = x93;
x100 = x102.forward(x101);

// val X40 = Convolv(1,1)(X39,cv42_W,cv42_B)
JCudaTensor x103;
JCudaTensor x104, x105, x106;
x104 = x100;
x105 = x107;
x106 = x108;
x103 = x109.forward(x104,x105,x106);

// val X41 = ReLU()(X40)
JCudaTensor x110;
JCudaTensor x111;
x111 = x103;
x110 = x112.forward(x111);

// val X42 = Convolv(1,1)(X41,cv43_W,cv43_B)
JCudaTensor x113;
JCudaTensor x114, x115, x116;
x114 = x110;
x115 = x117;
x116 = x118;
x113 = x119.forward(x114,x115,x116);

// val X43 = ReLU()(X42)
JCudaTensor x120;
JCudaTensor x121;
x121 = x113;
x120 = x122.forward(x121);

// val X44 = Pooling(2,2,0,true)(X43)
JCudaTensor x123;
JCudaTensor x124;
x124 = x120;
x123 = x125.forward(x124);

// val X45 = Convolv(1,1)(X44,cv51_W,cv51_B)
JCudaTensor x126;
JCudaTensor x127, x128, x129;
x127 = x123;
x128 = x130;
x129 = x131;
x126 = x132.forward(x127,x128,x129);

// val X46 = ReLU()(X45)
JCudaTensor x133;
JCudaTensor x134;
x134 = x126;
x133 = x135.forward(x134);

// val X47 = Convolv(1,1)(X46,cv52_W,cv52_B)
JCudaTensor x136;
JCudaTensor x137, x138, x139;
x137 = x133;
x138 = x140;
x139 = x141;
x136 = x142.forward(x137,x138,x139);

// val X48 = ReLU()(X47)
JCudaTensor x143;
JCudaTensor x144;
x144 = x136;
x143 = x145.forward(x144);

// val X49 = Convolv(1,1)(X48,cv53_W,cv53_B)
JCudaTensor x146;
JCudaTensor x147, x148, x149;
x147 = x143;
x148 = x150;
x149 = x151;
x146 = x152.forward(x147,x148,x149);

// val X50 = ReLU()(X49)
JCudaTensor x153;
JCudaTensor x154;
x154 = x146;
x153 = x155.forward(x154);

// val X51 = Pooling(2,2,0,true)(X50)
JCudaTensor x156;
JCudaTensor x157;
x157 = x153;
x156 = x158.forward(x157);

// val X52 = (X51[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x159;
JCudaMatrix x160;
JCudaMatrix x161;
JCudaTensor x162;
JCudaTensor x163;
x163 = x156;
x162 = x163.flatten(1, new int[]{512, 7, 7});
x160 = x162.asMatrix(1, true);
JCudaTensor x164;
x164 = x165;
x161 = x164.asMatrix(1, true);
x159 = x160.times(x161);

// val X54 = (X52 + (i) => fc6_B)
JCudaTensor x166;
JCudaTensor x167, x168;
x167 = x159;
x168 = x169;
x166 = x168.copy(64, x167);

// val X55 = ReLU()(X54)
JCudaTensor x170;
JCudaTensor x171;
x171 = x166;
x170 = x172.forward(x171);

// val X56 = Dropout(0.5)(X55)
JCudaTensor x173;
JCudaTensor x174;
x174 = x170;
x173 = x175.forward(x174);

// val X57 = (X56)(i | @) * (fc7_W)(j | @)
JCudaTensor x176;
JCudaMatrix x177;
JCudaMatrix x178;
JCudaTensor x179;
x179 = x173;
x177 = x179.asMatrix(1, true);
JCudaTensor x180;
x180 = x181;
x178 = x180.asMatrix(1, true);
x176 = x177.times(x178);

// val X59 = (X57 + (i) => fc7_B)
JCudaTensor x182;
JCudaTensor x183, x184;
x183 = x176;
x184 = x185;
x182 = x184.copy(64, x183);

// val X60 = ReLU()(X59)
JCudaTensor x186;
JCudaTensor x187;
x187 = x182;
x186 = x172.forward(x187);

// val X61 = Dropout(0.5)(X60)
JCudaTensor x188;
JCudaTensor x189;
x189 = x186;
x188 = x175.forward(x189);

// val X62 = (X61)(i | @) * (fc8_W)(j | @)
JCudaTensor x190;
JCudaMatrix x191;
JCudaMatrix x192;
JCudaTensor x193;
x193 = x188;
x191 = x193.asMatrix(1, true);
JCudaTensor x194;
x194 = x195;
x192 = x194.asMatrix(1, true);
x190 = x191.times(x192);

// val X64 = (X62 + (i) => fc8_B)
JCudaTensor x196;
JCudaTensor x197, x198;
x197 = x190;
x198 = x199;
x196 = x198.copy(64, x197);

// val X65 = LogSoftmax()(X64)
JCudaTensor x200;
JCudaTensor x201;
x201 = x196;
x200 = x202.forward(x201);

// Dealloc(X64)
JCudaTensor x203;
x203 = x196;
x203.free();

// val X195 = (X194 / |64|)
JCudaTensor x204;
JCudaTensor x205;
float x206;
x205 = x18;
float x207;
x207 = 64;
x206 = 1 / x207;
x204 = x205.times_i(x206);

// Print(((0 - (X66 . X65)) / |64|))
float x208;
float x209;
float x210;
float x211;
JCudaTensor x212, x213;
x212 = x9;
x213 = x200;
x211 = x212.dot(x213);
x209 = - x211;
x210 = 64;
x208 = x209 / x210;
System.out.println(x5 + " " + x208);
if (Float.isNaN(x208)) { System.exit(-1); }

// Dealloc(X66)
JCudaTensor x214;
x214 = x9;
x214.free();

// val X242 = X195 * d_LogSoftmax()(X65)/d_X64
JCudaTensor x215;
JCudaTensor x216, x217;
x216 = x204;
x217 = x200;
x215 = x202.backward(x216,x217);

// Dealloc(X195)
JCudaTensor x218;
x218 = x204;
x218.free();

// Dealloc(X65)
JCudaTensor x219;
x219 = x200;
x219.free();

// val m1 = (i829) => fc8_W[@, i829]
JCudaMatrix x220;
JCudaTensor x221;
x221 = x195;
x220 = x221.asMatrix(1, false);

// val m91 = (i71258) => X242[@, i71258]
JCudaMatrix x222;
JCudaTensor x223;
x223 = x215;
x222 = x223.asMatrix(1, false);

// val m93 = (i73314) => X61[@, i73314]
JCudaMatrix x224;
JCudaTensor x225;
x225 = x188;
x224 = x225.asMatrix(1, false);

// val X284 = (X242)(i828 | @) * m1
JCudaTensor x226;
JCudaMatrix x227;
JCudaMatrix x228;
JCudaTensor x229;
x229 = x215;
x227 = x229.asMatrix(1, true);
x228 = x220;
x226 = x227.times(x228);

// V_fc8_B <~~ Sum(m91)
float x231, x232;
x231 = lrn_rate_1;
x232 = momentum;
JCudaMatrix x233;
x233 = x222;
x233.sum(x230, x231, x232);

// val X285 = X284 * d_Dropout(0.5)()/d_X60
JCudaTensor x234;
JCudaTensor x235;
x235 = x226;
x234 = x175.backward(x235);

// Dealloc(X284)
JCudaTensor x236;
x236 = x226;
x236.free();

// V_fc8_W <~~ m91 * m93
float x238, x239;
x238 = lrn_rate_1;
x239 = momentum;
JCudaMatrix x240;
JCudaMatrix x241;
x240 = x222;
x241 = x224;
x240.times(x241, x237, x238, x239);

// Dealloc(X242)
JCudaTensor x242;
x242 = x215;
x242.free();

// Dealloc(X61)
JCudaTensor x243;
x243 = x188;
x243.free();

// fc8_B <~~ V_fc8_B
float x244, x245;
x244 = 1;
x245 = decay_1;
JCudaTensor x246;
x246 = x230;
x199.update(x246, x244, x245);


// fc8_W <~~ V_fc8_W
float x247, x248;
x247 = 1;
x248 = decay_1;
JCudaTensor x249;
x249 = x237;
x195.update(x249, x247, x248);


// val X287 = X285 * d_ReLU()(X60)/d_X59
JCudaTensor x250;
JCudaTensor x251, x252;
x251 = x234;
x252 = x186;
x250 = x172.backward(x251,x252);

// Dealloc(X60)
JCudaTensor x253;
x253 = x186;
x253.free();

// val m2 = (i849) => fc7_W[@, i849]
JCudaMatrix x254;
JCudaTensor x255;
x255 = x181;
x254 = x255.asMatrix(1, false);

// val m87 = (i67097) => X287[@, i67097]
JCudaMatrix x256;
JCudaTensor x257;
x257 = x250;
x256 = x257.asMatrix(1, false);

// val m90 = (i69184) => X56[@, i69184]
JCudaMatrix x258;
JCudaTensor x259;
x259 = x173;
x258 = x259.asMatrix(1, false);

// val X288 = (X287)(i848 | @) * m2
JCudaTensor x260;
JCudaMatrix x261;
JCudaMatrix x262;
JCudaTensor x263;
x263 = x250;
x261 = x263.asMatrix(1, true);
x262 = x254;
x260 = x261.times(x262);

// V_fc7_B <~~ Sum(m87)
float x265, x266;
x265 = lrn_rate_1;
x266 = momentum;
JCudaMatrix x267;
x267 = x256;
x267.sum(x264, x265, x266);

// V_fc7_W <~~ m87 * m90
float x269, x270;
x269 = lrn_rate_1;
x270 = momentum;
JCudaMatrix x271;
JCudaMatrix x272;
x271 = x256;
x272 = x258;
x271.times(x272, x268, x269, x270);

// Dealloc(X287)
JCudaTensor x273;
x273 = x250;
x273.free();

// Dealloc(X56)
JCudaTensor x274;
x274 = x173;
x274.free();

// val X289 = X288 * d_Dropout(0.5)()/d_X55
JCudaTensor x275;
JCudaTensor x276;
x276 = x260;
x275 = x175.backward(x276);

// Dealloc(X288)
JCudaTensor x277;
x277 = x260;
x277.free();

// fc7_B <~~ V_fc7_B
float x278, x279;
x278 = 1;
x279 = decay_1;
JCudaTensor x280;
x280 = x264;
x185.update(x280, x278, x279);


// fc7_W <~~ V_fc7_W
float x281, x282;
x281 = 1;
x282 = decay_1;
JCudaTensor x283;
x283 = x268;
x181.update(x283, x281, x282);


// val X291 = X289 * d_ReLU()(X55)/d_X54
JCudaTensor x284;
JCudaTensor x285, x286;
x285 = x275;
x286 = x170;
x284 = x172.backward(x285,x286);

// Dealloc(X55)
JCudaTensor x287;
x287 = x170;
x287.free();

// val m3 = (i871) => fc6_W[@, i871]
JCudaMatrix x288;
JCudaTensor x289;
x289 = x165;
x288 = x289.asMatrix(1, false);

// val X292 = (X291)(i870 | @) * m3
JCudaTensor x290;
JCudaMatrix x291;
JCudaMatrix x292;
JCudaTensor x293;
x293 = x284;
x291 = x293.asMatrix(1, true);
x292 = x288;
x290 = x291.times(x292);

// val m81 = (i62717) => X291[@, i62717]
JCudaMatrix x294;
JCudaTensor x295;
x295 = x284;
x294 = x295.asMatrix(1, false);

// val m85 = (i64839) => X51[1><3][@, i64839]
JCudaMatrix x296;
JCudaTensor x297;
JCudaTensor x298;
x298 = x156;
x297 = x298.flatten(1, new int[]{512, 7, 7});
x296 = x297.asMatrix(1, false);

// val X294 = X292[1<>3] * d_Pooling(2,2,0,true)(X51,X50)/d_X50
JCudaTensor x299;
JCudaTensor x300, x301, x302;
JCudaTensor x303;
x303 = x290;
x300 = x303.unflatten(1, new int[]{512, 7, 7});
x301 = x156;
x302 = x153;
x299 = x158.backward(x300,x301,x302);

// Dealloc(X292)
JCudaTensor x304;
x304 = x290;
x304.free();

// V_fc6_W <~~ m81 * m85
float x306, x307;
x306 = lrn_rate_1;
x307 = momentum;
JCudaMatrix x308;
JCudaMatrix x309;
x308 = x294;
x309 = x296;
x308.times(x309, x305, x306, x307);

// Dealloc(X51)
JCudaTensor x310;
x310 = x156;
x310.free();

// V_fc6_B <~~ Sum(m81)
float x312, x313;
x312 = lrn_rate_1;
x313 = momentum;
JCudaMatrix x314;
x314 = x294;
x314.sum(x311, x312, x313);

// Dealloc(X291)
JCudaTensor x315;
x315 = x284;
x315.free();

// fc6_W <~~ V_fc6_W
float x316, x317;
x316 = 1;
x317 = decay_1;
JCudaTensor x318;
x318 = x305;
x165.update(x318, x316, x317);


// fc6_B <~~ V_fc6_B
float x319, x320;
x319 = 1;
x320 = decay_1;
JCudaTensor x321;
x321 = x311;
x169.update(x321, x319, x320);


// val X296 = X294 * d_ReLU()(X50)/d_X49
JCudaTensor x322;
JCudaTensor x323, x324;
x323 = x299;
x324 = x153;
x322 = x155.backward(x323,x324);

// Dealloc(X50)
JCudaTensor x325;
x325 = x153;
x325.free();

// val X297 = X296 * d_Convolv(1,1)(cv53_W)/d_X48
JCudaTensor x326;
JCudaTensor x327, x328;
x327 = x322;
x328 = x150;
x326 = x152.backward_data(x327,x328);

// V_cv53_W <~~ X296 * d_Convolv(1,1)(X48)/d_cv53_W
float x330, x331;
x330 = lrn_rate_1;
x331 = momentum;
JCudaTensor x332, x333;
x332 = x322;
x333 = x143;
x152.backward_filter(x332,x333, x329, x330, x331);

// V_cv53_B <~~ X296 * d_Convolv(1,1)()/d_cv53_B
float x335, x336;
x335 = lrn_rate_1;
x336 = momentum;
JCudaTensor x337;
x337 = x322;
x152.backward_bias(x337, x334, x335, x336);

// Dealloc(X296)
JCudaTensor x338;
x338 = x322;
x338.free();

// cv53_W <~~ V_cv53_W
float x339, x340;
x339 = 1;
x340 = decay_1;
JCudaTensor x341;
x341 = x329;
x150.update(x341, x339, x340);


// cv53_B <~~ V_cv53_B
float x342, x343;
x342 = 1;
x343 = decay_1;
JCudaTensor x344;
x344 = x334;
x151.update(x344, x342, x343);


// val X299 = X297 * d_ReLU()(X48)/d_X47
JCudaTensor x345;
JCudaTensor x346, x347;
x346 = x326;
x347 = x143;
x345 = x145.backward(x346,x347);

// Dealloc(X48)
JCudaTensor x348;
x348 = x143;
x348.free();

// V_cv52_B <~~ X299 * d_Convolv(1,1)()/d_cv52_B
float x350, x351;
x350 = lrn_rate_1;
x351 = momentum;
JCudaTensor x352;
x352 = x345;
x142.backward_bias(x352, x349, x350, x351);

// val X300 = X299 * d_Convolv(1,1)(cv52_W)/d_X46
JCudaTensor x353;
JCudaTensor x354, x355;
x354 = x345;
x355 = x140;
x353 = x142.backward_data(x354,x355);

// V_cv52_W <~~ X299 * d_Convolv(1,1)(X46)/d_cv52_W
float x357, x358;
x357 = lrn_rate_1;
x358 = momentum;
JCudaTensor x359, x360;
x359 = x345;
x360 = x133;
x142.backward_filter(x359,x360, x356, x357, x358);

// Dealloc(X299)
JCudaTensor x361;
x361 = x345;
x361.free();

// cv52_B <~~ V_cv52_B
float x362, x363;
x362 = 1;
x363 = decay_1;
JCudaTensor x364;
x364 = x349;
x141.update(x364, x362, x363);


// cv52_W <~~ V_cv52_W
float x365, x366;
x365 = 1;
x366 = decay_1;
JCudaTensor x367;
x367 = x356;
x140.update(x367, x365, x366);


// val X302 = X300 * d_ReLU()(X46)/d_X45
JCudaTensor x368;
JCudaTensor x369, x370;
x369 = x353;
x370 = x133;
x368 = x135.backward(x369,x370);

// Dealloc(X46)
JCudaTensor x371;
x371 = x133;
x371.free();

// val X303 = X302 * d_Convolv(1,1)(cv51_W)/d_X44
JCudaTensor x372;
JCudaTensor x373, x374;
x373 = x368;
x374 = x130;
x372 = x132.backward_data(x373,x374);

// V_cv51_W <~~ X302 * d_Convolv(1,1)(X44)/d_cv51_W
float x376, x377;
x376 = lrn_rate_1;
x377 = momentum;
JCudaTensor x378, x379;
x378 = x368;
x379 = x123;
x132.backward_filter(x378,x379, x375, x376, x377);

// V_cv51_B <~~ X302 * d_Convolv(1,1)()/d_cv51_B
float x381, x382;
x381 = lrn_rate_1;
x382 = momentum;
JCudaTensor x383;
x383 = x368;
x132.backward_bias(x383, x380, x381, x382);

// Dealloc(X302)
JCudaTensor x384;
x384 = x368;
x384.free();

// cv51_W <~~ V_cv51_W
float x385, x386;
x385 = 1;
x386 = decay_1;
JCudaTensor x387;
x387 = x375;
x130.update(x387, x385, x386);


// cv51_B <~~ V_cv51_B
float x388, x389;
x388 = 1;
x389 = decay_1;
JCudaTensor x390;
x390 = x380;
x131.update(x390, x388, x389);


// val X305 = X303 * d_Pooling(2,2,0,true)(X44,X43)/d_X43
JCudaTensor x391;
JCudaTensor x392, x393, x394;
x392 = x372;
x393 = x123;
x394 = x120;
x391 = x125.backward(x392,x393,x394);

// Dealloc(X303)
JCudaTensor x395;
x395 = x372;
x395.free();

// Dealloc(X44)
JCudaTensor x396;
x396 = x123;
x396.free();

// val X307 = X305 * d_ReLU()(X43)/d_X42
JCudaTensor x397;
JCudaTensor x398, x399;
x398 = x391;
x399 = x120;
x397 = x122.backward(x398,x399);

// Dealloc(X43)
JCudaTensor x400;
x400 = x120;
x400.free();

// val X308 = X307 * d_Convolv(1,1)(cv43_W)/d_X41
JCudaTensor x401;
JCudaTensor x402, x403;
x402 = x397;
x403 = x117;
x401 = x119.backward_data(x402,x403);

// V_cv43_W <~~ X307 * d_Convolv(1,1)(X41)/d_cv43_W
float x405, x406;
x405 = lrn_rate_1;
x406 = momentum;
JCudaTensor x407, x408;
x407 = x397;
x408 = x110;
x119.backward_filter(x407,x408, x404, x405, x406);

// V_cv43_B <~~ X307 * d_Convolv(1,1)()/d_cv43_B
float x410, x411;
x410 = lrn_rate_1;
x411 = momentum;
JCudaTensor x412;
x412 = x397;
x119.backward_bias(x412, x409, x410, x411);

// Dealloc(X307)
JCudaTensor x413;
x413 = x397;
x413.free();

// cv43_W <~~ V_cv43_W
float x414, x415;
x414 = 1;
x415 = decay_1;
JCudaTensor x416;
x416 = x404;
x117.update(x416, x414, x415);


// cv43_B <~~ V_cv43_B
float x417, x418;
x417 = 1;
x418 = decay_1;
JCudaTensor x419;
x419 = x409;
x118.update(x419, x417, x418);


// val X310 = X308 * d_ReLU()(X41)/d_X40
JCudaTensor x420;
JCudaTensor x421, x422;
x421 = x401;
x422 = x110;
x420 = x112.backward(x421,x422);

// Dealloc(X41)
JCudaTensor x423;
x423 = x110;
x423.free();

// val X311 = X310 * d_Convolv(1,1)(cv42_W)/d_X39
JCudaTensor x424;
JCudaTensor x425, x426;
x425 = x420;
x426 = x107;
x424 = x109.backward_data(x425,x426);

// V_cv42_W <~~ X310 * d_Convolv(1,1)(X39)/d_cv42_W
float x428, x429;
x428 = lrn_rate_1;
x429 = momentum;
JCudaTensor x430, x431;
x430 = x420;
x431 = x100;
x109.backward_filter(x430,x431, x427, x428, x429);

// V_cv42_B <~~ X310 * d_Convolv(1,1)()/d_cv42_B
float x433, x434;
x433 = lrn_rate_1;
x434 = momentum;
JCudaTensor x435;
x435 = x420;
x109.backward_bias(x435, x432, x433, x434);

// Dealloc(X310)
JCudaTensor x436;
x436 = x420;
x436.free();

// cv42_W <~~ V_cv42_W
float x437, x438;
x437 = 1;
x438 = decay_1;
JCudaTensor x439;
x439 = x427;
x107.update(x439, x437, x438);


// cv42_B <~~ V_cv42_B
float x440, x441;
x440 = 1;
x441 = decay_1;
JCudaTensor x442;
x442 = x432;
x108.update(x442, x440, x441);


// val X313 = X311 * d_ReLU()(X39)/d_X38
JCudaTensor x443;
JCudaTensor x444, x445;
x444 = x424;
x445 = x100;
x443 = x102.backward(x444,x445);

// Dealloc(X39)
JCudaTensor x446;
x446 = x100;
x446.free();

// val X314 = X313 * d_Convolv(1,1)(cv41_W)/d_X37
JCudaTensor x447;
JCudaTensor x448, x449;
x448 = x443;
x449 = x97;
x447 = x99.backward_data(x448,x449);

// V_cv41_B <~~ X313 * d_Convolv(1,1)()/d_cv41_B
float x451, x452;
x451 = lrn_rate_1;
x452 = momentum;
JCudaTensor x453;
x453 = x443;
x99.backward_bias(x453, x450, x451, x452);

// V_cv41_W <~~ X313 * d_Convolv(1,1)(X37)/d_cv41_W
float x455, x456;
x455 = lrn_rate_1;
x456 = momentum;
JCudaTensor x457, x458;
x457 = x443;
x458 = x90;
x99.backward_filter(x457,x458, x454, x455, x456);

// Dealloc(X313)
JCudaTensor x459;
x459 = x443;
x459.free();

// cv41_B <~~ V_cv41_B
float x460, x461;
x460 = 1;
x461 = decay_1;
JCudaTensor x462;
x462 = x450;
x98.update(x462, x460, x461);


// cv41_W <~~ V_cv41_W
float x463, x464;
x463 = 1;
x464 = decay_1;
JCudaTensor x465;
x465 = x454;
x97.update(x465, x463, x464);


// val X316 = X314 * d_Pooling(2,2,0,true)(X37,X36)/d_X36
JCudaTensor x466;
JCudaTensor x467, x468, x469;
x467 = x447;
x468 = x90;
x469 = x87;
x466 = x92.backward(x467,x468,x469);

// Dealloc(X314)
JCudaTensor x470;
x470 = x447;
x470.free();

// Dealloc(X37)
JCudaTensor x471;
x471 = x90;
x471.free();

// val X318 = X316 * d_ReLU()(X36)/d_X35
JCudaTensor x472;
JCudaTensor x473, x474;
x473 = x466;
x474 = x87;
x472 = x89.backward(x473,x474);

// Dealloc(X36)
JCudaTensor x475;
x475 = x87;
x475.free();

// val X319 = X318 * d_Convolv(1,1)(cv33_W)/d_X34
JCudaTensor x476;
JCudaTensor x477, x478;
x477 = x472;
x478 = x84;
x476 = x86.backward_data(x477,x478);

// V_cv33_W <~~ X318 * d_Convolv(1,1)(X34)/d_cv33_W
float x480, x481;
x480 = lrn_rate_1;
x481 = momentum;
JCudaTensor x482, x483;
x482 = x472;
x483 = x77;
x86.backward_filter(x482,x483, x479, x480, x481);

// V_cv33_B <~~ X318 * d_Convolv(1,1)()/d_cv33_B
float x485, x486;
x485 = lrn_rate_1;
x486 = momentum;
JCudaTensor x487;
x487 = x472;
x86.backward_bias(x487, x484, x485, x486);

// Dealloc(X318)
JCudaTensor x488;
x488 = x472;
x488.free();

// cv33_W <~~ V_cv33_W
float x489, x490;
x489 = 1;
x490 = decay_1;
JCudaTensor x491;
x491 = x479;
x84.update(x491, x489, x490);


// cv33_B <~~ V_cv33_B
float x492, x493;
x492 = 1;
x493 = decay_1;
JCudaTensor x494;
x494 = x484;
x85.update(x494, x492, x493);


// val X321 = X319 * d_ReLU()(X34)/d_X33
JCudaTensor x495;
JCudaTensor x496, x497;
x496 = x476;
x497 = x77;
x495 = x79.backward(x496,x497);

// Dealloc(X34)
JCudaTensor x498;
x498 = x77;
x498.free();

// val X322 = X321 * d_Convolv(1,1)(cv32_W)/d_X32
JCudaTensor x499;
JCudaTensor x500, x501;
x500 = x495;
x501 = x74;
x499 = x76.backward_data(x500,x501);

// V_cv32_W <~~ X321 * d_Convolv(1,1)(X32)/d_cv32_W
float x503, x504;
x503 = lrn_rate_1;
x504 = momentum;
JCudaTensor x505, x506;
x505 = x495;
x506 = x67;
x76.backward_filter(x505,x506, x502, x503, x504);

// V_cv32_B <~~ X321 * d_Convolv(1,1)()/d_cv32_B
float x508, x509;
x508 = lrn_rate_1;
x509 = momentum;
JCudaTensor x510;
x510 = x495;
x76.backward_bias(x510, x507, x508, x509);

// Dealloc(X321)
JCudaTensor x511;
x511 = x495;
x511.free();

// cv32_W <~~ V_cv32_W
float x512, x513;
x512 = 1;
x513 = decay_1;
JCudaTensor x514;
x514 = x502;
x74.update(x514, x512, x513);


// cv32_B <~~ V_cv32_B
float x515, x516;
x515 = 1;
x516 = decay_1;
JCudaTensor x517;
x517 = x507;
x75.update(x517, x515, x516);


// val X324 = X322 * d_ReLU()(X32)/d_X31
JCudaTensor x518;
JCudaTensor x519, x520;
x519 = x499;
x520 = x67;
x518 = x69.backward(x519,x520);

// Dealloc(X32)
JCudaTensor x521;
x521 = x67;
x521.free();

// val X325 = X324 * d_Convolv(1,1)(cv31_W)/d_X30
JCudaTensor x522;
JCudaTensor x523, x524;
x523 = x518;
x524 = x64;
x522 = x66.backward_data(x523,x524);

// V_cv31_W <~~ X324 * d_Convolv(1,1)(X30)/d_cv31_W
float x526, x527;
x526 = lrn_rate_1;
x527 = momentum;
JCudaTensor x528, x529;
x528 = x518;
x529 = x57;
x66.backward_filter(x528,x529, x525, x526, x527);

// V_cv31_B <~~ X324 * d_Convolv(1,1)()/d_cv31_B
float x531, x532;
x531 = lrn_rate_1;
x532 = momentum;
JCudaTensor x533;
x533 = x518;
x66.backward_bias(x533, x530, x531, x532);

// Dealloc(X324)
JCudaTensor x534;
x534 = x518;
x534.free();

// cv31_W <~~ V_cv31_W
float x535, x536;
x535 = 1;
x536 = decay_1;
JCudaTensor x537;
x537 = x525;
x64.update(x537, x535, x536);


// cv31_B <~~ V_cv31_B
float x538, x539;
x538 = 1;
x539 = decay_1;
JCudaTensor x540;
x540 = x530;
x65.update(x540, x538, x539);


// val X327 = X325 * d_Pooling(2,2,0,true)(X30,X29)/d_X29
JCudaTensor x541;
JCudaTensor x542, x543, x544;
x542 = x522;
x543 = x57;
x544 = x54;
x541 = x59.backward(x542,x543,x544);

// Dealloc(X325)
JCudaTensor x545;
x545 = x522;
x545.free();

// Dealloc(X30)
JCudaTensor x546;
x546 = x57;
x546.free();

// val X329 = X327 * d_ReLU()(X29)/d_X28
JCudaTensor x547;
JCudaTensor x548, x549;
x548 = x541;
x549 = x54;
x547 = x56.backward(x548,x549);

// Dealloc(X29)
JCudaTensor x550;
x550 = x54;
x550.free();

// val X330 = X329 * d_Convolv(1,1)(cv22_W)/d_X27
JCudaTensor x551;
JCudaTensor x552, x553;
x552 = x547;
x553 = x51;
x551 = x53.backward_data(x552,x553);

// V_cv22_W <~~ X329 * d_Convolv(1,1)(X27)/d_cv22_W
float x555, x556;
x555 = lrn_rate_1;
x556 = momentum;
JCudaTensor x557, x558;
x557 = x547;
x558 = x44;
x53.backward_filter(x557,x558, x554, x555, x556);

// V_cv22_B <~~ X329 * d_Convolv(1,1)()/d_cv22_B
float x560, x561;
x560 = lrn_rate_1;
x561 = momentum;
JCudaTensor x562;
x562 = x547;
x53.backward_bias(x562, x559, x560, x561);

// Dealloc(X329)
JCudaTensor x563;
x563 = x547;
x563.free();

// cv22_W <~~ V_cv22_W
float x564, x565;
x564 = 1;
x565 = decay_1;
JCudaTensor x566;
x566 = x554;
x51.update(x566, x564, x565);


// cv22_B <~~ V_cv22_B
float x567, x568;
x567 = 1;
x568 = decay_1;
JCudaTensor x569;
x569 = x559;
x52.update(x569, x567, x568);


// val X332 = X330 * d_ReLU()(X27)/d_X26
JCudaTensor x570;
JCudaTensor x571, x572;
x571 = x551;
x572 = x44;
x570 = x46.backward(x571,x572);

// Dealloc(X27)
JCudaTensor x573;
x573 = x44;
x573.free();

// val X333 = X332 * d_Convolv(1,1)(cv21_W)/d_X25
JCudaTensor x574;
JCudaTensor x575, x576;
x575 = x570;
x576 = x41;
x574 = x43.backward_data(x575,x576);

// V_cv21_B <~~ X332 * d_Convolv(1,1)()/d_cv21_B
float x578, x579;
x578 = lrn_rate_1;
x579 = momentum;
JCudaTensor x580;
x580 = x570;
x43.backward_bias(x580, x577, x578, x579);

// V_cv21_W <~~ X332 * d_Convolv(1,1)(X25)/d_cv21_W
float x582, x583;
x582 = lrn_rate_1;
x583 = momentum;
JCudaTensor x584, x585;
x584 = x570;
x585 = x34;
x43.backward_filter(x584,x585, x581, x582, x583);

// Dealloc(X332)
JCudaTensor x586;
x586 = x570;
x586.free();

// cv21_B <~~ V_cv21_B
float x587, x588;
x587 = 1;
x588 = decay_1;
JCudaTensor x589;
x589 = x577;
x42.update(x589, x587, x588);


// cv21_W <~~ V_cv21_W
float x590, x591;
x590 = 1;
x591 = decay_1;
JCudaTensor x592;
x592 = x581;
x41.update(x592, x590, x591);


// val X335 = X333 * d_Pooling(2,2,0,true)(X25,X24)/d_X24
JCudaTensor x593;
JCudaTensor x594, x595, x596;
x594 = x574;
x595 = x34;
x596 = x31;
x593 = x36.backward(x594,x595,x596);

// Dealloc(X333)
JCudaTensor x597;
x597 = x574;
x597.free();

// Dealloc(X25)
JCudaTensor x598;
x598 = x34;
x598.free();

// val X337 = X335 * d_ReLU()(X24)/d_X23
JCudaTensor x599;
JCudaTensor x600, x601;
x600 = x593;
x601 = x31;
x599 = x33.backward(x600,x601);

// Dealloc(X24)
JCudaTensor x602;
x602 = x31;
x602.free();

// val X338 = X337 * d_Convolv(1,1)(cv12_W)/d_X22
JCudaTensor x603;
JCudaTensor x604, x605;
x604 = x599;
x605 = x28;
x603 = x30.backward_data(x604,x605);

// V_cv12_W <~~ X337 * d_Convolv(1,1)(X22)/d_cv12_W
float x607, x608;
x607 = lrn_rate_1;
x608 = momentum;
JCudaTensor x609, x610;
x609 = x599;
x610 = x21;
x30.backward_filter(x609,x610, x606, x607, x608);

// V_cv12_B <~~ X337 * d_Convolv(1,1)()/d_cv12_B
float x612, x613;
x612 = lrn_rate_1;
x613 = momentum;
JCudaTensor x614;
x614 = x599;
x30.backward_bias(x614, x611, x612, x613);

// Dealloc(X337)
JCudaTensor x615;
x615 = x599;
x615.free();

// cv12_W <~~ V_cv12_W
float x616, x617;
x616 = 1;
x617 = decay_1;
JCudaTensor x618;
x618 = x606;
x28.update(x618, x616, x617);


// cv12_B <~~ V_cv12_B
float x619, x620;
x619 = 1;
x620 = decay_1;
JCudaTensor x621;
x621 = x611;
x29.update(x621, x619, x620);


// val X340 = X338 * d_ReLU()(X22)/d_X21
JCudaTensor x622;
JCudaTensor x623, x624;
x623 = x603;
x624 = x21;
x622 = x23.backward(x623,x624);

// Dealloc(X22)
JCudaTensor x625;
x625 = x21;
x625.free();

// V_cv11_B <~~ X340 * d_Convolv(1,1)()/d_cv11_B
float x627, x628;
x627 = lrn_rate_1;
x628 = momentum;
JCudaTensor x629;
x629 = x622;
x17.backward_bias(x629, x626, x627, x628);

// V_cv11_W <~~ X340 * d_Convolv(1,1)(X20)/d_cv11_W
float x631, x632;
x631 = lrn_rate_1;
x632 = momentum;
JCudaTensor x633, x634;
x633 = x622;
x634 = x7;
x17.backward_filter(x633,x634, x630, x631, x632);

// Dealloc(X340)
JCudaTensor x635;
x635 = x622;
x635.free();

// Dealloc(X20)
JCudaTensor x636;
x636 = x7;
x636.free();

// cv11_B <~~ V_cv11_B
float x637, x638;
x637 = 1;
x638 = decay_1;
JCudaTensor x639;
x639 = x626;
x16.update(x639, x637, x638);


// cv11_W <~~ V_cv11_W
float x640, x641;
x640 = 1;
x641 = decay_1;
JCudaTensor x642;
x642 = x630;
x15.update(x642, x640, x641);


}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X7385 = Cuda(X)
JCudaTensor x643;
JTensorFloat x644;
x644 = x3;
x643 = x644.asJCudaTensor();

// val X7386 = Convolv(1,1)(X7385,cv11_W,cv11_B)
JCudaTensor x645;
JCudaTensor x646, x647, x648;
x646 = x643;
x647 = x15;
x648 = x16;
x645 = x17.forward(x646,x647,x648);

// Dealloc(X7385)
JCudaTensor x649;
x649 = x643;
x649.free();

// val X7387 = ReLU()(X7386)
JCudaTensor x650;
JCudaTensor x651;
x651 = x645;
x650 = x23.forward(x651);

// val X7388 = Convolv(1,1)(X7387,cv12_W,cv12_B)
JCudaTensor x652;
JCudaTensor x653, x654, x655;
x653 = x650;
x654 = x28;
x655 = x29;
x652 = x30.forward(x653,x654,x655);

// Dealloc(X7387)
JCudaTensor x656;
x656 = x650;
x656.free();

// val X7389 = ReLU()(X7388)
JCudaTensor x657;
JCudaTensor x658;
x658 = x652;
x657 = x33.forward(x658);

// val X7390 = Pooling(2,2,0,true)(X7389)
JCudaTensor x659;
JCudaTensor x660;
x660 = x657;
x659 = x36.forward(x660);

// Dealloc(X7389)
JCudaTensor x661;
x661 = x657;
x661.free();

// val X7391 = Convolv(1,1)(X7390,cv21_W,cv21_B)
JCudaTensor x662;
JCudaTensor x663, x664, x665;
x663 = x659;
x664 = x41;
x665 = x42;
x662 = x43.forward(x663,x664,x665);

// Dealloc(X7390)
JCudaTensor x666;
x666 = x659;
x666.free();

// val X7392 = ReLU()(X7391)
JCudaTensor x667;
JCudaTensor x668;
x668 = x662;
x667 = x46.forward(x668);

// val X7393 = Convolv(1,1)(X7392,cv22_W,cv22_B)
JCudaTensor x669;
JCudaTensor x670, x671, x672;
x670 = x667;
x671 = x51;
x672 = x52;
x669 = x53.forward(x670,x671,x672);

// Dealloc(X7392)
JCudaTensor x673;
x673 = x667;
x673.free();

// val X7394 = ReLU()(X7393)
JCudaTensor x674;
JCudaTensor x675;
x675 = x669;
x674 = x56.forward(x675);

// val X7395 = Pooling(2,2,0,true)(X7394)
JCudaTensor x676;
JCudaTensor x677;
x677 = x674;
x676 = x59.forward(x677);

// Dealloc(X7394)
JCudaTensor x678;
x678 = x674;
x678.free();

// val X7396 = Convolv(1,1)(X7395,cv31_W,cv31_B)
JCudaTensor x679;
JCudaTensor x680, x681, x682;
x680 = x676;
x681 = x64;
x682 = x65;
x679 = x66.forward(x680,x681,x682);

// Dealloc(X7395)
JCudaTensor x683;
x683 = x676;
x683.free();

// val X7397 = ReLU()(X7396)
JCudaTensor x684;
JCudaTensor x685;
x685 = x679;
x684 = x69.forward(x685);

// val X7398 = Convolv(1,1)(X7397,cv32_W,cv32_B)
JCudaTensor x686;
JCudaTensor x687, x688, x689;
x687 = x684;
x688 = x74;
x689 = x75;
x686 = x76.forward(x687,x688,x689);

// Dealloc(X7397)
JCudaTensor x690;
x690 = x684;
x690.free();

// val X7399 = ReLU()(X7398)
JCudaTensor x691;
JCudaTensor x692;
x692 = x686;
x691 = x79.forward(x692);

// val X7400 = Convolv(1,1)(X7399,cv33_W,cv33_B)
JCudaTensor x693;
JCudaTensor x694, x695, x696;
x694 = x691;
x695 = x84;
x696 = x85;
x693 = x86.forward(x694,x695,x696);

// Dealloc(X7399)
JCudaTensor x697;
x697 = x691;
x697.free();

// val X7401 = ReLU()(X7400)
JCudaTensor x698;
JCudaTensor x699;
x699 = x693;
x698 = x89.forward(x699);

// val X7402 = Pooling(2,2,0,true)(X7401)
JCudaTensor x700;
JCudaTensor x701;
x701 = x698;
x700 = x92.forward(x701);

// Dealloc(X7401)
JCudaTensor x702;
x702 = x698;
x702.free();

// val X7403 = Convolv(1,1)(X7402,cv41_W,cv41_B)
JCudaTensor x703;
JCudaTensor x704, x705, x706;
x704 = x700;
x705 = x97;
x706 = x98;
x703 = x99.forward(x704,x705,x706);

// Dealloc(X7402)
JCudaTensor x707;
x707 = x700;
x707.free();

// val X7404 = ReLU()(X7403)
JCudaTensor x708;
JCudaTensor x709;
x709 = x703;
x708 = x102.forward(x709);

// val X7405 = Convolv(1,1)(X7404,cv42_W,cv42_B)
JCudaTensor x710;
JCudaTensor x711, x712, x713;
x711 = x708;
x712 = x107;
x713 = x108;
x710 = x109.forward(x711,x712,x713);

// Dealloc(X7404)
JCudaTensor x714;
x714 = x708;
x714.free();

// val X7406 = ReLU()(X7405)
JCudaTensor x715;
JCudaTensor x716;
x716 = x710;
x715 = x112.forward(x716);

// val X7407 = Convolv(1,1)(X7406,cv43_W,cv43_B)
JCudaTensor x717;
JCudaTensor x718, x719, x720;
x718 = x715;
x719 = x117;
x720 = x118;
x717 = x119.forward(x718,x719,x720);

// Dealloc(X7406)
JCudaTensor x721;
x721 = x715;
x721.free();

// val X7408 = ReLU()(X7407)
JCudaTensor x722;
JCudaTensor x723;
x723 = x717;
x722 = x122.forward(x723);

// val X7409 = Pooling(2,2,0,true)(X7408)
JCudaTensor x724;
JCudaTensor x725;
x725 = x722;
x724 = x125.forward(x725);

// Dealloc(X7408)
JCudaTensor x726;
x726 = x722;
x726.free();

// val X7410 = Convolv(1,1)(X7409,cv51_W,cv51_B)
JCudaTensor x727;
JCudaTensor x728, x729, x730;
x728 = x724;
x729 = x130;
x730 = x131;
x727 = x132.forward(x728,x729,x730);

// Dealloc(X7409)
JCudaTensor x731;
x731 = x724;
x731.free();

// val X7411 = ReLU()(X7410)
JCudaTensor x732;
JCudaTensor x733;
x733 = x727;
x732 = x135.forward(x733);

// val X7412 = Convolv(1,1)(X7411,cv52_W,cv52_B)
JCudaTensor x734;
JCudaTensor x735, x736, x737;
x735 = x732;
x736 = x140;
x737 = x141;
x734 = x142.forward(x735,x736,x737);

// Dealloc(X7411)
JCudaTensor x738;
x738 = x732;
x738.free();

// val X7413 = ReLU()(X7412)
JCudaTensor x739;
JCudaTensor x740;
x740 = x734;
x739 = x145.forward(x740);

// val X7414 = Convolv(1,1)(X7413,cv53_W,cv53_B)
JCudaTensor x741;
JCudaTensor x742, x743, x744;
x742 = x739;
x743 = x150;
x744 = x151;
x741 = x152.forward(x742,x743,x744);

// Dealloc(X7413)
JCudaTensor x745;
x745 = x739;
x745.free();

// val X7415 = ReLU()(X7414)
JCudaTensor x746;
JCudaTensor x747;
x747 = x741;
x746 = x155.forward(x747);

// val X7416 = Pooling(2,2,0,true)(X7415)
JCudaTensor x748;
JCudaTensor x749;
x749 = x746;
x748 = x158.forward(x749);

// Dealloc(X7415)
JCudaTensor x750;
x750 = x746;
x750.free();

// val X7417 = (X7416[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x751;
JCudaMatrix x752;
JCudaMatrix x753;
JCudaTensor x754;
JCudaTensor x755;
x755 = x748;
x754 = x755.flatten(1, new int[]{512, 7, 7});
x752 = x754.asMatrix(1, true);
JCudaTensor x756;
x756 = x165;
x753 = x756.asMatrix(1, true);
x751 = x752.times(x753);

// Dealloc(X7416)
JCudaTensor x757;
x757 = x748;
x757.free();

// val X7419 = (X7417 + (i) => fc6_B)
JCudaTensor x758;
JCudaTensor x759, x760;
x759 = x751;
x760 = x169;
x758 = x760.copy(64, x759);

// val X7420 = ReLU()(X7419)
JCudaTensor x761;
JCudaTensor x762;
x762 = x758;
x761 = x172.forward(x762);

// val X7421 = Dropout(0.5)(X7420)
JCudaTensor x763;
JCudaTensor x764;
x764 = x761;
x763 = x175.forward(x764);

// Dealloc(X7420)
JCudaTensor x765;
x765 = x761;
x765.free();

// val X7422 = (X7421)(i | @) * (fc7_W)(j | @)
JCudaTensor x766;
JCudaMatrix x767;
JCudaMatrix x768;
JCudaTensor x769;
x769 = x763;
x767 = x769.asMatrix(1, true);
JCudaTensor x770;
x770 = x181;
x768 = x770.asMatrix(1, true);
x766 = x767.times(x768);

// Dealloc(X7421)
JCudaTensor x771;
x771 = x763;
x771.free();

// val X7424 = (X7422 + (i) => fc7_B)
JCudaTensor x772;
JCudaTensor x773, x774;
x773 = x766;
x774 = x185;
x772 = x774.copy(64, x773);

// val X7425 = ReLU()(X7424)
JCudaTensor x775;
JCudaTensor x776;
x776 = x772;
x775 = x172.forward(x776);

// val X7426 = Dropout(0.5)(X7425)
JCudaTensor x777;
JCudaTensor x778;
x778 = x775;
x777 = x175.forward(x778);

// Dealloc(X7425)
JCudaTensor x779;
x779 = x775;
x779.free();

// val X7427 = (X7426)(i | @) * (fc8_W)(j | @)
JCudaTensor x780;
JCudaMatrix x781;
JCudaMatrix x782;
JCudaTensor x783;
x783 = x777;
x781 = x783.asMatrix(1, true);
JCudaTensor x784;
x784 = x195;
x782 = x784.asMatrix(1, true);
x780 = x781.times(x782);

// Dealloc(X7426)
JCudaTensor x785;
x785 = x777;
x785.free();

// val X7429 = (X7427 + (i) => fc8_B)
JCudaTensor x786;
JCudaTensor x787, x788;
x787 = x780;
x788 = x199;
x786 = x788.copy(64, x787);

// val X7430 = Cuda(Indicator(Y, 1000))
JCudaTensor x789;
JTensorFloat x790;
x790 = x4.asIndicator(1000);
x789 = x790.asJCudaTensor();

// val X7431 = X7430 .* X7429
JCudaTensor x791;
JCudaTensor x792, x793;
x792 = x789;
x793 = x786;
x791 = x792.times_i(x793);

// val X7432 = Sum((X7431)(i337 | @))
JCudaTensor x794;
JCudaMatrix x795;
JCudaTensor x796;
x796 = x791;
x795 = x796.asMatrix(1, true);
x794 = x795.sum();

// Dealloc(X7431)
JCudaTensor x797;
x797 = x791;
x797.free();

// val X7433 = Max((X7429)(i337 | @))
JCudaTensor x798;
JCudaMatrix x799;
JCudaTensor x800;
x800 = x786;
x799 = x800.asMatrix(1, true);
x798 = x799.max();

// Dealloc(X7429)
JCudaTensor x801;
x801 = x786;
x801.free();

// val X7434 = 1{X7432 == X7433}
JCudaTensor x802;
JCudaTensor x803, x804;
x803 = x794;
x804 = x798;
x802 = x803.eq(x804);

// Dealloc(X7433)
JCudaTensor x805;
x805 = x798;
x805.free();

// Print((Sum(X7434) / |64|))
float x806;
float x807;
float x808;
JCudaTensor x809;
x809 = x802;
x807 = x809.sum();
x808 = 64;
x806 = x807 / x808;
System.out.println(x5 + " test precision "  + x806);

// Dealloc(X7434)
JCudaTensor x810;
x810 = x802;
x810.free();

}
 
}

}