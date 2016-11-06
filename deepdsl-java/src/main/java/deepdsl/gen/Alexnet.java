package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.tensor.*;
import deepdsl.util.*;


public class Alexnet {
static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.9999995f;
// lrn_rate_1
static float lrn_rate_1 = -0.001f;
// lrn_rate_2
static float lrn_rate_2 = -0.002f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/alexnet";
// test_data_path
static String test_data_path = "dataset/imagenet224/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// train_data_path
static String train_data_path = "dataset/imagenet224/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 1000;

// (Convolv(1,1),List(128, 256, 13, 13))
static JCudnnConvolution x67 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{256,384,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(128, 384, 13, 13))
static JCudnnConvolution x57 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{384,384,3,3},new int[]{384}, 1, 1);
// (Convolv(1,1),List(128, 384, 13, 13))
static JCudnnConvolution x47 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{384,256,3,3},new int[]{384}, 1, 1);
// (Convolv(1,2),List(128, 256, 27, 27))
static JCudnnConvolution x31 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
// (Convolv(4,2),List(128, 96, 55, 55))
static JCudnnConvolution x15 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 2);
// (Dropout(0.5),List(128, 4096))
static JCudnnDropout x90 = new JCudnnDropout(new int[]{128,4096}, 0.5f);
// (LMDB,false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, 640, new int[]{128, 3, 224, 224});
// (LMDB,true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, 6400, new int[]{128, 3, 224, 224});
// (LRN(5,1.0E-4,0.75),List(128, 256, 27, 27))
static JCudnnLRN x37 = new JCudnnLRN(new int[]{128,256,27,27}, 5, 1.0E-4, 0.75);
// (LRN(5,1.0E-4,0.75),List(128, 96, 55, 55))
static JCudnnLRN x21 = new JCudnnLRN(new int[]{128,96,55,55}, 5, 1.0E-4, 0.75);
// (Pooling(3,2,0,true),List(128, 256, 6, 6))
static JCudnnPooling x73 = new JCudnnPooling(new int[]{128,256,13,13}, 3, 2, 0, 0);
// (Pooling(3,2,0,true),List(128, 256, 13, 13))
static JCudnnPooling x40 = new JCudnnPooling(new int[]{128,256,27,27}, 3, 2, 0, 0);
// (Pooling(3,2,0,true),List(128, 96, 27, 27))
static JCudnnPooling x24 = new JCudnnPooling(new int[]{128,96,55,55}, 3, 2, 0, 0);
// (ReLU(),List(128, 256, 13, 13))
static JCudnnActivation x70 = new JCudnnActivation(new int[]{128,256,13,13}, 1);
// (ReLU(),List(128, 256, 27, 27))
static JCudnnActivation x34 = new JCudnnActivation(new int[]{128,256,27,27}, 1);
// (ReLU(),List(128, 384, 13, 13))
static JCudnnActivation x60 = new JCudnnActivation(new int[]{128,384,13,13}, 1);
// (ReLU(),List(128, 384, 13, 13))
static JCudnnActivation x50 = new JCudnnActivation(new int[]{128,384,13,13}, 1);
// (ReLU(),List(128, 4096))
static JCudnnActivation x87 = new JCudnnActivation(new int[]{128,4096}, 1);
// (ReLU(),List(128, 96, 55, 55))
static JCudnnActivation x18 = new JCudnnActivation(new int[]{128,96,55,55}, 1);
// (Softmax(),List(128, 1000))
static JCudnnSoftmax x117 = new JCudnnSoftmax(new int[]{128,1000}, 1);
// V_cv1_B
static JCudaTensor x376 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv1_W
static JCudaTensor x370 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
// V_cv2_B
static JCudaTensor x336 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv2_W
static JCudaTensor x343 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
// V_cv3_B
static JCudaTensor x302 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
// V_cv3_W
static JCudaTensor x309 = JTensor.constFloat(0.0f, 384, 256, 3, 3).asJCudaTensor();
// V_cv4_B
static JCudaTensor x287 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
// V_cv4_W
static JCudaTensor x282 = JTensor.constFloat(0.0f, 384, 384, 3, 3).asJCudaTensor();
// V_cv5_B
static JCudaTensor x256 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv5_W
static JCudaTensor x263 = JTensor.constFloat(0.0f, 256, 384, 3, 3).asJCudaTensor();
// V_fc6_B
static JCudaTensor x241 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc6_W
static JCudaTensor x235 = JTensor.constFloat(0.0f, 4096, 9216).asJCudaTensor();
// V_fc7_B
static JCudaTensor x194 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc7_W
static JCudaTensor x198 = JTensor.constFloat(0.0f, 4096, 4096).asJCudaTensor();
// V_fc8_B
static JCudaTensor x163 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc8_W
static JCudaTensor x167 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// cv1_B
static JCudaTensor x14 = JTensor.constFloat(0.0f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
// cv1_W
static JCudaTensor x13 = JTensor.gaussianFloat(0.01f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
// cv2_B
static JCudaTensor x30 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x29 = JTensor.gaussianFloat(0.01f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
// cv3_B
static JCudaTensor x46 = JTensor.constFloat(0.0f, 384).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x45 = JTensor.gaussianFloat(0.01f, 384, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv4_B
static JCudaTensor x56 = JTensor.constFloat(0.1f, 384).load(network_dir + "/cv4_B").asJCudaTensor();
// cv4_W
static JCudaTensor x55 = JTensor.gaussianFloat(0.01f, 384, 384, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
// cv5_B
static JCudaTensor x66 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv5_B").asJCudaTensor();
// cv5_W
static JCudaTensor x65 = JTensor.gaussianFloat(0.01f, 256, 384, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
// fc6_B
static JCudaTensor x84 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x80 = JTensor.gaussianFloat(0.005f, 4096, 9216).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x100 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x96 = JTensor.gaussianFloat(0.005f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x114 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x110 = JTensor.gaussianFloat(0.01f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
ArithStats.isStats = false;
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
System.out.println(ArithStats.outStats());
test();
x14.save(network_dir + "/cv1_B");
x13.save(network_dir + "/cv1_W");
x30.save(network_dir + "/cv2_B");
x29.save(network_dir + "/cv2_W");
x46.save(network_dir + "/cv3_B");
x45.save(network_dir + "/cv3_W");
x56.save(network_dir + "/cv4_B");
x55.save(network_dir + "/cv4_W");
x66.save(network_dir + "/cv5_B");
x65.save(network_dir + "/cv5_W");
x84.save(network_dir + "/fc6_B");
x80.save(network_dir + "/fc6_W");
x100.save(network_dir + "/fc7_B");
x96.save(network_dir + "/fc7_W");
x114.save(network_dir + "/fc8_B");
x110.save(network_dir + "/fc8_W");
x167.free();
x100.free();
x376.free();
x241.free();
x343.free();
x235.free();
x84.free();
x29.free();
x370.free();
x96.free();
x163.free();
x263.free();
x110.free();
x65.free();
x256.free();
x45.free();
x55.free();
x309.free();
x80.free();
x46.free();
x13.free();
x30.free();
x336.free();
x287.free();
x66.free();
x194.free();
x198.free();
x114.free();
x282.free();
x14.free();
x56.free();
x302.free();
x15.free();
x24.free();
x34.free();
x18.free();
x67.free();
x40.free();
x50.free();
x70.free();
x90.free();
x60.free();
x57.free();
x73.free();
x37.free();
x87.free();
x21.free();
x31.free();
x47.free();
x117.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X13 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X14 = Convolv(4,2)(X13,cv1_W,cv1_B)
JCudaTensor x9;
JCudaTensor x10, x11, x12;
x10 = x7;
x11 = x13;
x12 = x14;
x9 = x15.forward(x10,x11,x12);

// val X15 = ReLU()(X14)
JCudaTensor x16;
JCudaTensor x17;
x17 = x9;
x16 = x18.forward(x17);

// val X16 = LRN(5,1.0E-4,0.75)(X15)
JCudaTensor x19;
JCudaTensor x20;
x20 = x16;
x19 = x21.forward(x20);

// val X17 = Pooling(3,2,0,true)(X16)
JCudaTensor x22;
JCudaTensor x23;
x23 = x19;
x22 = x24.forward(x23);

// val X18 = Convolv(1,2)(X17,cv2_W,cv2_B)
JCudaTensor x25;
JCudaTensor x26, x27, x28;
x26 = x22;
x27 = x29;
x28 = x30;
x25 = x31.forward(x26,x27,x28);

// val X19 = ReLU()(X18)
JCudaTensor x32;
JCudaTensor x33;
x33 = x25;
x32 = x34.forward(x33);

// val X20 = LRN(5,1.0E-4,0.75)(X19)
JCudaTensor x35;
JCudaTensor x36;
x36 = x32;
x35 = x37.forward(x36);

// val X21 = Pooling(3,2,0,true)(X20)
JCudaTensor x38;
JCudaTensor x39;
x39 = x35;
x38 = x40.forward(x39);

// val X22 = Convolv(1,1)(X21,cv3_W,cv3_B)
JCudaTensor x41;
JCudaTensor x42, x43, x44;
x42 = x38;
x43 = x45;
x44 = x46;
x41 = x47.forward(x42,x43,x44);

// val X23 = ReLU()(X22)
JCudaTensor x48;
JCudaTensor x49;
x49 = x41;
x48 = x50.forward(x49);

// val X24 = Convolv(1,1)(X23,cv4_W,cv4_B)
JCudaTensor x51;
JCudaTensor x52, x53, x54;
x52 = x48;
x53 = x55;
x54 = x56;
x51 = x57.forward(x52,x53,x54);

// val X25 = ReLU()(X24)
JCudaTensor x58;
JCudaTensor x59;
x59 = x51;
x58 = x60.forward(x59);

// val X26 = Convolv(1,1)(X25,cv5_W,cv5_B)
JCudaTensor x61;
JCudaTensor x62, x63, x64;
x62 = x58;
x63 = x65;
x64 = x66;
x61 = x67.forward(x62,x63,x64);

// val X27 = ReLU()(X26)
JCudaTensor x68;
JCudaTensor x69;
x69 = x61;
x68 = x70.forward(x69);

// val X28 = Pooling(3,2,0,true)(X27)
JCudaTensor x71;
JCudaTensor x72;
x72 = x68;
x71 = x73.forward(x72);

// val X29 = (X28[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x74;
JCudaMatrix x75;
JCudaMatrix x76;
JCudaTensor x77;
JCudaTensor x78;
x78 = x71;
x77 = x78.flatten(1, new int[]{256, 6, 6});
x75 = x77.asMatrix(1, true);
JCudaTensor x79;
x79 = x80;
x76 = x79.asMatrix(1, true);
x74 = x75.times(x76);

// val X31 = (X29 + (i) => fc6_B)
JCudaTensor x81;
JCudaTensor x82, x83;
x82 = x74;
x83 = x84;
x81 = x83.copy(128, x82);

// val X32 = ReLU()(X31)
JCudaTensor x85;
JCudaTensor x86;
x86 = x81;
x85 = x87.forward(x86);

// val X33 = Dropout(0.5)(X32)
JCudaTensor x88;
JCudaTensor x89;
x89 = x85;
x88 = x90.forward(x89);

// val X34 = (X33)(i | @) * (fc7_W)(j | @)
JCudaTensor x91;
JCudaMatrix x92;
JCudaMatrix x93;
JCudaTensor x94;
x94 = x88;
x92 = x94.asMatrix(1, true);
JCudaTensor x95;
x95 = x96;
x93 = x95.asMatrix(1, true);
x91 = x92.times(x93);

// val X36 = (X34 + (i) => fc7_B)
JCudaTensor x97;
JCudaTensor x98, x99;
x98 = x91;
x99 = x100;
x97 = x99.copy(128, x98);

// val X37 = ReLU()(X36)
JCudaTensor x101;
JCudaTensor x102;
x102 = x97;
x101 = x87.forward(x102);

// val X38 = Dropout(0.5)(X37)
JCudaTensor x103;
JCudaTensor x104;
x104 = x101;
x103 = x90.forward(x104);

// val X39 = (X38)(i | @) * (fc8_W)(j | @)
JCudaTensor x105;
JCudaMatrix x106;
JCudaMatrix x107;
JCudaTensor x108;
x108 = x103;
x106 = x108.asMatrix(1, true);
JCudaTensor x109;
x109 = x110;
x107 = x109.asMatrix(1, true);
x105 = x106.times(x107);

// val X41 = (X39 + (i) => fc8_B)
JCudaTensor x111;
JCudaTensor x112, x113;
x112 = x105;
x113 = x114;
x111 = x113.copy(128, x112);

// val X42 = Softmax()(X41)
JCudaTensor x115;
JCudaTensor x116;
x116 = x111;
x115 = x117.forward(x116);

// Dealloc(X41)
JCudaTensor x118;
x118 = x111;
x118.free();

// val X43 = Cuda(Indicator(Y, 1000))
JCudaTensor x119;
JTensorFloat x120;
x120 = x4.asIndicator(1000);
x119 = x120.asJCudaTensor();

// val X44 = Log X42.copy
JCudaTensor x121;
JCudaTensor x122;
x122 = x115;
x122 = x122.clone();
x121 = x122.log();

// val X124 = 1/(X42.copy)
JCudaTensor x123;
JCudaTensor x124;
float x125;
x124 = x115;
x124 = x124.clone();
x125 = -1;
x123 = x124.pow(x125);

// Print(((0 - (X43 . X44)) / |128|))
float x126;
float x127;
float x128;
float x129;
JCudaTensor x130, x131;
x130 = x119;
x131 = x121;
x129 = x130.dot(x131);
x127 = - x129;
x128 = 128;
x126 = x127 / x128;
System.out.println(x5 + " " + x126);
if (Float.isNaN(x126)) { System.exit(-1); }

// Dealloc(X44)
JCudaTensor x132;
x132 = x121;
x132.free();

// val X125 = X43.copy .* X124
JCudaTensor x133;
JCudaTensor x134, x135;
x134 = x119;
x134 = x134.clone();
x135 = x123;
x133 = x134.times_i(x135);

// Dealloc(X124)
JCudaTensor x136;
x136 = x123;
x136.free();

// Dealloc(X43)
JCudaTensor x137;
x137 = x119;
x137.free();

// val X126 = - X125
JCudaTensor x138;
JCudaTensor x139;
float x140;
x139 = x133;
x140 = -1;
x138 = x139.times_i(x140);

// val X127 = (X126 / |128|)
JCudaTensor x141;
JCudaTensor x142;
float x143;
x142 = x138;
float x144;
x144 = 128;
x143 = 1 / x144;
x141 = x142.times_i(x143);

// val X158 = X127 * d_Softmax()(X42)/d_X41
JCudaTensor x145;
JCudaTensor x146, x147;
x146 = x141;
x147 = x115;
x145 = x117.backward(x146,x147);

// Dealloc(X127)
JCudaTensor x148;
x148 = x141;
x148.free();

// Dealloc(X42)
JCudaTensor x149;
x149 = x115;
x149.free();

// val m1 = (i495) => fc8_W[@, i495]
JCudaMatrix x150;
JCudaTensor x151;
x151 = x110;
x150 = x151.asMatrix(1, false);

// val m43 = (i19110) => X158[@, i19110]
JCudaMatrix x152;
JCudaTensor x153;
x153 = x145;
x152 = x153.asMatrix(1, false);

// val m45 = (i20274) => X38[@, i20274]
JCudaMatrix x154;
JCudaTensor x155;
x155 = x103;
x154 = x155.asMatrix(1, false);

// val X184 = (X158)(i494 | @) * m1
JCudaTensor x156;
JCudaMatrix x157;
JCudaMatrix x158;
JCudaTensor x159;
x159 = x145;
x157 = x159.asMatrix(1, true);
x158 = x150;
x156 = x157.times(x158);

// val X185 = X184 * d_Dropout(0.5)()/d_X37
JCudaTensor x160;
JCudaTensor x161;
x161 = x156;
x160 = x90.backward(x161);

// Dealloc(X184)
JCudaTensor x162;
x162 = x156;
x162.free();

// V_fc8_B <~~ Sum(m43)
float x164, x165;
x164 = lrn_rate_2;
x165 = momentum;
JCudaMatrix x166;
x166 = x152;
x166.sum(x163, x164, x165);

// V_fc8_W <~~ m43 * m45
float x168, x169;
x168 = lrn_rate_1;
x169 = momentum;
JCudaMatrix x170;
JCudaMatrix x171;
x170 = x152;
x171 = x154;
x170.times(x171, x167, x168, x169);

// Dealloc(X158)
JCudaTensor x172;
x172 = x145;
x172.free();

// Dealloc(X38)
JCudaTensor x173;
x173 = x103;
x173.free();

// fc8_B <~~ V_fc8_B
float x174, x175;
x174 = 1;
x175 = 1;
JCudaTensor x176;
x176 = x163;
x114.update(x176, x174, x175);


// fc8_W <~~ V_fc8_W
float x177, x178;
x177 = 1;
x178 = decay_1;
JCudaTensor x179;
x179 = x167;
x110.update(x179, x177, x178);


// val X187 = X185 * d_ReLU()(X37)/d_X36
JCudaTensor x180;
JCudaTensor x181, x182;
x181 = x160;
x182 = x101;
x180 = x87.backward(x181,x182);

// Dealloc(X37)
JCudaTensor x183;
x183 = x101;
x183.free();

// val m2 = (i515) => fc7_W[@, i515]
JCudaMatrix x184;
JCudaTensor x185;
x185 = x96;
x184 = x185.asMatrix(1, false);

// val X188 = (X187)(i514 | @) * m2
JCudaTensor x186;
JCudaMatrix x187;
JCudaMatrix x188;
JCudaTensor x189;
x189 = x180;
x187 = x189.asMatrix(1, true);
x188 = x184;
x186 = x187.times(x188);

// val m39 = (i16733) => X187[@, i16733]
JCudaMatrix x190;
JCudaTensor x191;
x191 = x180;
x190 = x191.asMatrix(1, false);

// val m42 = (i17928) => X33[@, i17928]
JCudaMatrix x192;
JCudaTensor x193;
x193 = x88;
x192 = x193.asMatrix(1, false);

// V_fc7_B <~~ Sum(m39)
float x195, x196;
x195 = lrn_rate_2;
x196 = momentum;
JCudaMatrix x197;
x197 = x190;
x197.sum(x194, x195, x196);

// V_fc7_W <~~ m39 * m42
float x199, x200;
x199 = lrn_rate_1;
x200 = momentum;
JCudaMatrix x201;
JCudaMatrix x202;
x201 = x190;
x202 = x192;
x201.times(x202, x198, x199, x200);

// Dealloc(X187)
JCudaTensor x203;
x203 = x180;
x203.free();

// Dealloc(X33)
JCudaTensor x204;
x204 = x88;
x204.free();

// val X189 = X188 * d_Dropout(0.5)()/d_X32
JCudaTensor x205;
JCudaTensor x206;
x206 = x186;
x205 = x90.backward(x206);

// Dealloc(X188)
JCudaTensor x207;
x207 = x186;
x207.free();

// fc7_B <~~ V_fc7_B
float x208, x209;
x208 = 1;
x209 = 1;
JCudaTensor x210;
x210 = x194;
x100.update(x210, x208, x209);


// fc7_W <~~ V_fc7_W
float x211, x212;
x211 = 1;
x212 = decay_1;
JCudaTensor x213;
x213 = x198;
x96.update(x213, x211, x212);


// val X191 = X189 * d_ReLU()(X32)/d_X31
JCudaTensor x214;
JCudaTensor x215, x216;
x215 = x205;
x216 = x85;
x214 = x87.backward(x215,x216);

// Dealloc(X32)
JCudaTensor x217;
x217 = x85;
x217.free();

// val m3 = (i537) => fc6_W[@, i537]
JCudaMatrix x218;
JCudaTensor x219;
x219 = x80;
x218 = x219.asMatrix(1, false);

// val m33 = (i14183) => X191[@, i14183]
JCudaMatrix x220;
JCudaTensor x221;
x221 = x214;
x220 = x221.asMatrix(1, false);

// val m37 = (i15413) => X28[1><3][@, i15413]
JCudaMatrix x222;
JCudaTensor x223;
JCudaTensor x224;
x224 = x71;
x223 = x224.flatten(1, new int[]{256, 6, 6});
x222 = x223.asMatrix(1, false);

// val X192 = (X191)(i536 | @) * m3
JCudaTensor x225;
JCudaMatrix x226;
JCudaMatrix x227;
JCudaTensor x228;
x228 = x214;
x226 = x228.asMatrix(1, true);
x227 = x218;
x225 = x226.times(x227);

// val X194 = X192[1<>3] * d_Pooling(3,2,0,true)(X28,X27)/d_X27
JCudaTensor x229;
JCudaTensor x230, x231, x232;
JCudaTensor x233;
x233 = x225;
x230 = x233.unflatten(1, new int[]{256, 6, 6});
x231 = x71;
x232 = x68;
x229 = x73.backward(x230,x231,x232);

// Dealloc(X192)
JCudaTensor x234;
x234 = x225;
x234.free();

// V_fc6_W <~~ m33 * m37
float x236, x237;
x236 = lrn_rate_1;
x237 = momentum;
JCudaMatrix x238;
JCudaMatrix x239;
x238 = x220;
x239 = x222;
x238.times(x239, x235, x236, x237);

// Dealloc(X28)
JCudaTensor x240;
x240 = x71;
x240.free();

// V_fc6_B <~~ Sum(m33)
float x242, x243;
x242 = lrn_rate_2;
x243 = momentum;
JCudaMatrix x244;
x244 = x220;
x244.sum(x241, x242, x243);

// Dealloc(X191)
JCudaTensor x245;
x245 = x214;
x245.free();

// fc6_W <~~ V_fc6_W
float x246, x247;
x246 = 1;
x247 = decay_1;
JCudaTensor x248;
x248 = x235;
x80.update(x248, x246, x247);


// fc6_B <~~ V_fc6_B
float x249, x250;
x249 = 1;
x250 = 1;
JCudaTensor x251;
x251 = x241;
x84.update(x251, x249, x250);


// val X196 = X194 * d_ReLU()(X27)/d_X26
JCudaTensor x252;
JCudaTensor x253, x254;
x253 = x229;
x254 = x68;
x252 = x70.backward(x253,x254);

// Dealloc(X27)
JCudaTensor x255;
x255 = x68;
x255.free();

// V_cv5_B <~~ X196 * d_Convolv(1,1)()/d_cv5_B
float x257, x258;
x257 = lrn_rate_2;
x258 = momentum;
JCudaTensor x259;
x259 = x252;
x67.backward_bias(x259, x256, x257, x258);

// val X197 = X196 * d_Convolv(1,1)(cv5_W)/d_X25
JCudaTensor x260;
JCudaTensor x261, x262;
x261 = x252;
x262 = x65;
x260 = x67.backward_data(x261,x262);

// V_cv5_W <~~ X196 * d_Convolv(1,1)(X25)/d_cv5_W
float x264, x265;
x264 = lrn_rate_1;
x265 = momentum;
JCudaTensor x266, x267;
x266 = x252;
x267 = x58;
x67.backward_filter(x266,x267, x263, x264, x265);

// Dealloc(X196)
JCudaTensor x268;
x268 = x252;
x268.free();

// cv5_B <~~ V_cv5_B
float x269, x270;
x269 = 1;
x270 = 1;
JCudaTensor x271;
x271 = x256;
x66.update(x271, x269, x270);


// cv5_W <~~ V_cv5_W
float x272, x273;
x272 = 1;
x273 = decay_1;
JCudaTensor x274;
x274 = x263;
x65.update(x274, x272, x273);


// val X199 = X197 * d_ReLU()(X25)/d_X24
JCudaTensor x275;
JCudaTensor x276, x277;
x276 = x260;
x277 = x58;
x275 = x60.backward(x276,x277);

// Dealloc(X25)
JCudaTensor x278;
x278 = x58;
x278.free();

// val X200 = X199 * d_Convolv(1,1)(cv4_W)/d_X23
JCudaTensor x279;
JCudaTensor x280, x281;
x280 = x275;
x281 = x55;
x279 = x57.backward_data(x280,x281);

// V_cv4_W <~~ X199 * d_Convolv(1,1)(X23)/d_cv4_W
float x283, x284;
x283 = lrn_rate_1;
x284 = momentum;
JCudaTensor x285, x286;
x285 = x275;
x286 = x48;
x57.backward_filter(x285,x286, x282, x283, x284);

// V_cv4_B <~~ X199 * d_Convolv(1,1)()/d_cv4_B
float x288, x289;
x288 = lrn_rate_2;
x289 = momentum;
JCudaTensor x290;
x290 = x275;
x57.backward_bias(x290, x287, x288, x289);

// Dealloc(X199)
JCudaTensor x291;
x291 = x275;
x291.free();

// cv4_W <~~ V_cv4_W
float x292, x293;
x292 = 1;
x293 = decay_1;
JCudaTensor x294;
x294 = x282;
x55.update(x294, x292, x293);


// cv4_B <~~ V_cv4_B
float x295, x296;
x295 = 1;
x296 = 1;
JCudaTensor x297;
x297 = x287;
x56.update(x297, x295, x296);


// val X202 = X200 * d_ReLU()(X23)/d_X22
JCudaTensor x298;
JCudaTensor x299, x300;
x299 = x279;
x300 = x48;
x298 = x50.backward(x299,x300);

// Dealloc(X23)
JCudaTensor x301;
x301 = x48;
x301.free();

// V_cv3_B <~~ X202 * d_Convolv(1,1)()/d_cv3_B
float x303, x304;
x303 = lrn_rate_2;
x304 = momentum;
JCudaTensor x305;
x305 = x298;
x47.backward_bias(x305, x302, x303, x304);

// val X203 = X202 * d_Convolv(1,1)(cv3_W)/d_X21
JCudaTensor x306;
JCudaTensor x307, x308;
x307 = x298;
x308 = x45;
x306 = x47.backward_data(x307,x308);

// V_cv3_W <~~ X202 * d_Convolv(1,1)(X21)/d_cv3_W
float x310, x311;
x310 = lrn_rate_1;
x311 = momentum;
JCudaTensor x312, x313;
x312 = x298;
x313 = x38;
x47.backward_filter(x312,x313, x309, x310, x311);

// Dealloc(X202)
JCudaTensor x314;
x314 = x298;
x314.free();

// cv3_B <~~ V_cv3_B
float x315, x316;
x315 = 1;
x316 = 1;
JCudaTensor x317;
x317 = x302;
x46.update(x317, x315, x316);


// cv3_W <~~ V_cv3_W
float x318, x319;
x318 = 1;
x319 = decay_1;
JCudaTensor x320;
x320 = x309;
x45.update(x320, x318, x319);


// val X205 = X203 * d_Pooling(3,2,0,true)(X21,X20)/d_X20
JCudaTensor x321;
JCudaTensor x322, x323, x324;
x322 = x306;
x323 = x38;
x324 = x35;
x321 = x40.backward(x322,x323,x324);

// Dealloc(X203)
JCudaTensor x325;
x325 = x306;
x325.free();

// Dealloc(X21)
JCudaTensor x326;
x326 = x38;
x326.free();

// val X207 = X205 * d_LRN(5,1.0E-4,0.75)(X20,X19)/d_X19
JCudaTensor x327;
JCudaTensor x328, x329, x330;
x328 = x321;
x329 = x35;
x330 = x32;
x327 = x37.backward(x328,x329,x330);

// Dealloc(X20)
JCudaTensor x331;
x331 = x35;
x331.free();

// val X209 = X207 * d_ReLU()(X19)/d_X18
JCudaTensor x332;
JCudaTensor x333, x334;
x333 = x327;
x334 = x32;
x332 = x34.backward(x333,x334);

// Dealloc(X19)
JCudaTensor x335;
x335 = x32;
x335.free();

// V_cv2_B <~~ X209 * d_Convolv(1,2)()/d_cv2_B
float x337, x338;
x337 = lrn_rate_2;
x338 = momentum;
JCudaTensor x339;
x339 = x332;
x31.backward_bias(x339, x336, x337, x338);

// val X210 = X209 * d_Convolv(1,2)(cv2_W)/d_X17
JCudaTensor x340;
JCudaTensor x341, x342;
x341 = x332;
x342 = x29;
x340 = x31.backward_data(x341,x342);

// V_cv2_W <~~ X209 * d_Convolv(1,2)(X17)/d_cv2_W
float x344, x345;
x344 = lrn_rate_1;
x345 = momentum;
JCudaTensor x346, x347;
x346 = x332;
x347 = x22;
x31.backward_filter(x346,x347, x343, x344, x345);

// Dealloc(X209)
JCudaTensor x348;
x348 = x332;
x348.free();

// cv2_B <~~ V_cv2_B
float x349, x350;
x349 = 1;
x350 = 1;
JCudaTensor x351;
x351 = x336;
x30.update(x351, x349, x350);


// cv2_W <~~ V_cv2_W
float x352, x353;
x352 = 1;
x353 = decay_1;
JCudaTensor x354;
x354 = x343;
x29.update(x354, x352, x353);


// val X212 = X210 * d_Pooling(3,2,0,true)(X17,X16)/d_X16
JCudaTensor x355;
JCudaTensor x356, x357, x358;
x356 = x340;
x357 = x22;
x358 = x19;
x355 = x24.backward(x356,x357,x358);

// Dealloc(X210)
JCudaTensor x359;
x359 = x340;
x359.free();

// Dealloc(X17)
JCudaTensor x360;
x360 = x22;
x360.free();

// val X214 = X212 * d_LRN(5,1.0E-4,0.75)(X16,X15)/d_X15
JCudaTensor x361;
JCudaTensor x362, x363, x364;
x362 = x355;
x363 = x19;
x364 = x16;
x361 = x21.backward(x362,x363,x364);

// Dealloc(X16)
JCudaTensor x365;
x365 = x19;
x365.free();

// val X216 = X214 * d_ReLU()(X15)/d_X14
JCudaTensor x366;
JCudaTensor x367, x368;
x367 = x361;
x368 = x16;
x366 = x18.backward(x367,x368);

// Dealloc(X15)
JCudaTensor x369;
x369 = x16;
x369.free();

// V_cv1_W <~~ X216 * d_Convolv(4,2)(X13)/d_cv1_W
float x371, x372;
x371 = lrn_rate_1;
x372 = momentum;
JCudaTensor x373, x374;
x373 = x366;
x374 = x7;
x15.backward_filter(x373,x374, x370, x371, x372);

// Dealloc(X13)
JCudaTensor x375;
x375 = x7;
x375.free();

// V_cv1_B <~~ X216 * d_Convolv(4,2)()/d_cv1_B
float x377, x378;
x377 = lrn_rate_2;
x378 = momentum;
JCudaTensor x379;
x379 = x366;
x15.backward_bias(x379, x376, x377, x378);

// Dealloc(X216)
JCudaTensor x380;
x380 = x366;
x380.free();

// cv1_W <~~ V_cv1_W
float x381, x382;
x381 = 1;
x382 = decay_1;
JCudaTensor x383;
x383 = x370;
x13.update(x383, x381, x382);


// cv1_B <~~ V_cv1_B
float x384, x385;
x384 = 1;
x385 = 1;
JCudaTensor x386;
x386 = x376;
x14.update(x386, x384, x385);


}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X2341 = Cuda(X)
JCudaTensor x387;
JTensorFloat x388;
x388 = x3;
x387 = x388.asJCudaTensor();

// val X2342 = Convolv(4,2)(X2341,cv1_W,cv1_B)
JCudaTensor x389;
JCudaTensor x390, x391, x392;
x390 = x387;
x391 = x13;
x392 = x14;
x389 = x15.forward(x390,x391,x392);

// Dealloc(X2341)
JCudaTensor x393;
x393 = x387;
x393.free();

// val X2343 = ReLU()(X2342)
JCudaTensor x394;
JCudaTensor x395;
x395 = x389;
x394 = x18.forward(x395);

// val X2344 = LRN(5,1.0E-4,0.75)(X2343)
JCudaTensor x396;
JCudaTensor x397;
x397 = x394;
x396 = x21.forward(x397);

// Dealloc(X2343)
JCudaTensor x398;
x398 = x394;
x398.free();

// val X2345 = Pooling(3,2,0,true)(X2344)
JCudaTensor x399;
JCudaTensor x400;
x400 = x396;
x399 = x24.forward(x400);

// Dealloc(X2344)
JCudaTensor x401;
x401 = x396;
x401.free();

// val X2346 = Convolv(1,2)(X2345,cv2_W,cv2_B)
JCudaTensor x402;
JCudaTensor x403, x404, x405;
x403 = x399;
x404 = x29;
x405 = x30;
x402 = x31.forward(x403,x404,x405);

// Dealloc(X2345)
JCudaTensor x406;
x406 = x399;
x406.free();

// val X2347 = ReLU()(X2346)
JCudaTensor x407;
JCudaTensor x408;
x408 = x402;
x407 = x34.forward(x408);

// val X2348 = LRN(5,1.0E-4,0.75)(X2347)
JCudaTensor x409;
JCudaTensor x410;
x410 = x407;
x409 = x37.forward(x410);

// Dealloc(X2347)
JCudaTensor x411;
x411 = x407;
x411.free();

// val X2349 = Pooling(3,2,0,true)(X2348)
JCudaTensor x412;
JCudaTensor x413;
x413 = x409;
x412 = x40.forward(x413);

// Dealloc(X2348)
JCudaTensor x414;
x414 = x409;
x414.free();

// val X2350 = Convolv(1,1)(X2349,cv3_W,cv3_B)
JCudaTensor x415;
JCudaTensor x416, x417, x418;
x416 = x412;
x417 = x45;
x418 = x46;
x415 = x47.forward(x416,x417,x418);

// Dealloc(X2349)
JCudaTensor x419;
x419 = x412;
x419.free();

// val X2351 = ReLU()(X2350)
JCudaTensor x420;
JCudaTensor x421;
x421 = x415;
x420 = x50.forward(x421);

// val X2352 = Convolv(1,1)(X2351,cv4_W,cv4_B)
JCudaTensor x422;
JCudaTensor x423, x424, x425;
x423 = x420;
x424 = x55;
x425 = x56;
x422 = x57.forward(x423,x424,x425);

// Dealloc(X2351)
JCudaTensor x426;
x426 = x420;
x426.free();

// val X2353 = ReLU()(X2352)
JCudaTensor x427;
JCudaTensor x428;
x428 = x422;
x427 = x60.forward(x428);

// val X2354 = Convolv(1,1)(X2353,cv5_W,cv5_B)
JCudaTensor x429;
JCudaTensor x430, x431, x432;
x430 = x427;
x431 = x65;
x432 = x66;
x429 = x67.forward(x430,x431,x432);

// Dealloc(X2353)
JCudaTensor x433;
x433 = x427;
x433.free();

// val X2355 = ReLU()(X2354)
JCudaTensor x434;
JCudaTensor x435;
x435 = x429;
x434 = x70.forward(x435);

// val X2356 = Pooling(3,2,0,true)(X2355)
JCudaTensor x436;
JCudaTensor x437;
x437 = x434;
x436 = x73.forward(x437);

// Dealloc(X2355)
JCudaTensor x438;
x438 = x434;
x438.free();

// val X2357 = (X2356[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x439;
JCudaMatrix x440;
JCudaMatrix x441;
JCudaTensor x442;
JCudaTensor x443;
x443 = x436;
x442 = x443.flatten(1, new int[]{256, 6, 6});
x440 = x442.asMatrix(1, true);
JCudaTensor x444;
x444 = x80;
x441 = x444.asMatrix(1, true);
x439 = x440.times(x441);

// Dealloc(X2356)
JCudaTensor x445;
x445 = x436;
x445.free();

// val X2359 = (X2357 + (i) => fc6_B)
JCudaTensor x446;
JCudaTensor x447, x448;
x447 = x439;
x448 = x84;
x446 = x448.copy(128, x447);

// val X2360 = ReLU()(X2359)
JCudaTensor x449;
JCudaTensor x450;
x450 = x446;
x449 = x87.forward(x450);

// val X2361 = Dropout(0.5)(X2360)
JCudaTensor x451;
JCudaTensor x452;
x452 = x449;
x451 = x90.forward(x452);

// Dealloc(X2360)
JCudaTensor x453;
x453 = x449;
x453.free();

// val X2362 = (X2361)(i | @) * (fc7_W)(j | @)
JCudaTensor x454;
JCudaMatrix x455;
JCudaMatrix x456;
JCudaTensor x457;
x457 = x451;
x455 = x457.asMatrix(1, true);
JCudaTensor x458;
x458 = x96;
x456 = x458.asMatrix(1, true);
x454 = x455.times(x456);

// Dealloc(X2361)
JCudaTensor x459;
x459 = x451;
x459.free();

// val X2364 = (X2362 + (i) => fc7_B)
JCudaTensor x460;
JCudaTensor x461, x462;
x461 = x454;
x462 = x100;
x460 = x462.copy(128, x461);

// val X2365 = ReLU()(X2364)
JCudaTensor x463;
JCudaTensor x464;
x464 = x460;
x463 = x87.forward(x464);

// val X2366 = Dropout(0.5)(X2365)
JCudaTensor x465;
JCudaTensor x466;
x466 = x463;
x465 = x90.forward(x466);

// Dealloc(X2365)
JCudaTensor x467;
x467 = x463;
x467.free();

// val X2367 = (X2366)(i | @) * (fc8_W)(j | @)
JCudaTensor x468;
JCudaMatrix x469;
JCudaMatrix x470;
JCudaTensor x471;
x471 = x465;
x469 = x471.asMatrix(1, true);
JCudaTensor x472;
x472 = x110;
x470 = x472.asMatrix(1, true);
x468 = x469.times(x470);

// Dealloc(X2366)
JCudaTensor x473;
x473 = x465;
x473.free();

// val X2369 = (X2367 + (i) => fc8_B)
JCudaTensor x474;
JCudaTensor x475, x476;
x475 = x468;
x476 = x114;
x474 = x476.copy(128, x475);

// val X2370 = Cuda(Indicator(Y, 1000))
JCudaTensor x477;
JTensorFloat x478;
x478 = x4.asIndicator(1000);
x477 = x478.asJCudaTensor();

// val X2371 = X2370 .* X2369
JCudaTensor x479;
JCudaTensor x480, x481;
x480 = x477;
x481 = x474;
x479 = x480.times_i(x481);

// val X2372 = Sum((X2371)(i209 | @))
JCudaTensor x482;
JCudaMatrix x483;
JCudaTensor x484;
x484 = x479;
x483 = x484.asMatrix(1, true);
x482 = x483.sum();

// Dealloc(X2371)
JCudaTensor x485;
x485 = x479;
x485.free();

// val X2373 = Max((X2369)(i209 | @))
JCudaTensor x486;
JCudaMatrix x487;
JCudaTensor x488;
x488 = x474;
x487 = x488.asMatrix(1, true);
x486 = x487.max();

// Dealloc(X2369)
JCudaTensor x489;
x489 = x474;
x489.free();

// val X2374 = 1{X2372 == X2373}
JCudaTensor x490;
JCudaTensor x491, x492;
x491 = x482;
x492 = x486;
x490 = x491.eq(x492);

// Dealloc(X2373)
JCudaTensor x493;
x493 = x486;
x493.free();

// Print((Sum(X2374) / |128|))
float x494;
float x495;
float x496;
JCudaTensor x497;
x497 = x490;
x495 = x497.sum();
x496 = 128;
x494 = x495 / x496;
System.out.println(x5 + " test precision "  + x494);

// Dealloc(X2374)
JCudaTensor x498;
x498 = x490;
x498.free();

}
 
}

}