package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Overfeat {
// comment the line below for memory efficient mode
 static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.999995f;
// lrn_rate_1
static float lrn_rate_1 = -0.01f;
// lrn_rate_2
static float lrn_rate_2 = -0.02f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/overfeat";
// platform
static LmdbUtils.OS platform = LmdbUtils.OS.WINDOWS;
// test_data_path
static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// test_size
static int test_size = 10000;
// train_data_path
static String train_data_path = "dataset/imagenet/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 1000;
// train_size
static int train_size = 1000000;

// (Convolv(1,1),List(List(128, 1024, 13, 13), List(1024, 1024, 3, 3), List(1024)))
static JCudnnConvolution x66 = new JCudnnConvolution(new int[]{128,1024,13,13},new int[]{1024,1024,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,1),List(List(128, 256, 13, 13), List(512, 256, 3, 3), List(512)))
static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{512,256,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(128, 512, 13, 13), List(1024, 512, 3, 3), List(1024)))
static JCudnnConvolution x56 = new JCudnnConvolution(new int[]{128,512,13,13},new int[]{1024,512,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,2),List(List(128, 96, 27, 27), List(256, 96, 5, 5), List(256)))
static JCudnnConvolution x33 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
// (Convolv(4,0),List(List(128, 3, 224, 224), List(96, 3, 11, 11), List(96)))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 0);
// (Lmdb(1000000,10000,Win32,1000),false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
// (Lmdb(1000000,10000,Win32,1000),true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, platform, 1000, false);
// (LogSoftmax(),List(List(128, 1000)))
static JCudnnSoftmax x105 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
// (Pooling(2,2,0,true),List(List(128, 1024, 13, 13)))
static JCudnnPooling x71 = new JCudnnPooling(new int[]{128,1024,13,13}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(128, 256, 27, 27)))
static JCudnnPooling x39 = new JCudnnPooling(new int[]{128,256,27,27}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(128, 96, 54, 54)))
static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,96,54,54}, 2, 2, 0, PoolingType.MAX);
// (ReLU(),List(List(128, 1024, 13, 13)))
static JCudnnActivation x59 = new JCudnnActivation(new int[]{128,1024,13,13}, ActivationMode.RELU);
// (ReLU(),List(List(128, 256, 27, 27)))
static JCudnnActivation x36 = new JCudnnActivation(new int[]{128,256,27,27}, ActivationMode.RELU);
// (ReLU(),List(List(128, 512, 13, 13)))
static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,512,13,13}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 54, 54)))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,96,54,54}, ActivationMode.RELU);
// BatchSum(((Sum(X1794) / |128|) / 10))
static float x427;
// V_cv1_B
static JCudaTensor x325 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv1_W
static JCudaTensor x319 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
// V_cv2_B
static JCudaTensor x293 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv2_W
static JCudaTensor x297 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
// V_cv3_B
static JCudaTensor x264 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv3_W
static JCudaTensor x268 = JTensor.constFloat(0.0f, 512, 256, 3, 3).asJCudaTensor();
// V_cv4_B
static JCudaTensor x246 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_cv4_W
static JCudaTensor x241 = JTensor.constFloat(0.0f, 1024, 512, 3, 3).asJCudaTensor();
// V_cv5_B
static JCudaTensor x223 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_cv5_W
static JCudaTensor x218 = JTensor.constFloat(0.0f, 1024, 1024, 3, 3).asJCudaTensor();
// V_fc6_B
static JCudaTensor x188 = JTensor.constFloat(0.0f, 3072).asJCudaTensor();
// V_fc6_W
static JCudaTensor x192 = JTensor.constFloat(0.0f, 3072, 36864).asJCudaTensor();
// V_fc7_B
static JCudaTensor x170 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc7_W
static JCudaTensor x162 = JTensor.constFloat(0.0f, 4096, 3072).asJCudaTensor();
// V_fc8_B
static JCudaTensor x149 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc8_W
static JCudaTensor x139 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// cv1_B
static JCudaTensor x16 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
// cv1_W
static JCudaTensor x15 = JTensor.randomFloat(-0.07422696f, 0.07422696f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
// cv2_B
static JCudaTensor x32 = JTensor.constFloat(0.2f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x31 = JTensor.randomFloat(-0.028867513f, 0.028867513f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
// cv3_B
static JCudaTensor x45 = JTensor.constFloat(0.2f, 512).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x44 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv4_B
static JCudaTensor x55 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv4_B").asJCudaTensor();
// cv4_W
static JCudaTensor x54 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 1024, 512, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
// cv5_B
static JCudaTensor x65 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv5_B").asJCudaTensor();
// cv5_W
static JCudaTensor x64 = JTensor.randomFloat(-0.014731391f, 0.014731391f, 1024, 1024, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
// fc6_B
static JCudaTensor x82 = JTensor.constFloat(0.0f, 3072).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x78 = JTensor.randomFloat(-0.0073656957f, 0.0073656957f, 3072, 36864).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x92 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x88 = JTensor.randomFloat(-0.02551552f, 0.02551552f, 4096, 3072).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x102 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x98 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
test();
x16.save(network_dir + "/cv1_B");
x15.save(network_dir + "/cv1_W");
x32.save(network_dir + "/cv2_B");
x31.save(network_dir + "/cv2_W");
x45.save(network_dir + "/cv3_B");
x44.save(network_dir + "/cv3_W");
x55.save(network_dir + "/cv4_B");
x54.save(network_dir + "/cv4_W");
x65.save(network_dir + "/cv5_B");
x64.save(network_dir + "/cv5_W");
x82.save(network_dir + "/fc6_B");
x78.save(network_dir + "/fc6_W");
x92.save(network_dir + "/fc7_B");
x88.save(network_dir + "/fc7_W");
x102.save(network_dir + "/fc8_B");
x98.save(network_dir + "/fc8_W");
x15.free();
x293.free();
x241.free();
x64.free();
x98.free();
x188.free();
x319.free();
x297.free();
x170.free();
x54.free();
x65.free();
x192.free();
x223.free();
x45.free();
x78.free();
x55.free();
x88.free();
x92.free();
x82.free();
x32.free();
x268.free();
x139.free();
x264.free();
x149.free();
x325.free();
x44.free();
x16.free();
x246.free();
x162.free();
x102.free();
x31.free();
x218.free();
x39.free();
x59.free();
x49.free();
x36.free();
x71.free();
x26.free();
x23.free();
x46.free();
x17.free();
x66.free();
x105.free();
x33.free();
x56.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X34 = Cuda(Indicator(Y, 1000))
JCudaTensor x7;
JTensorFloat x8;
x8 = x4.asIndicator(1000);
x7 = x8.asJCudaTensor();

// val X10 = Cuda(X)
JCudaTensor x9;
JTensorFloat x10;
x10 = x3;
x9 = x10.asJCudaTensor();

// val X11 = Convolv(4,0)(X10,cv1_W,cv1_B)
JCudaTensor x11;
JCudaTensor x12, x13, x14;
x12 = x9;
x13 = x15;
x14 = x16;
x11 = x17.forward(x12, x13, x14);

// val X96 = - X34.copy
JCudaTensor x18;
JCudaTensor x19;
float x20;
x19 = x7;
x19 = x19.clone();
x20 = -1;
x18 = x19.times_i(x20);

// val X12 = ReLU()(X11)
JCudaTensor x21;
JCudaTensor x22;
x22 = x11;
x21 = x23.forward(x22);

// val X13 = Pooling(2,2,0,true)(X12)
JCudaTensor x24;
JCudaTensor x25;
x25 = x21;
x24 = x26.forward(x25);

// val X14 = Convolv(1,2)(X13,cv2_W,cv2_B)
JCudaTensor x27;
JCudaTensor x28, x29, x30;
x28 = x24;
x29 = x31;
x30 = x32;
x27 = x33.forward(x28, x29, x30);

// val X15 = ReLU()(X14)
JCudaTensor x34;
JCudaTensor x35;
x35 = x27;
x34 = x36.forward(x35);

// val X16 = Pooling(2,2,0,true)(X15)
JCudaTensor x37;
JCudaTensor x38;
x38 = x34;
x37 = x39.forward(x38);

// val X17 = Convolv(1,1)(X16,cv3_W,cv3_B)
JCudaTensor x40;
JCudaTensor x41, x42, x43;
x41 = x37;
x42 = x44;
x43 = x45;
x40 = x46.forward(x41, x42, x43);

// val X18 = ReLU()(X17)
JCudaTensor x47;
JCudaTensor x48;
x48 = x40;
x47 = x49.forward(x48);

// val X19 = Convolv(1,1)(X18,cv4_W,cv4_B)
JCudaTensor x50;
JCudaTensor x51, x52, x53;
x51 = x47;
x52 = x54;
x53 = x55;
x50 = x56.forward(x51, x52, x53);

// val X20 = ReLU()(X19)
JCudaTensor x57;
JCudaTensor x58;
x58 = x50;
x57 = x59.forward(x58);

// val X21 = Convolv(1,1)(X20,cv5_W,cv5_B)
JCudaTensor x60;
JCudaTensor x61, x62, x63;
x61 = x57;
x62 = x64;
x63 = x65;
x60 = x66.forward(x61, x62, x63);

// val X22 = ReLU()(X21)
JCudaTensor x67;
JCudaTensor x68;
x68 = x60;
x67 = x59.forward(x68);

// val X23 = Pooling(2,2,0,true)(X22)
JCudaTensor x69;
JCudaTensor x70;
x70 = x67;
x69 = x71.forward(x70);

// val X24 = (X23[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x72;
JCudaMatrix x73;
JCudaMatrix x74;
JCudaTensor x75;
JCudaTensor x76;
x76 = x69;
x75 = x76.flatten(1, new int[]{1024, 6, 6});
x73 = x75.asMatrix(1, true);
JCudaTensor x77;
x77 = x78;
x74 = x77.asMatrix(1, true);
x72 = x73.times(x74);

// val X26 = (X24 + (i) => fc6_B)
JCudaTensor x79;
JCudaTensor x80, x81;
x80 = x72;
x81 = x82;
x79 = x81.copy(128, x80);

// val X27 = (X26)(i | @) * (fc7_W)(j | @)
JCudaTensor x83;
JCudaMatrix x84;
JCudaMatrix x85;
JCudaTensor x86;
x86 = x79;
x84 = x86.asMatrix(1, true);
JCudaTensor x87;
x87 = x88;
x85 = x87.asMatrix(1, true);
x83 = x84.times(x85);

// val X29 = (X27 + (i) => fc7_B)
JCudaTensor x89;
JCudaTensor x90, x91;
x90 = x83;
x91 = x92;
x89 = x91.copy(128, x90);

// val X30 = (X29)(i | @) * (fc8_W)(j | @)
JCudaTensor x93;
JCudaMatrix x94;
JCudaMatrix x95;
JCudaTensor x96;
x96 = x89;
x94 = x96.asMatrix(1, true);
JCudaTensor x97;
x97 = x98;
x95 = x97.asMatrix(1, true);
x93 = x94.times(x95);

// val X32 = (X30 + (i) => fc8_B)
JCudaTensor x99;
JCudaTensor x100, x101;
x100 = x93;
x101 = x102;
x99 = x101.copy(128, x100);

// val X33 = LogSoftmax()(X32)
JCudaTensor x103;
JCudaTensor x104;
x104 = x99;
x103 = x105.forward(x104);

// Dealloc(X32)
JCudaTensor x106;
x106 = x99;
x106.free();

// val X97 = (X96 / |128|)
JCudaTensor x107;
JCudaTensor x108;
float x109;
x108 = x18;
float x110;
x110 = 128;
x109 = 1 / x110;
x107 = x108.times_i(x109);

// Print(((0 - (X34 . X33)) / |128|))
float x111;
float x112;
float x113;
float x114;
JCudaTensor x115, x116;
x115 = x7;
x116 = x103;
x114 = x115.dot(x116);
x112 = - x114;
x113 = 128;
x111 = x112 / x113;
System.out.println(x5 + " " + x111);
if (Float.isNaN(x111)) { System.exit(-1); }

// Dealloc(X34)
JCudaTensor x117;
x117 = x7;
x117.free();

// val X122 = X97 * d_LogSoftmax()(X33)/d_X32
JCudaTensor x118;
JCudaTensor x119, x120;
x119 = x107;
x120 = x103;
x118 = x105.backward(x119, x120);

// Dealloc(X97)
JCudaTensor x121;
x121 = x107;
x121.free();

// Dealloc(X33)
JCudaTensor x122;
x122 = x103;
x122.free();

// val m1 = (i21) => fc8_W[@, i21]
JCudaMatrix x123;
JCudaTensor x124;
x124 = x98;
x123 = x124.asMatrix(1, false);

// val X142 = (X122)(i20 | @) * m1
JCudaTensor x125;
JCudaMatrix x126;
JCudaMatrix x127;
JCudaTensor x128;
x128 = x118;
x126 = x128.asMatrix(1, true);
x127 = x123;
x125 = x126.times(x127);

// val m43 = (i490) => X122[@, i490]
JCudaMatrix x129;
JCudaTensor x130;
x130 = x118;
x129 = x130.asMatrix(1, false);

// val m45 = (i498) => X29[@, i498]
JCudaMatrix x131;
JCudaTensor x132;
x132 = x89;
x131 = x132.asMatrix(1, false);

// val m2 = (i25) => fc7_W[@, i25]
JCudaMatrix x133;
JCudaTensor x134;
x134 = x88;
x133 = x134.asMatrix(1, false);

// val m42 = (i471) => X26[@, i471]
JCudaMatrix x135;
JCudaTensor x136;
x136 = x79;
x135 = x136.asMatrix(1, false);

// val m39 = (i453) => X142[@, i453]
JCudaMatrix x137;
JCudaTensor x138;
x138 = x125;
x137 = x138.asMatrix(1, false);

// V_fc8_W <~~ m43 * m45
float x140, x141;
x140 = lrn_rate_1;
x141 = momentum;
JCudaMatrix x142;
JCudaMatrix x143;
x142 = x129;
x143 = x131;
x142.times(x143, x139, x140, x141);

// Dealloc(X29)
JCudaTensor x144;
x144 = x89;
x144.free();

// val X143 = (X142)(i24 | @) * m2
JCudaTensor x145;
JCudaMatrix x146;
JCudaMatrix x147;
JCudaTensor x148;
x148 = x125;
x146 = x148.asMatrix(1, true);
x147 = x133;
x145 = x146.times(x147);

// V_fc8_B <~~ Sum(m43)
float x150, x151;
x150 = lrn_rate_1;
x151 = momentum;
JCudaMatrix x152;
x152 = x129;
x152.sum(x149, x150, x151);

// Dealloc(X122)
JCudaTensor x153;
x153 = x118;
x153.free();

// val m3 = (i29) => fc6_W[@, i29]
JCudaMatrix x154;
JCudaTensor x155;
x155 = x78;
x154 = x155.asMatrix(1, false);

// fc8_B <~~ V_fc8_B
float x156, x157;
x156 = 1;
x157 = decay_1;
JCudaTensor x158;
x158 = x149;
x102.update(x158, x156, x157);

// fc8_W <~~ V_fc8_W
float x159, x160;
x159 = 1;
x160 = decay_1;
JCudaTensor x161;
x161 = x139;
x98.update(x161, x159, x160);

// V_fc7_W <~~ m39 * m42
float x163, x164;
x163 = lrn_rate_1;
x164 = momentum;
JCudaMatrix x165;
JCudaMatrix x166;
x165 = x137;
x166 = x135;
x165.times(x166, x162, x163, x164);

// Dealloc(X26)
JCudaTensor x167;
x167 = x79;
x167.free();

// val m33 = (i396) => X143[@, i396]
JCudaMatrix x168;
JCudaTensor x169;
x169 = x145;
x168 = x169.asMatrix(1, false);

// V_fc7_B <~~ Sum(m39)
float x171, x172;
x171 = lrn_rate_1;
x172 = momentum;
JCudaMatrix x173;
x173 = x137;
x173.sum(x170, x171, x172);

// Dealloc(X142)
JCudaTensor x174;
x174 = x125;
x174.free();

// val m37 = (i424) => X23[1><3][@, i424]
JCudaMatrix x175;
JCudaTensor x176;
JCudaTensor x177;
x177 = x69;
x176 = x177.flatten(1, new int[]{1024, 6, 6});
x175 = x176.asMatrix(1, false);

// val X144 = (X143)(i28 | @) * m3
JCudaTensor x178;
JCudaMatrix x179;
JCudaMatrix x180;
JCudaTensor x181;
x181 = x145;
x179 = x181.asMatrix(1, true);
x180 = x154;
x178 = x179.times(x180);

// fc7_B <~~ V_fc7_B
float x182, x183;
x182 = 1;
x183 = decay_1;
JCudaTensor x184;
x184 = x170;
x92.update(x184, x182, x183);

// fc7_W <~~ V_fc7_W
float x185, x186;
x185 = 1;
x186 = decay_1;
JCudaTensor x187;
x187 = x162;
x88.update(x187, x185, x186);

// V_fc6_B <~~ Sum(m33)
float x189, x190;
x189 = lrn_rate_1;
x190 = momentum;
JCudaMatrix x191;
x191 = x168;
x191.sum(x188, x189, x190);

// V_fc6_W <~~ m33 * m37
float x193, x194;
x193 = lrn_rate_1;
x194 = momentum;
JCudaMatrix x195;
JCudaMatrix x196;
x195 = x168;
x196 = x175;
x195.times(x196, x192, x193, x194);

// Dealloc(X143)
JCudaTensor x197;
x197 = x145;
x197.free();

// val X146 = X144[1<>3] * d_Pooling(2,2,0,true)(X23,X22)/d_X22
JCudaTensor x198;
JCudaTensor x199, x200, x201;
JCudaTensor x202;
x202 = x178;
x199 = x202.unflatten(1, new int[]{1024, 6, 6});
x200 = x69;
x201 = x67;
x198 = x71.backward(x199, x200, x201);

// Dealloc(X144)
JCudaTensor x203;
x203 = x178;
x203.free();

// Dealloc(X23)
JCudaTensor x204;
x204 = x69;
x204.free();

// fc6_B <~~ V_fc6_B
float x205, x206;
x205 = 1;
x206 = decay_1;
JCudaTensor x207;
x207 = x188;
x82.update(x207, x205, x206);

// fc6_W <~~ V_fc6_W
float x208, x209;
x208 = 1;
x209 = decay_1;
JCudaTensor x210;
x210 = x192;
x78.update(x210, x208, x209);

// val X148 = X146 * d_ReLU()(X22)/d_X21
JCudaTensor x211;
JCudaTensor x212, x213;
x212 = x198;
x213 = x67;
x211 = x59.backward(x212, x213);

// Dealloc(X22)
JCudaTensor x214;
x214 = x67;
x214.free();

// val X149 = X148 * d_Convolv(1,1)(cv5_W)/d_X20
JCudaTensor x215;
JCudaTensor x216, x217;
x216 = x211;
x217 = x64;
x215 = x66.backward_data(x216, x217);

// V_cv5_W <~~ X148 * d_Convolv(1,1)(X20)/d_cv5_W
float x219, x220;
x219 = lrn_rate_1;
x220 = momentum;
JCudaTensor x221, x222;
x221 = x211;
x222 = x57;
x66.backward_filter(x221, x222, x218, x219, x220);

// V_cv5_B <~~ X148 * d_Convolv(1,1)()/d_cv5_B
float x224, x225;
x224 = lrn_rate_2;
x225 = momentum;
JCudaTensor x226;
x226 = x211;
x66.backward_bias(x226, x223, x224, x225);

// Dealloc(X148)
JCudaTensor x227;
x227 = x211;
x227.free();

// cv5_W <~~ V_cv5_W
float x228, x229;
x228 = 1;
x229 = decay_1;
JCudaTensor x230;
x230 = x218;
x64.update(x230, x228, x229);

// cv5_B <~~ V_cv5_B
float x231, x232;
x231 = 1;
x232 = 1;
JCudaTensor x233;
x233 = x223;
x65.update(x233, x231, x232);

// val X151 = X149 * d_ReLU()(X20)/d_X19
JCudaTensor x234;
JCudaTensor x235, x236;
x235 = x215;
x236 = x57;
x234 = x59.backward(x235, x236);

// Dealloc(X20)
JCudaTensor x237;
x237 = x57;
x237.free();

// val X152 = X151 * d_Convolv(1,1)(cv4_W)/d_X18
JCudaTensor x238;
JCudaTensor x239, x240;
x239 = x234;
x240 = x54;
x238 = x56.backward_data(x239, x240);

// V_cv4_W <~~ X151 * d_Convolv(1,1)(X18)/d_cv4_W
float x242, x243;
x242 = lrn_rate_1;
x243 = momentum;
JCudaTensor x244, x245;
x244 = x234;
x245 = x47;
x56.backward_filter(x244, x245, x241, x242, x243);

// V_cv4_B <~~ X151 * d_Convolv(1,1)()/d_cv4_B
float x247, x248;
x247 = lrn_rate_2;
x248 = momentum;
JCudaTensor x249;
x249 = x234;
x56.backward_bias(x249, x246, x247, x248);

// Dealloc(X151)
JCudaTensor x250;
x250 = x234;
x250.free();

// cv4_W <~~ V_cv4_W
float x251, x252;
x251 = 1;
x252 = decay_1;
JCudaTensor x253;
x253 = x241;
x54.update(x253, x251, x252);

// cv4_B <~~ V_cv4_B
float x254, x255;
x254 = 1;
x255 = 1;
JCudaTensor x256;
x256 = x246;
x55.update(x256, x254, x255);

// val X154 = X152 * d_ReLU()(X18)/d_X17
JCudaTensor x257;
JCudaTensor x258, x259;
x258 = x238;
x259 = x47;
x257 = x49.backward(x258, x259);

// Dealloc(X18)
JCudaTensor x260;
x260 = x47;
x260.free();

// val X155 = X154 * d_Convolv(1,1)(cv3_W)/d_X16
JCudaTensor x261;
JCudaTensor x262, x263;
x262 = x257;
x263 = x44;
x261 = x46.backward_data(x262, x263);

// V_cv3_B <~~ X154 * d_Convolv(1,1)()/d_cv3_B
float x265, x266;
x265 = lrn_rate_2;
x266 = momentum;
JCudaTensor x267;
x267 = x257;
x46.backward_bias(x267, x264, x265, x266);

// V_cv3_W <~~ X154 * d_Convolv(1,1)(X16)/d_cv3_W
float x269, x270;
x269 = lrn_rate_1;
x270 = momentum;
JCudaTensor x271, x272;
x271 = x257;
x272 = x37;
x46.backward_filter(x271, x272, x268, x269, x270);

// Dealloc(X154)
JCudaTensor x273;
x273 = x257;
x273.free();

// cv3_B <~~ V_cv3_B
float x274, x275;
x274 = 1;
x275 = 1;
JCudaTensor x276;
x276 = x264;
x45.update(x276, x274, x275);

// cv3_W <~~ V_cv3_W
float x277, x278;
x277 = 1;
x278 = decay_1;
JCudaTensor x279;
x279 = x268;
x44.update(x279, x277, x278);

// val X157 = X155 * d_Pooling(2,2,0,true)(X16,X15)/d_X15
JCudaTensor x280;
JCudaTensor x281, x282, x283;
x281 = x261;
x282 = x37;
x283 = x34;
x280 = x39.backward(x281, x282, x283);

// Dealloc(X155)
JCudaTensor x284;
x284 = x261;
x284.free();

// Dealloc(X16)
JCudaTensor x285;
x285 = x37;
x285.free();

// val X159 = X157 * d_ReLU()(X15)/d_X14
JCudaTensor x286;
JCudaTensor x287, x288;
x287 = x280;
x288 = x34;
x286 = x36.backward(x287, x288);

// Dealloc(X15)
JCudaTensor x289;
x289 = x34;
x289.free();

// val X160 = X159 * d_Convolv(1,2)(cv2_W)/d_X13
JCudaTensor x290;
JCudaTensor x291, x292;
x291 = x286;
x292 = x31;
x290 = x33.backward_data(x291, x292);

// V_cv2_B <~~ X159 * d_Convolv(1,2)()/d_cv2_B
float x294, x295;
x294 = lrn_rate_2;
x295 = momentum;
JCudaTensor x296;
x296 = x286;
x33.backward_bias(x296, x293, x294, x295);

// V_cv2_W <~~ X159 * d_Convolv(1,2)(X13)/d_cv2_W
float x298, x299;
x298 = lrn_rate_1;
x299 = momentum;
JCudaTensor x300, x301;
x300 = x286;
x301 = x24;
x33.backward_filter(x300, x301, x297, x298, x299);

// Dealloc(X159)
JCudaTensor x302;
x302 = x286;
x302.free();

// cv2_B <~~ V_cv2_B
float x303, x304;
x303 = 1;
x304 = 1;
JCudaTensor x305;
x305 = x293;
x32.update(x305, x303, x304);

// cv2_W <~~ V_cv2_W
float x306, x307;
x306 = 1;
x307 = decay_1;
JCudaTensor x308;
x308 = x297;
x31.update(x308, x306, x307);

// val X162 = X160 * d_Pooling(2,2,0,true)(X13,X12)/d_X12
JCudaTensor x309;
JCudaTensor x310, x311, x312;
x310 = x290;
x311 = x24;
x312 = x21;
x309 = x26.backward(x310, x311, x312);

// Dealloc(X160)
JCudaTensor x313;
x313 = x290;
x313.free();

// Dealloc(X13)
JCudaTensor x314;
x314 = x24;
x314.free();

// val X164 = X162 * d_ReLU()(X12)/d_X11
JCudaTensor x315;
JCudaTensor x316, x317;
x316 = x309;
x317 = x21;
x315 = x23.backward(x316, x317);

// Dealloc(X12)
JCudaTensor x318;
x318 = x21;
x318.free();

// V_cv1_W <~~ X164 * d_Convolv(4,0)(X10)/d_cv1_W
float x320, x321;
x320 = lrn_rate_1;
x321 = momentum;
JCudaTensor x322, x323;
x322 = x315;
x323 = x9;
x17.backward_filter(x322, x323, x319, x320, x321);

// Dealloc(X10)
JCudaTensor x324;
x324 = x9;
x324.free();

// V_cv1_B <~~ X164 * d_Convolv(4,0)()/d_cv1_B
float x326, x327;
x326 = lrn_rate_2;
x327 = momentum;
JCudaTensor x328;
x328 = x315;
x17.backward_bias(x328, x325, x326, x327);

// Dealloc(X164)
JCudaTensor x329;
x329 = x315;
x329.free();

// cv1_W <~~ V_cv1_W
float x330, x331;
x330 = 1;
x331 = decay_1;
JCudaTensor x332;
x332 = x319;
x15.update(x332, x330, x331);

// cv1_B <~~ V_cv1_B
float x333, x334;
x333 = 1;
x334 = 1;
JCudaTensor x335;
x335 = x325;
x16.update(x335, x333, x334);

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X1767 = Cuda(X)
JCudaTensor x336;
JTensorFloat x337;
x337 = x3;
x336 = x337.asJCudaTensor();

// val X1768 = Convolv(4,0)(X1767,cv1_W,cv1_B)
JCudaTensor x338;
JCudaTensor x339, x340, x341;
x339 = x336;
x340 = x15;
x341 = x16;
x338 = x17.forward(x339, x340, x341);

// Dealloc(X1767)
JCudaTensor x342;
x342 = x336;
x342.free();

// val X1769 = ReLU()(X1768)
JCudaTensor x343;
JCudaTensor x344;
x344 = x338;
x343 = x23.forward(x344);

// val X1770 = Pooling(2,2,0,true)(X1769)
JCudaTensor x345;
JCudaTensor x346;
x346 = x343;
x345 = x26.forward(x346);

// Dealloc(X1769)
JCudaTensor x347;
x347 = x343;
x347.free();

// val X1771 = Convolv(1,2)(X1770,cv2_W,cv2_B)
JCudaTensor x348;
JCudaTensor x349, x350, x351;
x349 = x345;
x350 = x31;
x351 = x32;
x348 = x33.forward(x349, x350, x351);

// Dealloc(X1770)
JCudaTensor x352;
x352 = x345;
x352.free();

// val X1772 = ReLU()(X1771)
JCudaTensor x353;
JCudaTensor x354;
x354 = x348;
x353 = x36.forward(x354);

// val X1773 = Pooling(2,2,0,true)(X1772)
JCudaTensor x355;
JCudaTensor x356;
x356 = x353;
x355 = x39.forward(x356);

// Dealloc(X1772)
JCudaTensor x357;
x357 = x353;
x357.free();

// val X1774 = Convolv(1,1)(X1773,cv3_W,cv3_B)
JCudaTensor x358;
JCudaTensor x359, x360, x361;
x359 = x355;
x360 = x44;
x361 = x45;
x358 = x46.forward(x359, x360, x361);

// Dealloc(X1773)
JCudaTensor x362;
x362 = x355;
x362.free();

// val X1775 = ReLU()(X1774)
JCudaTensor x363;
JCudaTensor x364;
x364 = x358;
x363 = x49.forward(x364);

// val X1776 = Convolv(1,1)(X1775,cv4_W,cv4_B)
JCudaTensor x365;
JCudaTensor x366, x367, x368;
x366 = x363;
x367 = x54;
x368 = x55;
x365 = x56.forward(x366, x367, x368);

// Dealloc(X1775)
JCudaTensor x369;
x369 = x363;
x369.free();

// val X1777 = ReLU()(X1776)
JCudaTensor x370;
JCudaTensor x371;
x371 = x365;
x370 = x59.forward(x371);

// val X1778 = Convolv(1,1)(X1777,cv5_W,cv5_B)
JCudaTensor x372;
JCudaTensor x373, x374, x375;
x373 = x370;
x374 = x64;
x375 = x65;
x372 = x66.forward(x373, x374, x375);

// Dealloc(X1777)
JCudaTensor x376;
x376 = x370;
x376.free();

// val X1779 = ReLU()(X1778)
JCudaTensor x377;
JCudaTensor x378;
x378 = x372;
x377 = x59.forward(x378);

// val X1780 = Pooling(2,2,0,true)(X1779)
JCudaTensor x379;
JCudaTensor x380;
x380 = x377;
x379 = x71.forward(x380);

// Dealloc(X1779)
JCudaTensor x381;
x381 = x377;
x381.free();

// val X1781 = (X1780[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x382;
JCudaMatrix x383;
JCudaMatrix x384;
JCudaTensor x385;
JCudaTensor x386;
x386 = x379;
x385 = x386.flatten(1, new int[]{1024, 6, 6});
x383 = x385.asMatrix(1, true);
JCudaTensor x387;
x387 = x78;
x384 = x387.asMatrix(1, true);
x382 = x383.times(x384);

// Dealloc(X1780)
JCudaTensor x388;
x388 = x379;
x388.free();

// val X1783 = (X1781 + (i) => fc6_B)
JCudaTensor x389;
JCudaTensor x390, x391;
x390 = x382;
x391 = x82;
x389 = x391.copy(128, x390);

// val X1784 = (X1783)(i | @) * (fc7_W)(j | @)
JCudaTensor x392;
JCudaMatrix x393;
JCudaMatrix x394;
JCudaTensor x395;
x395 = x389;
x393 = x395.asMatrix(1, true);
JCudaTensor x396;
x396 = x88;
x394 = x396.asMatrix(1, true);
x392 = x393.times(x394);

// Dealloc(X1783)
JCudaTensor x397;
x397 = x389;
x397.free();

// val X1786 = (X1784 + (i) => fc7_B)
JCudaTensor x398;
JCudaTensor x399, x400;
x399 = x392;
x400 = x92;
x398 = x400.copy(128, x399);

// val X1787 = (X1786)(i | @) * (fc8_W)(j | @)
JCudaTensor x401;
JCudaMatrix x402;
JCudaMatrix x403;
JCudaTensor x404;
x404 = x398;
x402 = x404.asMatrix(1, true);
JCudaTensor x405;
x405 = x98;
x403 = x405.asMatrix(1, true);
x401 = x402.times(x403);

// Dealloc(X1786)
JCudaTensor x406;
x406 = x398;
x406.free();

// val X1789 = (X1787 + (i) => fc8_B)
JCudaTensor x407;
JCudaTensor x408, x409;
x408 = x401;
x409 = x102;
x407 = x409.copy(128, x408);

// val X1790 = Cuda(Indicator(Y, 1000))
JCudaTensor x410;
JTensorFloat x411;
x411 = x4.asIndicator(1000);
x410 = x411.asJCudaTensor();

// val X1791 = X1790 .* X1789
JCudaTensor x412;
JCudaTensor x413, x414;
x413 = x410;
x414 = x407;
x412 = x413.times_i(x414);

// val X1792 = Sum((X1791)(i15 | @))
JCudaTensor x415;
JCudaMatrix x416;
JCudaTensor x417;
x417 = x412;
x416 = x417.asMatrix(1, true);
x415 = x416.sum();

// Dealloc(X1791)
JCudaTensor x418;
x418 = x412;
x418.free();

// val X1793 = Max((X1789)(i15 | @))
JCudaTensor x419;
JCudaMatrix x420;
JCudaTensor x421;
x421 = x407;
x420 = x421.asMatrix(1, true);
x419 = x420.max();

// Dealloc(X1789)
JCudaTensor x422;
x422 = x407;
x422.free();

// val X1794 = 1{X1792 == X1793}
JCudaTensor x423;
JCudaTensor x424, x425;
x424 = x415;
x425 = x419;
x423 = x424.eq(x425);

// Dealloc(X1793)
JCudaTensor x426;
x426 = x419;
x426.free();

// BatchSum(((Sum(X1794) / |128|) / 10))
float x428;
float x429;
float x430;
float x431;
float x432;
JCudaTensor x433;
x433 = x423;
x431 = x433.sum();
x432 = 128;
x429 = x431 / x432;
x430 = 10;
x428 = x429 / x430;
x427 += x428;
// Print((Sum(X1794) / |128|))
float x434;
float x435;
float x436;
JCudaTensor x437;
x437 = x423;
x435 = x437.sum();
x436 = 128;
x434 = x435 / x436;
System.out.println(x5 + " test precision "  + x434);

// Dealloc(X1794)
JCudaTensor x438;
x438 = x423;
x438.free();

}
System.out.println(x427); 
}

}