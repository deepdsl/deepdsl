package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.tensor.*;
import deepdsl.util.*;


public class Overfeat {
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
// test_data_path
static String test_data_path = "dataset/imagenet224/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 40;
// train_data_path
static String train_data_path = "dataset/imagenet224/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 400;

// (Convolv(1,1),List(128, 1024, 13, 13))
static JCudnnConvolution x66 = new JCudnnConvolution(new int[]{128,1024,13,13},new int[]{1024,1024,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,1),List(128, 1024, 13, 13))
static JCudnnConvolution x56 = new JCudnnConvolution(new int[]{128,512,13,13},new int[]{1024,512,3,3},new int[]{1024}, 1, 1);
// (Convolv(1,1),List(128, 512, 13, 13))
static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{512,256,3,3},new int[]{512}, 1, 1);
// (Convolv(1,2),List(128, 256, 27, 27))
static JCudnnConvolution x33 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
// (Convolv(4,0),List(128, 96, 54, 54))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 0);
// (LMDB,false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, 640, new int[]{128, 3, 224, 224});
// (LMDB,true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, 6400, new int[]{128, 3, 224, 224});
// (LogSoftmax(),List(128, 1000))
static JCudnnSoftmax x106 = new JCudnnSoftmax(new int[]{128,1000}, 2);
// (Pooling(2,2,0,true),List(128, 1024, 6, 6))
static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,1024,13,13}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(128, 256, 13, 13))
static JCudnnPooling x39 = new JCudnnPooling(new int[]{128,256,27,27}, 2, 2, 0, 0);
// (Pooling(2,2,0,true),List(128, 96, 27, 27))
static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,96,54,54}, 2, 2, 0, 0);
// (ReLU(),List(128, 1024, 13, 13))
static JCudnnActivation x69 = new JCudnnActivation(new int[]{128,1024,13,13}, 1);
// (ReLU(),List(128, 1024, 13, 13))
static JCudnnActivation x59 = new JCudnnActivation(new int[]{128,1024,13,13}, 1);
// (ReLU(),List(128, 256, 27, 27))
static JCudnnActivation x36 = new JCudnnActivation(new int[]{128,256,27,27}, 1);
// (ReLU(),List(128, 512, 13, 13))
static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,512,13,13}, 1);
// (ReLU(),List(128, 96, 54, 54))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,96,54,54}, 1);
// V_cv1_B
static JCudaTensor x326 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv1_W
static JCudaTensor x320 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
// V_cv2_B
static JCudaTensor x294 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// V_cv2_W
static JCudaTensor x298 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
// V_cv3_B
static JCudaTensor x265 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// V_cv3_W
static JCudaTensor x269 = JTensor.constFloat(0.0f, 512, 256, 3, 3).asJCudaTensor();
// V_cv4_B
static JCudaTensor x247 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_cv4_W
static JCudaTensor x242 = JTensor.constFloat(0.0f, 1024, 512, 3, 3).asJCudaTensor();
// V_cv5_B
static JCudaTensor x224 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_cv5_W
static JCudaTensor x219 = JTensor.constFloat(0.0f, 1024, 1024, 3, 3).asJCudaTensor();
// V_fc6_B
static JCudaTensor x189 = JTensor.constFloat(0.0f, 3072).asJCudaTensor();
// V_fc6_W
static JCudaTensor x193 = JTensor.constFloat(0.0f, 3072, 36864).asJCudaTensor();
// V_fc7_B
static JCudaTensor x178 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
// V_fc7_W
static JCudaTensor x163 = JTensor.constFloat(0.0f, 4096, 3072).asJCudaTensor();
// V_fc8_B
static JCudaTensor x140 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc8_W
static JCudaTensor x144 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
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
static JCudaTensor x83 = JTensor.constFloat(0.0f, 3072).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x79 = JTensor.randomFloat(-0.0073656957f, 0.0073656957f, 3072, 36864).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x93 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x89 = JTensor.randomFloat(-0.02551552f, 0.02551552f, 4096, 3072).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x103 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x99 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
ArithStats.isStats = false;
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
System.out.println(ArithStats.outStats());
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
x83.save(network_dir + "/fc6_B");
x79.save(network_dir + "/fc6_W");
x93.save(network_dir + "/fc7_B");
x89.save(network_dir + "/fc7_W");
x103.save(network_dir + "/fc8_B");
x99.save(network_dir + "/fc8_W");
x15.free();
x178.free();
x242.free();
x140.free();
x219.free();
x269.free();
x103.free();
x247.free();
x224.free();
x64.free();
x144.free();
x163.free();
x294.free();
x54.free();
x99.free();
x65.free();
x89.free();
x45.free();
x55.free();
x83.free();
x32.free();
x79.free();
x326.free();
x93.free();
x298.free();
x44.free();
x193.free();
x16.free();
x265.free();
x189.free();
x31.free();
x320.free();
x39.free();
x106.free();
x59.free();
x49.free();
x36.free();
x26.free();
x23.free();
x46.free();
x17.free();
x69.free();
x72.free();
x66.free();
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
x11 = x17.forward(x12,x13,x14);

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
x27 = x33.forward(x28,x29,x30);

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
x40 = x46.forward(x41,x42,x43);

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
x50 = x56.forward(x51,x52,x53);

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
x60 = x66.forward(x61,x62,x63);

// val X22 = ReLU()(X21)
JCudaTensor x67;
JCudaTensor x68;
x68 = x60;
x67 = x69.forward(x68);

// val X23 = Pooling(2,2,0,true)(X22)
JCudaTensor x70;
JCudaTensor x71;
x71 = x67;
x70 = x72.forward(x71);

// val X24 = (X23[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x73;
JCudaMatrix x74;
JCudaMatrix x75;
JCudaTensor x76;
JCudaTensor x77;
x77 = x70;
x76 = x77.flatten(1, new int[]{1024, 6, 6});
x74 = x76.asMatrix(1, true);
JCudaTensor x78;
x78 = x79;
x75 = x78.asMatrix(1, true);
x73 = x74.times(x75);

// val X26 = (X24 + (i) => fc6_B)
JCudaTensor x80;
JCudaTensor x81, x82;
x81 = x73;
x82 = x83;
x80 = x82.copy(128, x81);

// val X27 = (X26)(i | @) * (fc7_W)(j | @)
JCudaTensor x84;
JCudaMatrix x85;
JCudaMatrix x86;
JCudaTensor x87;
x87 = x80;
x85 = x87.asMatrix(1, true);
JCudaTensor x88;
x88 = x89;
x86 = x88.asMatrix(1, true);
x84 = x85.times(x86);

// val X29 = (X27 + (i) => fc7_B)
JCudaTensor x90;
JCudaTensor x91, x92;
x91 = x84;
x92 = x93;
x90 = x92.copy(128, x91);

// val X30 = (X29)(i | @) * (fc8_W)(j | @)
JCudaTensor x94;
JCudaMatrix x95;
JCudaMatrix x96;
JCudaTensor x97;
x97 = x90;
x95 = x97.asMatrix(1, true);
JCudaTensor x98;
x98 = x99;
x96 = x98.asMatrix(1, true);
x94 = x95.times(x96);

// val X32 = (X30 + (i) => fc8_B)
JCudaTensor x100;
JCudaTensor x101, x102;
x101 = x94;
x102 = x103;
x100 = x102.copy(128, x101);

// val X33 = LogSoftmax()(X32)
JCudaTensor x104;
JCudaTensor x105;
x105 = x100;
x104 = x106.forward(x105);

// Dealloc(X32)
JCudaTensor x107;
x107 = x100;
x107.free();

// val X97 = (X96 / |128|)
JCudaTensor x108;
JCudaTensor x109;
float x110;
x109 = x18;
float x111;
x111 = 128;
x110 = 1 / x111;
x108 = x109.times_i(x110);

// Print(((0 - (X34 . X33)) / |128|))
float x112;
float x113;
float x114;
float x115;
JCudaTensor x116, x117;
x116 = x7;
x117 = x104;
x115 = x116.dot(x117);
x113 = - x115;
x114 = 128;
x112 = x113 / x114;
System.out.println(x5 + " " + x112);
if (Float.isNaN(x112)) { System.exit(-1); }

// Dealloc(X34)
JCudaTensor x118;
x118 = x7;
x118.free();

// val X122 = X97 * d_LogSoftmax()(X33)/d_X32
JCudaTensor x119;
JCudaTensor x120, x121;
x120 = x108;
x121 = x104;
x119 = x106.backward(x120,x121);

// Dealloc(X97)
JCudaTensor x122;
x122 = x108;
x122.free();

// Dealloc(X33)
JCudaTensor x123;
x123 = x104;
x123.free();

// val m1 = (i419) => fc8_W[@, i419]
JCudaMatrix x124;
JCudaTensor x125;
x125 = x99;
x124 = x125.asMatrix(1, false);

// val X142 = (X122)(i418 | @) * m1
JCudaTensor x126;
JCudaMatrix x127;
JCudaMatrix x128;
JCudaTensor x129;
x129 = x119;
x127 = x129.asMatrix(1, true);
x128 = x124;
x126 = x127.times(x128);

// val m2 = (i429) => fc7_W[@, i429]
JCudaMatrix x130;
JCudaTensor x131;
x131 = x89;
x130 = x131.asMatrix(1, false);

// val m43 = (i15916) => X122[@, i15916]
JCudaMatrix x132;
JCudaTensor x133;
x133 = x119;
x132 = x133.asMatrix(1, false);

// val m45 = (i16892) => X29[@, i16892]
JCudaMatrix x134;
JCudaTensor x135;
x135 = x90;
x134 = x135.asMatrix(1, false);

// val X143 = (X142)(i428 | @) * m2
JCudaTensor x136;
JCudaMatrix x137;
JCudaMatrix x138;
JCudaTensor x139;
x139 = x126;
x137 = x139.asMatrix(1, true);
x138 = x130;
x136 = x137.times(x138);

// V_fc8_B <~~ Sum(m43)
float x141, x142;
x141 = lrn_rate_1;
x142 = momentum;
JCudaMatrix x143;
x143 = x132;
x143.sum(x140, x141, x142);

// V_fc8_W <~~ m43 * m45
float x145, x146;
x145 = lrn_rate_1;
x146 = momentum;
JCudaMatrix x147;
JCudaMatrix x148;
x147 = x132;
x148 = x134;
x147.times(x148, x144, x145, x146);

// Dealloc(X122)
JCudaTensor x149;
x149 = x119;
x149.free();

// Dealloc(X29)
JCudaTensor x150;
x150 = x90;
x150.free();

// val m42 = (i14926) => X26[@, i14926]
JCudaMatrix x151;
JCudaTensor x152;
x152 = x80;
x151 = x152.asMatrix(1, false);

// val m3 = (i439) => fc6_W[@, i439]
JCudaMatrix x153;
JCudaTensor x154;
x154 = x79;
x153 = x154.asMatrix(1, false);

// val m39 = (i13933) => X142[@, i13933]
JCudaMatrix x155;
JCudaTensor x156;
x156 = x126;
x155 = x156.asMatrix(1, false);

// fc8_B <~~ V_fc8_B
float x157, x158;
x157 = 1;
x158 = decay_1;
JCudaTensor x159;
x159 = x140;
x103.update(x159, x157, x158);


// fc8_W <~~ V_fc8_W
float x160, x161;
x160 = 1;
x161 = decay_1;
JCudaTensor x162;
x162 = x144;
x99.update(x162, x160, x161);


// V_fc7_W <~~ m39 * m42
float x164, x165;
x164 = lrn_rate_1;
x165 = momentum;
JCudaMatrix x166;
JCudaMatrix x167;
x166 = x155;
x167 = x151;
x166.times(x167, x163, x164, x165);

// Dealloc(X26)
JCudaTensor x168;
x168 = x80;
x168.free();

// val m37 = (i12843) => X23[1><3][@, i12843]
JCudaMatrix x169;
JCudaTensor x170;
JCudaTensor x171;
x171 = x70;
x170 = x171.flatten(1, new int[]{1024, 6, 6});
x169 = x170.asMatrix(1, false);

// val X144 = (X143)(i438 | @) * m3
JCudaTensor x172;
JCudaMatrix x173;
JCudaMatrix x174;
JCudaTensor x175;
x175 = x136;
x173 = x175.asMatrix(1, true);
x174 = x153;
x172 = x173.times(x174);

// val m33 = (i11829) => X143[@, i11829]
JCudaMatrix x176;
JCudaTensor x177;
x177 = x136;
x176 = x177.asMatrix(1, false);

// V_fc7_B <~~ Sum(m39)
float x179, x180;
x179 = lrn_rate_1;
x180 = momentum;
JCudaMatrix x181;
x181 = x155;
x181.sum(x178, x179, x180);

// Dealloc(X142)
JCudaTensor x182;
x182 = x126;
x182.free();

// fc7_B <~~ V_fc7_B
float x183, x184;
x183 = 1;
x184 = decay_1;
JCudaTensor x185;
x185 = x178;
x93.update(x185, x183, x184);


// fc7_W <~~ V_fc7_W
float x186, x187;
x186 = 1;
x187 = decay_1;
JCudaTensor x188;
x188 = x163;
x89.update(x188, x186, x187);


// V_fc6_B <~~ Sum(m33)
float x190, x191;
x190 = lrn_rate_1;
x191 = momentum;
JCudaMatrix x192;
x192 = x176;
x192.sum(x189, x190, x191);

// V_fc6_W <~~ m33 * m37
float x194, x195;
x194 = lrn_rate_1;
x195 = momentum;
JCudaMatrix x196;
JCudaMatrix x197;
x196 = x176;
x197 = x169;
x196.times(x197, x193, x194, x195);

// Dealloc(X143)
JCudaTensor x198;
x198 = x136;
x198.free();

// val X146 = X144[1<>3] * d_Pooling(2,2,0,true)(X23,X22)/d_X22
JCudaTensor x199;
JCudaTensor x200, x201, x202;
JCudaTensor x203;
x203 = x172;
x200 = x203.unflatten(1, new int[]{1024, 6, 6});
x201 = x70;
x202 = x67;
x199 = x72.backward(x200,x201,x202);

// Dealloc(X144)
JCudaTensor x204;
x204 = x172;
x204.free();

// Dealloc(X23)
JCudaTensor x205;
x205 = x70;
x205.free();

// fc6_B <~~ V_fc6_B
float x206, x207;
x206 = 1;
x207 = decay_1;
JCudaTensor x208;
x208 = x189;
x83.update(x208, x206, x207);


// fc6_W <~~ V_fc6_W
float x209, x210;
x209 = 1;
x210 = decay_1;
JCudaTensor x211;
x211 = x193;
x79.update(x211, x209, x210);


// val X148 = X146 * d_ReLU()(X22)/d_X21
JCudaTensor x212;
JCudaTensor x213, x214;
x213 = x199;
x214 = x67;
x212 = x69.backward(x213,x214);

// Dealloc(X22)
JCudaTensor x215;
x215 = x67;
x215.free();

// val X149 = X148 * d_Convolv(1,1)(cv5_W)/d_X20
JCudaTensor x216;
JCudaTensor x217, x218;
x217 = x212;
x218 = x64;
x216 = x66.backward_data(x217,x218);

// V_cv5_W <~~ X148 * d_Convolv(1,1)(X20)/d_cv5_W
float x220, x221;
x220 = lrn_rate_1;
x221 = momentum;
JCudaTensor x222, x223;
x222 = x212;
x223 = x57;
x66.backward_filter(x222,x223, x219, x220, x221);

// V_cv5_B <~~ X148 * d_Convolv(1,1)()/d_cv5_B
float x225, x226;
x225 = lrn_rate_2;
x226 = momentum;
JCudaTensor x227;
x227 = x212;
x66.backward_bias(x227, x224, x225, x226);

// Dealloc(X148)
JCudaTensor x228;
x228 = x212;
x228.free();

// cv5_W <~~ V_cv5_W
float x229, x230;
x229 = 1;
x230 = decay_1;
JCudaTensor x231;
x231 = x219;
x64.update(x231, x229, x230);


// cv5_B <~~ V_cv5_B
float x232, x233;
x232 = 1;
x233 = 1;
JCudaTensor x234;
x234 = x224;
x65.update(x234, x232, x233);


// val X151 = X149 * d_ReLU()(X20)/d_X19
JCudaTensor x235;
JCudaTensor x236, x237;
x236 = x216;
x237 = x57;
x235 = x59.backward(x236,x237);

// Dealloc(X20)
JCudaTensor x238;
x238 = x57;
x238.free();

// val X152 = X151 * d_Convolv(1,1)(cv4_W)/d_X18
JCudaTensor x239;
JCudaTensor x240, x241;
x240 = x235;
x241 = x54;
x239 = x56.backward_data(x240,x241);

// V_cv4_W <~~ X151 * d_Convolv(1,1)(X18)/d_cv4_W
float x243, x244;
x243 = lrn_rate_1;
x244 = momentum;
JCudaTensor x245, x246;
x245 = x235;
x246 = x47;
x56.backward_filter(x245,x246, x242, x243, x244);

// V_cv4_B <~~ X151 * d_Convolv(1,1)()/d_cv4_B
float x248, x249;
x248 = lrn_rate_2;
x249 = momentum;
JCudaTensor x250;
x250 = x235;
x56.backward_bias(x250, x247, x248, x249);

// Dealloc(X151)
JCudaTensor x251;
x251 = x235;
x251.free();

// cv4_W <~~ V_cv4_W
float x252, x253;
x252 = 1;
x253 = decay_1;
JCudaTensor x254;
x254 = x242;
x54.update(x254, x252, x253);


// cv4_B <~~ V_cv4_B
float x255, x256;
x255 = 1;
x256 = 1;
JCudaTensor x257;
x257 = x247;
x55.update(x257, x255, x256);


// val X154 = X152 * d_ReLU()(X18)/d_X17
JCudaTensor x258;
JCudaTensor x259, x260;
x259 = x239;
x260 = x47;
x258 = x49.backward(x259,x260);

// Dealloc(X18)
JCudaTensor x261;
x261 = x47;
x261.free();

// val X155 = X154 * d_Convolv(1,1)(cv3_W)/d_X16
JCudaTensor x262;
JCudaTensor x263, x264;
x263 = x258;
x264 = x44;
x262 = x46.backward_data(x263,x264);

// V_cv3_B <~~ X154 * d_Convolv(1,1)()/d_cv3_B
float x266, x267;
x266 = lrn_rate_2;
x267 = momentum;
JCudaTensor x268;
x268 = x258;
x46.backward_bias(x268, x265, x266, x267);

// V_cv3_W <~~ X154 * d_Convolv(1,1)(X16)/d_cv3_W
float x270, x271;
x270 = lrn_rate_1;
x271 = momentum;
JCudaTensor x272, x273;
x272 = x258;
x273 = x37;
x46.backward_filter(x272,x273, x269, x270, x271);

// Dealloc(X154)
JCudaTensor x274;
x274 = x258;
x274.free();

// cv3_B <~~ V_cv3_B
float x275, x276;
x275 = 1;
x276 = 1;
JCudaTensor x277;
x277 = x265;
x45.update(x277, x275, x276);


// cv3_W <~~ V_cv3_W
float x278, x279;
x278 = 1;
x279 = decay_1;
JCudaTensor x280;
x280 = x269;
x44.update(x280, x278, x279);


// val X157 = X155 * d_Pooling(2,2,0,true)(X16,X15)/d_X15
JCudaTensor x281;
JCudaTensor x282, x283, x284;
x282 = x262;
x283 = x37;
x284 = x34;
x281 = x39.backward(x282,x283,x284);

// Dealloc(X155)
JCudaTensor x285;
x285 = x262;
x285.free();

// Dealloc(X16)
JCudaTensor x286;
x286 = x37;
x286.free();

// val X159 = X157 * d_ReLU()(X15)/d_X14
JCudaTensor x287;
JCudaTensor x288, x289;
x288 = x281;
x289 = x34;
x287 = x36.backward(x288,x289);

// Dealloc(X15)
JCudaTensor x290;
x290 = x34;
x290.free();

// val X160 = X159 * d_Convolv(1,2)(cv2_W)/d_X13
JCudaTensor x291;
JCudaTensor x292, x293;
x292 = x287;
x293 = x31;
x291 = x33.backward_data(x292,x293);

// V_cv2_B <~~ X159 * d_Convolv(1,2)()/d_cv2_B
float x295, x296;
x295 = lrn_rate_2;
x296 = momentum;
JCudaTensor x297;
x297 = x287;
x33.backward_bias(x297, x294, x295, x296);

// V_cv2_W <~~ X159 * d_Convolv(1,2)(X13)/d_cv2_W
float x299, x300;
x299 = lrn_rate_1;
x300 = momentum;
JCudaTensor x301, x302;
x301 = x287;
x302 = x24;
x33.backward_filter(x301,x302, x298, x299, x300);

// Dealloc(X159)
JCudaTensor x303;
x303 = x287;
x303.free();

// cv2_B <~~ V_cv2_B
float x304, x305;
x304 = 1;
x305 = 1;
JCudaTensor x306;
x306 = x294;
x32.update(x306, x304, x305);


// cv2_W <~~ V_cv2_W
float x307, x308;
x307 = 1;
x308 = decay_1;
JCudaTensor x309;
x309 = x298;
x31.update(x309, x307, x308);


// val X162 = X160 * d_Pooling(2,2,0,true)(X13,X12)/d_X12
JCudaTensor x310;
JCudaTensor x311, x312, x313;
x311 = x291;
x312 = x24;
x313 = x21;
x310 = x26.backward(x311,x312,x313);

// Dealloc(X160)
JCudaTensor x314;
x314 = x291;
x314.free();

// Dealloc(X13)
JCudaTensor x315;
x315 = x24;
x315.free();

// val X164 = X162 * d_ReLU()(X12)/d_X11
JCudaTensor x316;
JCudaTensor x317, x318;
x317 = x310;
x318 = x21;
x316 = x23.backward(x317,x318);

// Dealloc(X12)
JCudaTensor x319;
x319 = x21;
x319.free();

// V_cv1_W <~~ X164 * d_Convolv(4,0)(X10)/d_cv1_W
float x321, x322;
x321 = lrn_rate_1;
x322 = momentum;
JCudaTensor x323, x324;
x323 = x316;
x324 = x9;
x17.backward_filter(x323,x324, x320, x321, x322);

// Dealloc(X10)
JCudaTensor x325;
x325 = x9;
x325.free();

// V_cv1_B <~~ X164 * d_Convolv(4,0)()/d_cv1_B
float x327, x328;
x327 = lrn_rate_2;
x328 = momentum;
JCudaTensor x329;
x329 = x316;
x17.backward_bias(x329, x326, x327, x328);

// Dealloc(X164)
JCudaTensor x330;
x330 = x316;
x330.free();

// cv1_W <~~ V_cv1_W
float x331, x332;
x331 = 1;
x332 = decay_1;
JCudaTensor x333;
x333 = x320;
x15.update(x333, x331, x332);


// cv1_B <~~ V_cv1_B
float x334, x335;
x334 = 1;
x335 = 1;
JCudaTensor x336;
x336 = x326;
x16.update(x336, x334, x335);


}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X1767 = Cuda(X)
JCudaTensor x337;
JTensorFloat x338;
x338 = x3;
x337 = x338.asJCudaTensor();

// val X1768 = Convolv(4,0)(X1767,cv1_W,cv1_B)
JCudaTensor x339;
JCudaTensor x340, x341, x342;
x340 = x337;
x341 = x15;
x342 = x16;
x339 = x17.forward(x340,x341,x342);

// Dealloc(X1767)
JCudaTensor x343;
x343 = x337;
x343.free();

// val X1769 = ReLU()(X1768)
JCudaTensor x344;
JCudaTensor x345;
x345 = x339;
x344 = x23.forward(x345);

// val X1770 = Pooling(2,2,0,true)(X1769)
JCudaTensor x346;
JCudaTensor x347;
x347 = x344;
x346 = x26.forward(x347);

// Dealloc(X1769)
JCudaTensor x348;
x348 = x344;
x348.free();

// val X1771 = Convolv(1,2)(X1770,cv2_W,cv2_B)
JCudaTensor x349;
JCudaTensor x350, x351, x352;
x350 = x346;
x351 = x31;
x352 = x32;
x349 = x33.forward(x350,x351,x352);

// Dealloc(X1770)
JCudaTensor x353;
x353 = x346;
x353.free();

// val X1772 = ReLU()(X1771)
JCudaTensor x354;
JCudaTensor x355;
x355 = x349;
x354 = x36.forward(x355);

// val X1773 = Pooling(2,2,0,true)(X1772)
JCudaTensor x356;
JCudaTensor x357;
x357 = x354;
x356 = x39.forward(x357);

// Dealloc(X1772)
JCudaTensor x358;
x358 = x354;
x358.free();

// val X1774 = Convolv(1,1)(X1773,cv3_W,cv3_B)
JCudaTensor x359;
JCudaTensor x360, x361, x362;
x360 = x356;
x361 = x44;
x362 = x45;
x359 = x46.forward(x360,x361,x362);

// Dealloc(X1773)
JCudaTensor x363;
x363 = x356;
x363.free();

// val X1775 = ReLU()(X1774)
JCudaTensor x364;
JCudaTensor x365;
x365 = x359;
x364 = x49.forward(x365);

// val X1776 = Convolv(1,1)(X1775,cv4_W,cv4_B)
JCudaTensor x366;
JCudaTensor x367, x368, x369;
x367 = x364;
x368 = x54;
x369 = x55;
x366 = x56.forward(x367,x368,x369);

// Dealloc(X1775)
JCudaTensor x370;
x370 = x364;
x370.free();

// val X1777 = ReLU()(X1776)
JCudaTensor x371;
JCudaTensor x372;
x372 = x366;
x371 = x59.forward(x372);

// val X1778 = Convolv(1,1)(X1777,cv5_W,cv5_B)
JCudaTensor x373;
JCudaTensor x374, x375, x376;
x374 = x371;
x375 = x64;
x376 = x65;
x373 = x66.forward(x374,x375,x376);

// Dealloc(X1777)
JCudaTensor x377;
x377 = x371;
x377.free();

// val X1779 = ReLU()(X1778)
JCudaTensor x378;
JCudaTensor x379;
x379 = x373;
x378 = x69.forward(x379);

// val X1780 = Pooling(2,2,0,true)(X1779)
JCudaTensor x380;
JCudaTensor x381;
x381 = x378;
x380 = x72.forward(x381);

// Dealloc(X1779)
JCudaTensor x382;
x382 = x378;
x382.free();

// val X1781 = (X1780[1><3])(i | @) * (fc6_W)(j | @)
JCudaTensor x383;
JCudaMatrix x384;
JCudaMatrix x385;
JCudaTensor x386;
JCudaTensor x387;
x387 = x380;
x386 = x387.flatten(1, new int[]{1024, 6, 6});
x384 = x386.asMatrix(1, true);
JCudaTensor x388;
x388 = x79;
x385 = x388.asMatrix(1, true);
x383 = x384.times(x385);

// Dealloc(X1780)
JCudaTensor x389;
x389 = x380;
x389.free();

// val X1783 = (X1781 + (i) => fc6_B)
JCudaTensor x390;
JCudaTensor x391, x392;
x391 = x383;
x392 = x83;
x390 = x392.copy(128, x391);

// val X1784 = (X1783)(i | @) * (fc7_W)(j | @)
JCudaTensor x393;
JCudaMatrix x394;
JCudaMatrix x395;
JCudaTensor x396;
x396 = x390;
x394 = x396.asMatrix(1, true);
JCudaTensor x397;
x397 = x89;
x395 = x397.asMatrix(1, true);
x393 = x394.times(x395);

// Dealloc(X1783)
JCudaTensor x398;
x398 = x390;
x398.free();

// val X1786 = (X1784 + (i) => fc7_B)
JCudaTensor x399;
JCudaTensor x400, x401;
x400 = x393;
x401 = x93;
x399 = x401.copy(128, x400);

// val X1787 = (X1786)(i | @) * (fc8_W)(j | @)
JCudaTensor x402;
JCudaMatrix x403;
JCudaMatrix x404;
JCudaTensor x405;
x405 = x399;
x403 = x405.asMatrix(1, true);
JCudaTensor x406;
x406 = x99;
x404 = x406.asMatrix(1, true);
x402 = x403.times(x404);

// Dealloc(X1786)
JCudaTensor x407;
x407 = x399;
x407.free();

// val X1789 = (X1787 + (i) => fc8_B)
JCudaTensor x408;
JCudaTensor x409, x410;
x409 = x402;
x410 = x103;
x408 = x410.copy(128, x409);

// val X1790 = Cuda(Indicator(Y, 1000))
JCudaTensor x411;
JTensorFloat x412;
x412 = x4.asIndicator(1000);
x411 = x412.asJCudaTensor();

// val X1791 = X1790 .* X1789
JCudaTensor x413;
JCudaTensor x414, x415;
x414 = x411;
x415 = x408;
x413 = x414.times_i(x415);

// val X1792 = Sum((X1791)(i177 | @))
JCudaTensor x416;
JCudaMatrix x417;
JCudaTensor x418;
x418 = x413;
x417 = x418.asMatrix(1, true);
x416 = x417.sum();

// Dealloc(X1791)
JCudaTensor x419;
x419 = x413;
x419.free();

// val X1793 = Max((X1789)(i177 | @))
JCudaTensor x420;
JCudaMatrix x421;
JCudaTensor x422;
x422 = x408;
x421 = x422.asMatrix(1, true);
x420 = x421.max();

// Dealloc(X1789)
JCudaTensor x423;
x423 = x408;
x423.free();

// val X1794 = 1{X1792 == X1793}
JCudaTensor x424;
JCudaTensor x425, x426;
x425 = x416;
x426 = x420;
x424 = x425.eq(x426);

// Dealloc(X1793)
JCudaTensor x427;
x427 = x420;
x427.free();

// Print((Sum(X1794) / |128|))
float x428;
float x429;
float x430;
JCudaTensor x431;
x431 = x424;
x429 = x431.sum();
x430 = 128;
x428 = x429 / x430;
System.out.println(x5 + " test precision "  + x428);

// Dealloc(X1794)
JCudaTensor x432;
x432 = x424;
x432.free();

}
 
}

}