package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Googlenet {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// decay
static float decay = 5.0E-4f;
// loss1
static float loss1 = 0.3f;
// loss2
static float loss2 = 0.3f;
// lrn_rate
static float lrn_rate = -0.01f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/googlenet";
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

// (Convolv(1,0),List(List(128, 192, 28, 28), List(16, 192, 1, 1), List(16)))
static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(32, 192, 1, 1), List(32)))
static JCudnnConvolution x86 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(64, 192, 1, 1), List(64)))
static JCudnnConvolution x72 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(96, 192, 1, 1), List(96)))
static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x196 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x223 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x210 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x203 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x133 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x154 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x140 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x147 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 4, 4), List(128, 256, 1, 1), List(128)))
static JCudnnConvolution x292 = new JCudnnConvolution(new int[]{128,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x726 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x756 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x733 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x740 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 64, 56, 56), List(64, 64, 1, 1), List(64)))
static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,1),List(List(128, 64, 56, 56), List(192, 64, 3, 3), List(192)))
static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 14, 14), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x237 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 28, 28), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x99 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 7, 7), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x766 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x230 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x106 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(List(128, 16, 7, 7), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x773 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(2,3),List(List(128, 3, 224, 224), List(64, 3, 7, 7), List(64)))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (Dropout(0.4),List(List(128, 256, 1, 1)))
static JCudnnDropout x975 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
// (Dropout(0.7),List(List(128, 1024)))
static JCudnnDropout x359 = new JCudnnDropout(new int[]{128,1024}, 0.7f);
// (Imagenet,false)
static ImagenetFactory x2 = ImagenetFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
// (Imagenet,true)
static ImagenetFactory x1 = ImagenetFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, 1000, false);
// (LRN(5,1.0E-4,0.75),List(List(128, 192, 56, 56)))
static JCudnnLRN x52 = new JCudnnLRN(new int[]{128,192,56,56}, 5, 1.0E-4, 0.75);
// (LRN(5,1.0E-4,0.75),List(List(128, 64, 56, 56)))
static JCudnnLRN x29 = new JCudnnLRN(new int[]{128,64,56,56}, 5, 1.0E-4, 0.75);
// (LogSoftmax(),List(List(128, 1000)))
static JCudnnSoftmax x407 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
static JCudnnPooling x65 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
static JCudnnPooling x189 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
static JCudnnPooling x126 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
static JCudnnPooling x719 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 192, 56, 56)))
static JCudnnPooling x55 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 256, 14, 14)))
static JCudnnPooling x714 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 256, 28, 28)))
static JCudnnPooling x186 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 64, 112, 112)))
static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, PoolingType.MAX);
// (Pooling(5,3,0,false),List(List(128, 256, 14, 14)))
static JCudnnPooling x269 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (Pooling(7,1,0,false),List(List(128, 256, 7, 7)))
static JCudnnPooling x938 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (ReLU(),List(List(128, 1024)))
static JCudnnActivation x350 = new JCudnnActivation(new int[]{128,1024}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 14, 14)))
static JCudnnActivation x243 = new JCudnnActivation(new int[]{128,128,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 28, 28)))
static JCudnnActivation x112 = new JCudnnActivation(new int[]{128,128,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 4, 4)))
static JCudnnActivation x303 = new JCudnnActivation(new int[]{128,128,4,4}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 7, 7)))
static JCudnnActivation x780 = new JCudnnActivation(new int[]{128,128,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 14, 14)))
static JCudnnActivation x213 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 28, 28)))
static JCudnnActivation x92 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 7, 7)))
static JCudnnActivation x759 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 192, 56, 56)))
static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,192,56,56}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 14, 14)))
static JCudnnActivation x246 = new JCudnnActivation(new int[]{128,32,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 28, 28)))
static JCudnnActivation x115 = new JCudnnActivation(new int[]{128,32,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 7, 7)))
static JCudnnActivation x783 = new JCudnnActivation(new int[]{128,32,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 112, 112)))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,64,112,112}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 14, 14)))
static JCudnnActivation x240 = new JCudnnActivation(new int[]{128,64,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 28, 28)))
static JCudnnActivation x109 = new JCudnnActivation(new int[]{128,64,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 56, 56)))
static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,64,56,56}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 7, 7)))
static JCudnnActivation x788 = new JCudnnActivation(new int[]{128,64,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 14, 14)))
static JCudnnActivation x216 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 28, 28)))
static JCudnnActivation x89 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 7, 7)))
static JCudnnActivation x743 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
static JCudnnConcat x250 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
static JCudnnConcat x119 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
static JCudnnConcat x793 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
// Precision(Accuracy(X7862, Y, 1))
static float x3831;
// V_b1cv_B
static JCudaTensor x662 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b1cv_W
static JCudaTensor x680 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b1fc1_B
static JCudaTensor x629 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b1fc1_W
static JCudaTensor x610 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b1fc2_B
static JCudaTensor x498 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b1fc2_W
static JCudaTensor x519 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_b2cv_B
static JCudaTensor x1011 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b2cv_W
static JCudaTensor x997 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b2fc1_B
static JCudaTensor x976 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b2fc1_W
static JCudaTensor x957 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b2fc2_B
static JCudaTensor x887 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b2fc2_W
static JCudaTensor x880 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_cv11_B
static JCudaTensor x2993 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv11_W
static JCudaTensor x3032 = JTensor.constFloat(0.0f, 64, 192, 1, 1).asJCudaTensor();
// V_cv12_B
static JCudaTensor x3126 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv12_W
static JCudaTensor x3142 = JTensor.constFloat(0.0f, 96, 192, 1, 1).asJCudaTensor();
// V_cv13_B
static JCudaTensor x2987 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv13_W
static JCudaTensor x3022 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv14_B
static JCudaTensor x3136 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv14_W
static JCudaTensor x3119 = JTensor.constFloat(0.0f, 16, 192, 1, 1).asJCudaTensor();
// V_cv15_B
static JCudaTensor x3040 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv15_W
static JCudaTensor x3012 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv16_B
static JCudaTensor x3006 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv16_W
static JCudaTensor x2999 = JTensor.constFloat(0.0f, 32, 192, 1, 1).asJCudaTensor();
// V_cv1_B
static JCudaTensor x3294 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv1_W
static JCudaTensor x3286 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
// V_cv21_B
static JCudaTensor x2821 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv21_W
static JCudaTensor x2763 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv22_B
static JCudaTensor x2905 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv22_W
static JCudaTensor x2915 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv23_B
static JCudaTensor x2783 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv23_W
static JCudaTensor x2792 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv24_B
static JCudaTensor x2899 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv24_W
static JCudaTensor x2892 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv25_B
static JCudaTensor x2770 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv25_W
static JCudaTensor x2810 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv26_B
static JCudaTensor x2803 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv26_W
static JCudaTensor x2776 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv2_B
static JCudaTensor x3243 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv2_W
static JCudaTensor x3249 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
// V_cv31_B
static JCudaTensor x2530 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv31_W
static JCudaTensor x2542 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv32_B
static JCudaTensor x2672 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv32_W
static JCudaTensor x2682 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv33_B
static JCudaTensor x2588 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv33_W
static JCudaTensor x2549 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv34_B
static JCudaTensor x2666 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv34_W
static JCudaTensor x2659 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv35_B
static JCudaTensor x2564 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv35_W
static JCudaTensor x2570 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv36_B
static JCudaTensor x2536 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv36_W
static JCudaTensor x2556 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv3_B
static JCudaTensor x3208 = JTensor.constFloat(0.0f, 192).asJCudaTensor();
// V_cv3_W
static JCudaTensor x3214 = JTensor.constFloat(0.0f, 192, 64, 3, 3).asJCudaTensor();
// V_cv41_B
static JCudaTensor x2340 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv41_W
static JCudaTensor x2333 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv42_B
static JCudaTensor x2436 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv42_W
static JCudaTensor x2442 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv43_B
static JCudaTensor x2293 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv43_W
static JCudaTensor x2305 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv44_B
static JCudaTensor x2450 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv44_W
static JCudaTensor x2425 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv45_B
static JCudaTensor x2320 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv45_W
static JCudaTensor x2313 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv46_B
static JCudaTensor x2354 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv46_W
static JCudaTensor x2347 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv51_B
static JCudaTensor x2099 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv51_W
static JCudaTensor x2111 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv52_B
static JCudaTensor x2205 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv52_W
static JCudaTensor x2221 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv53_B
static JCudaTensor x2092 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv53_W
static JCudaTensor x2079 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv54_B
static JCudaTensor x2211 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv54_W
static JCudaTensor x2198 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv55_B
static JCudaTensor x2127 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv55_W
static JCudaTensor x2066 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv56_B
static JCudaTensor x2105 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv56_W
static JCudaTensor x2119 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv61_B
static JCudaTensor x1900 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv61_W
static JCudaTensor x1861 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv62_B
static JCudaTensor x1989 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv62_W
static JCudaTensor x1982 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv63_B
static JCudaTensor x1855 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv63_W
static JCudaTensor x1876 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv64_B
static JCudaTensor x1996 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv64_W
static JCudaTensor x1971 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv65_B
static JCudaTensor x1846 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv65_W
static JCudaTensor x1868 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv66_B
static JCudaTensor x1883 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv66_W
static JCudaTensor x1839 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv71_B
static JCudaTensor x1648 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv71_W
static JCudaTensor x1633 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv72_B
static JCudaTensor x1755 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv72_W
static JCudaTensor x1748 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv73_B
static JCudaTensor x1618 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv73_W
static JCudaTensor x1655 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv74_B
static JCudaTensor x1762 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv74_W
static JCudaTensor x1737 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv75_B
static JCudaTensor x1663 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv75_W
static JCudaTensor x1605 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv76_B
static JCudaTensor x1627 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv76_W
static JCudaTensor x1640 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv81_B
static JCudaTensor x1408 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv81_W
static JCudaTensor x1381 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv82_B
static JCudaTensor x1511 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv82_W
static JCudaTensor x1521 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv83_B
static JCudaTensor x1388 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv83_W
static JCudaTensor x1394 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv84_B
static JCudaTensor x1529 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv84_W
static JCudaTensor x1504 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv85_B
static JCudaTensor x1375 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv85_W
static JCudaTensor x1414 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv86_B
static JCudaTensor x1426 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv86_W
static JCudaTensor x1401 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv91_B
static JCudaTensor x1161 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv91_W
static JCudaTensor x1191 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv92_B
static JCudaTensor x1301 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv92_W
static JCudaTensor x1294 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv93_B
static JCudaTensor x1199 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv93_W
static JCudaTensor x1205 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv94_B
static JCudaTensor x1284 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv94_W
static JCudaTensor x1277 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv95_B
static JCudaTensor x1174 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv95_W
static JCudaTensor x1145 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv96_B
static JCudaTensor x1181 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv96_W
static JCudaTensor x1167 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_fc_B
static JCudaTensor x1094 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc_W
static JCudaTensor x1086 = JTensor.constFloat(0.0f, 1000, 256).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// b1cv_B
static JCudaTensor x291 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b1cv_B").asJCudaTensor();
// b1cv_W
static JCudaTensor x290 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b1cv_W").asJCudaTensor();
// b1fc1_B
static JCudaTensor x333 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b1fc1_B").asJCudaTensor();
// b1fc1_W
static JCudaTensor x320 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b1fc1_W").asJCudaTensor();
// b1fc2_B
static JCudaTensor x399 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b1fc2_B").asJCudaTensor();
// b1fc2_W
static JCudaTensor x387 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b1fc2_W").asJCudaTensor();
// b2cv_B
static JCudaTensor x603 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b2cv_B").asJCudaTensor();
// b2cv_W
static JCudaTensor x602 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b2cv_W").asJCudaTensor();
// b2fc1_B
static JCudaTensor x709 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b2fc1_B").asJCudaTensor();
// b2fc1_W
static JCudaTensor x679 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b2fc1_W").asJCudaTensor();
// b2fc2_B
static JCudaTensor x777 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b2fc2_B").asJCudaTensor();
// b2fc2_W
static JCudaTensor x749 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b2fc2_W").asJCudaTensor();
// cv11_B
static JCudaTensor x71 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
// cv11_W
static JCudaTensor x70 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 64, 192, 1, 1).load(network_dir + "/cv11_W").asJCudaTensor();
// cv12_B
static JCudaTensor x61 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv12_B").asJCudaTensor();
// cv12_W
static JCudaTensor x60 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 96, 192, 1, 1).load(network_dir + "/cv12_W").asJCudaTensor();
// cv13_B
static JCudaTensor x98 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv13_B").asJCudaTensor();
// cv13_W
static JCudaTensor x97 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv13_W").asJCudaTensor();
// cv14_B
static JCudaTensor x78 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv14_B").asJCudaTensor();
// cv14_W
static JCudaTensor x77 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 16, 192, 1, 1).load(network_dir + "/cv14_W").asJCudaTensor();
// cv15_B
static JCudaTensor x105 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv15_B").asJCudaTensor();
// cv15_W
static JCudaTensor x104 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv15_W").asJCudaTensor();
// cv16_B
static JCudaTensor x85 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv16_B").asJCudaTensor();
// cv16_W
static JCudaTensor x84 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 32, 192, 1, 1).load(network_dir + "/cv16_W").asJCudaTensor();
// cv1_B
static JCudaTensor x16 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv1_B").asJCudaTensor();
// cv1_W
static JCudaTensor x15 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/cv1_W").asJCudaTensor();
// cv21_B
static JCudaTensor x139 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv21_B").asJCudaTensor();
// cv21_W
static JCudaTensor x138 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv21_W").asJCudaTensor();
// cv22_B
static JCudaTensor x146 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv22_B").asJCudaTensor();
// cv22_W
static JCudaTensor x145 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv22_W").asJCudaTensor();
// cv23_B
static JCudaTensor x170 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv23_B").asJCudaTensor();
// cv23_W
static JCudaTensor x169 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv23_W").asJCudaTensor();
// cv24_B
static JCudaTensor x132 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
// cv24_W
static JCudaTensor x131 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
// cv25_B
static JCudaTensor x164 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
// cv25_W
static JCudaTensor x163 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
// cv26_B
static JCudaTensor x153 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
// cv26_W
static JCudaTensor x152 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
// cv2_B
static JCudaTensor x35 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x34 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
// cv31_B
static JCudaTensor x209 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
// cv31_W
static JCudaTensor x208 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
// cv32_B
static JCudaTensor x202 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
// cv32_W
static JCudaTensor x201 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
// cv33_B
static JCudaTensor x236 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
// cv33_W
static JCudaTensor x235 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
// cv34_B
static JCudaTensor x195 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
// cv34_W
static JCudaTensor x194 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
// cv35_B
static JCudaTensor x229 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
// cv35_W
static JCudaTensor x228 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
// cv36_B
static JCudaTensor x222 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
// cv36_W
static JCudaTensor x221 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
// cv3_B
static JCudaTensor x45 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x44 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv41_B
static JCudaTensor x266 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
// cv41_W
static JCudaTensor x265 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
// cv42_B
static JCudaTensor x277 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
// cv42_W
static JCudaTensor x276 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
// cv43_B
static JCudaTensor x309 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
// cv43_W
static JCudaTensor x308 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
// cv44_B
static JCudaTensor x260 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
// cv44_W
static JCudaTensor x259 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
// cv45_B
static JCudaTensor x300 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
// cv45_W
static JCudaTensor x299 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
// cv46_B
static JCudaTensor x285 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
// cv46_W
static JCudaTensor x284 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
// cv51_B
static JCudaTensor x356 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
// cv51_W
static JCudaTensor x355 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
// cv52_B
static JCudaTensor x345 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
// cv52_W
static JCudaTensor x344 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
// cv53_B
static JCudaTensor x375 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv53_B").asJCudaTensor();
// cv53_W
static JCudaTensor x374 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
// cv54_B
static JCudaTensor x339 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv54_B").asJCudaTensor();
// cv54_W
static JCudaTensor x338 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv54_W").asJCudaTensor();
// cv55_B
static JCudaTensor x381 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv55_B").asJCudaTensor();
// cv55_W
static JCudaTensor x380 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv55_W").asJCudaTensor();
// cv56_B
static JCudaTensor x365 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
// cv56_W
static JCudaTensor x364 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
// cv61_B
static JCudaTensor x424 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
// cv61_W
static JCudaTensor x423 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
// cv62_B
static JCudaTensor x435 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
// cv62_W
static JCudaTensor x434 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
// cv63_B
static JCudaTensor x470 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
// cv63_W
static JCudaTensor x469 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
// cv64_B
static JCudaTensor x418 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
// cv64_W
static JCudaTensor x417 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
// cv65_B
static JCudaTensor x461 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
// cv65_W
static JCudaTensor x460 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
// cv66_B
static JCudaTensor x451 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
// cv66_W
static JCudaTensor x450 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
// cv71_B
static JCudaTensor x545 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
// cv71_W
static JCudaTensor x544 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
// cv72_B
static JCudaTensor x557 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
// cv72_W
static JCudaTensor x556 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
// cv73_B
static JCudaTensor x628 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
// cv73_W
static JCudaTensor x627 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
// cv74_B
static JCudaTensor x563 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
// cv74_W
static JCudaTensor x562 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
// cv75_B
static JCudaTensor x622 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
// cv75_W
static JCudaTensor x621 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
// cv76_B
static JCudaTensor x593 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
// cv76_W
static JCudaTensor x592 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
// cv81_B
static JCudaTensor x732 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
// cv81_W
static JCudaTensor x731 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
// cv82_B
static JCudaTensor x739 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
// cv82_W
static JCudaTensor x738 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
// cv83_B
static JCudaTensor x765 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
// cv83_W
static JCudaTensor x764 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
// cv84_B
static JCudaTensor x725 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
// cv84_W
static JCudaTensor x724 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
// cv85_B
static JCudaTensor x772 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
// cv85_W
static JCudaTensor x771 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
// cv86_B
static JCudaTensor x755 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
// cv86_W
static JCudaTensor x754 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
// cv91_B
static JCudaTensor x808 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
// cv91_W
static JCudaTensor x807 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
// cv92_B
static JCudaTensor x828 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
// cv92_W
static JCudaTensor x827 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
// cv93_B
static JCudaTensor x856 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
// cv93_W
static JCudaTensor x855 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
// cv94_B
static JCudaTensor x818 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
// cv94_W
static JCudaTensor x817 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
// cv95_B
static JCudaTensor x871 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
// cv95_W
static JCudaTensor x870 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
// cv96_B
static JCudaTensor x845 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
// cv96_W
static JCudaTensor x844 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
// fc_B
static JCudaTensor x1035 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x1010 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
test();
x291.save(network_dir + "/b1cv_B");
x290.save(network_dir + "/b1cv_W");
x333.save(network_dir + "/b1fc1_B");
x320.save(network_dir + "/b1fc1_W");
x399.save(network_dir + "/b1fc2_B");
x387.save(network_dir + "/b1fc2_W");
x603.save(network_dir + "/b2cv_B");
x602.save(network_dir + "/b2cv_W");
x709.save(network_dir + "/b2fc1_B");
x679.save(network_dir + "/b2fc1_W");
x777.save(network_dir + "/b2fc2_B");
x749.save(network_dir + "/b2fc2_W");
x71.save(network_dir + "/cv11_B");
x70.save(network_dir + "/cv11_W");
x61.save(network_dir + "/cv12_B");
x60.save(network_dir + "/cv12_W");
x98.save(network_dir + "/cv13_B");
x97.save(network_dir + "/cv13_W");
x78.save(network_dir + "/cv14_B");
x77.save(network_dir + "/cv14_W");
x105.save(network_dir + "/cv15_B");
x104.save(network_dir + "/cv15_W");
x85.save(network_dir + "/cv16_B");
x84.save(network_dir + "/cv16_W");
x16.save(network_dir + "/cv1_B");
x15.save(network_dir + "/cv1_W");
x139.save(network_dir + "/cv21_B");
x138.save(network_dir + "/cv21_W");
x146.save(network_dir + "/cv22_B");
x145.save(network_dir + "/cv22_W");
x170.save(network_dir + "/cv23_B");
x169.save(network_dir + "/cv23_W");
x132.save(network_dir + "/cv24_B");
x131.save(network_dir + "/cv24_W");
x164.save(network_dir + "/cv25_B");
x163.save(network_dir + "/cv25_W");
x153.save(network_dir + "/cv26_B");
x152.save(network_dir + "/cv26_W");
x35.save(network_dir + "/cv2_B");
x34.save(network_dir + "/cv2_W");
x209.save(network_dir + "/cv31_B");
x208.save(network_dir + "/cv31_W");
x202.save(network_dir + "/cv32_B");
x201.save(network_dir + "/cv32_W");
x236.save(network_dir + "/cv33_B");
x235.save(network_dir + "/cv33_W");
x195.save(network_dir + "/cv34_B");
x194.save(network_dir + "/cv34_W");
x229.save(network_dir + "/cv35_B");
x228.save(network_dir + "/cv35_W");
x222.save(network_dir + "/cv36_B");
x221.save(network_dir + "/cv36_W");
x45.save(network_dir + "/cv3_B");
x44.save(network_dir + "/cv3_W");
x266.save(network_dir + "/cv41_B");
x265.save(network_dir + "/cv41_W");
x277.save(network_dir + "/cv42_B");
x276.save(network_dir + "/cv42_W");
x309.save(network_dir + "/cv43_B");
x308.save(network_dir + "/cv43_W");
x260.save(network_dir + "/cv44_B");
x259.save(network_dir + "/cv44_W");
x300.save(network_dir + "/cv45_B");
x299.save(network_dir + "/cv45_W");
x285.save(network_dir + "/cv46_B");
x284.save(network_dir + "/cv46_W");
x356.save(network_dir + "/cv51_B");
x355.save(network_dir + "/cv51_W");
x345.save(network_dir + "/cv52_B");
x344.save(network_dir + "/cv52_W");
x375.save(network_dir + "/cv53_B");
x374.save(network_dir + "/cv53_W");
x339.save(network_dir + "/cv54_B");
x338.save(network_dir + "/cv54_W");
x381.save(network_dir + "/cv55_B");
x380.save(network_dir + "/cv55_W");
x365.save(network_dir + "/cv56_B");
x364.save(network_dir + "/cv56_W");
x424.save(network_dir + "/cv61_B");
x423.save(network_dir + "/cv61_W");
x435.save(network_dir + "/cv62_B");
x434.save(network_dir + "/cv62_W");
x470.save(network_dir + "/cv63_B");
x469.save(network_dir + "/cv63_W");
x418.save(network_dir + "/cv64_B");
x417.save(network_dir + "/cv64_W");
x461.save(network_dir + "/cv65_B");
x460.save(network_dir + "/cv65_W");
x451.save(network_dir + "/cv66_B");
x450.save(network_dir + "/cv66_W");
x545.save(network_dir + "/cv71_B");
x544.save(network_dir + "/cv71_W");
x557.save(network_dir + "/cv72_B");
x556.save(network_dir + "/cv72_W");
x628.save(network_dir + "/cv73_B");
x627.save(network_dir + "/cv73_W");
x563.save(network_dir + "/cv74_B");
x562.save(network_dir + "/cv74_W");
x622.save(network_dir + "/cv75_B");
x621.save(network_dir + "/cv75_W");
x593.save(network_dir + "/cv76_B");
x592.save(network_dir + "/cv76_W");
x732.save(network_dir + "/cv81_B");
x731.save(network_dir + "/cv81_W");
x739.save(network_dir + "/cv82_B");
x738.save(network_dir + "/cv82_W");
x765.save(network_dir + "/cv83_B");
x764.save(network_dir + "/cv83_W");
x725.save(network_dir + "/cv84_B");
x724.save(network_dir + "/cv84_W");
x772.save(network_dir + "/cv85_B");
x771.save(network_dir + "/cv85_W");
x755.save(network_dir + "/cv86_B");
x754.save(network_dir + "/cv86_W");
x808.save(network_dir + "/cv91_B");
x807.save(network_dir + "/cv91_W");
x828.save(network_dir + "/cv92_B");
x827.save(network_dir + "/cv92_W");
x856.save(network_dir + "/cv93_B");
x855.save(network_dir + "/cv93_W");
x818.save(network_dir + "/cv94_B");
x817.save(network_dir + "/cv94_W");
x871.save(network_dir + "/cv95_B");
x870.save(network_dir + "/cv95_W");
x845.save(network_dir + "/cv96_B");
x844.save(network_dir + "/cv96_W");
x1035.save(network_dir + "/fc_B");
x1010.save(network_dir + "/fc_W");
x381.free();
x61.free();
x1284.free();
x871.free();
x1086.free();
x1846.free();
x1504.free();
x469.free();
x771.free();
x131.free();
x1971.free();
x15.free();
x1605.free();
x3126.free();
x2776.free();
x1839.free();
x2810.free();
x364.free();
x731.free();
x1996.free();
x2127.free();
x855.free();
x1737.free();
x957.free();
x424.free();
x229.free();
x1167.free();
x828.free();
x356.free();
x435.free();
x2570.free();
x2333.free();
x260.free();
x602.free();
x844.free();
x1181.free();
x1618.free();
x1388.free();
x1883.free();
x235.free();
x856.free();
x709.free();
x1161.free();
x164.free();
x97.free();
x138.free();
x2436.free();
x765.free();
x169.free();
x2588.free();
x34.free();
x817.free();
x732.free();
x1191.free();
x84.free();
x299.free();
x3294.free();
x285.free();
x3249.free();
x1381.free();
x2821.free();
x2442.free();
x2221.free();
x739.free();
x2905.free();
x98.free();
x3136.free();
x680.free();
x563.free();
x622.free();
x2999.free();
x221.free();
x1035.free();
x451.free();
x152.free();
x1408.free();
x1762.free();
x35.free();
x344.free();
x461.free();
x2198.free();
x375.free();
x2079.free();
x3006.free();
x71.free();
x355.free();
x2092.free();
x145.free();
x284.free();
x3012.free();
x333.free();
x880.free();
x163.free();
x1010.free();
x2542.free();
x339.free();
x777.free();
x308.free();
x70.free();
x3142.free();
x3286.free();
x77.free();
x1375.free();
x1199.free();
x2066.free();
x870.free();
x170.free();
x380.free();
x1627.free();
x1529.free();
x195.free();
x2347.free();
x519.free();
x2305.free();
x1855.free();
x399.free();
x1648.free();
x338.free();
x470.free();
x201.free();
x1011.free();
x85.free();
x1145.free();
x724.free();
x545.free();
x460.free();
x3243.free();
x290.free();
x1401.free();
x1982.free();
x2783.free();
x621.free();
x772.free();
x45.free();
x60.free();
x3208.free();
x300.free();
x1301.free();
x544.free();
x78.free();
x259.free();
x2205.free();
x808.free();
x266.free();
x153.free();
x1655.free();
x603.free();
x2672.free();
x1633.free();
x1394.free();
x628.free();
x749.free();
x104.free();
x2320.free();
x2792.free();
x1640.free();
x1989.free();
x309.free();
x827.free();
x1748.free();
x2987.free();
x1426.free();
x627.free();
x725.free();
x2915.free();
x1414.free();
x139.free();
x1868.free();
x498.free();
x276.free();
x679.free();
x1294.free();
x291.free();
x2564.free();
x2099.free();
x1876.free();
x2119.free();
x3040.free();
x2354.free();
x1900.free();
x202.free();
x2666.free();
x557.free();
x2340.free();
x562.free();
x146.free();
x2425.free();
x997.free();
x418.free();
x44.free();
x764.free();
x3214.free();
x16.free();
x1663.free();
x105.free();
x345.free();
x3032.free();
x3022.free();
x629.free();
x2105.free();
x209.free();
x1094.free();
x2803.free();
x417.free();
x2899.free();
x2313.free();
x845.free();
x222.free();
x1521.free();
x374.free();
x593.free();
x450.free();
x132.free();
x194.free();
x2659.free();
x755.free();
x1755.free();
x2211.free();
x434.free();
x2450.free();
x208.free();
x2530.free();
x738.free();
x1861.free();
x818.free();
x236.free();
x592.free();
x2536.free();
x265.free();
x2763.free();
x2770.free();
x556.free();
x1174.free();
x365.free();
x976.free();
x1511.free();
x807.free();
x2111.free();
x277.free();
x1205.free();
x228.free();
x662.free();
x2892.free();
x2993.free();
x2549.free();
x754.free();
x2682.free();
x2556.free();
x423.free();
x887.free();
x320.free();
x610.free();
x3119.free();
x2293.free();
x1277.free();
x387.free();
x407.free();
x292.free();
x350.free();
x733.free();
x714.free();
x140.free();
x780.free();
x154.free();
x39.free();
x269.free();
x740.free();
x29.free();
x237.free();
x106.free();
x147.free();
x112.free();
x213.free();
x230.free();
x126.free();
x49.free();
x36.free();
x743.free();
x186.free();
x759.free();
x975.free();
x938.free();
x196.free();
x719.free();
x203.free();
x99.free();
x65.free();
x26.free();
x23.free();
x89.free();
x773.free();
x223.free();
x86.free();
x55.free();
x92.free();
x52.free();
x788.free();
x79.free();
x133.free();
x359.free();
x756.free();
x303.free();
x46.free();
x17.free();
x62.free();
x115.free();
x72.free();
x109.free();
x766.free();
x216.free();
x246.free();
x243.free();
x783.free();
x189.free();
x726.free();
x210.free();
x240.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X7547 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X7692 = Cuda(Indicator(Y, 1000))
JCudaTensor x9;
JTensorFloat x10;
x10 = x4.asIndicator(1000);
x9 = x10.asJCudaTensor();

// val X7548 = Convolv(2,3)(X7547,cv1_W,cv1_B)
JCudaTensor x11;
JCudaTensor x12, x13, x14;
x12 = x7;
x13 = x15;
x14 = x16;
x11 = x17.forward(x12, x13, x14);

// val X2089 = - X7692.copy
JCudaTensor x18;
JCudaTensor x19;
float x20;
x19 = x9;
x19 = x19.clone();
x20 = -1;
x18 = x19.times_i(x20);

// val X7549 = ReLU()(X7548)
JCudaTensor x21;
JCudaTensor x22;
x22 = x11;
x21 = x23.forward(x22);

// val X7550 = Pooling(3,2,1,true)(X7549)
JCudaTensor x24;
JCudaTensor x25;
x25 = x21;
x24 = x26.forward(x25);

// val X7551 = LRN(5,1.0E-4,0.75)(X7550)
JCudaTensor x27;
JCudaTensor x28;
x28 = x24;
x27 = x29.forward(x28);

// val X7552 = Convolv(1,0)(X7551,cv2_W,cv2_B)
JCudaTensor x30;
JCudaTensor x31, x32, x33;
x31 = x27;
x32 = x34;
x33 = x35;
x30 = x36.forward(x31, x32, x33);

// val X7553 = ReLU()(X7552)
JCudaTensor x37;
JCudaTensor x38;
x38 = x30;
x37 = x39.forward(x38);

// val X7554 = Convolv(1,1)(X7553,cv3_W,cv3_B)
JCudaTensor x40;
JCudaTensor x41, x42, x43;
x41 = x37;
x42 = x44;
x43 = x45;
x40 = x46.forward(x41, x42, x43);

// val X7555 = ReLU()(X7554)
JCudaTensor x47;
JCudaTensor x48;
x48 = x40;
x47 = x49.forward(x48);

// val X7556 = LRN(5,1.0E-4,0.75)(X7555)
JCudaTensor x50;
JCudaTensor x51;
x51 = x47;
x50 = x52.forward(x51);

// val X7557 = Pooling(3,2,1,true)(X7556)
JCudaTensor x53;
JCudaTensor x54;
x54 = x50;
x53 = x55.forward(x54);

// val X7560 = Convolv(1,0)(X7557,cv12_W,cv12_B)
JCudaTensor x56;
JCudaTensor x57, x58, x59;
x57 = x53;
x58 = x60;
x59 = x61;
x56 = x62.forward(x57, x58, x59);

// val X7568 = Pooling(3,1,1,true)(X7557)
JCudaTensor x63;
JCudaTensor x64;
x64 = x53;
x63 = x65.forward(x64);

// val X7558 = Convolv(1,0)(X7557,cv11_W,cv11_B)
JCudaTensor x66;
JCudaTensor x67, x68, x69;
x67 = x53;
x68 = x70;
x69 = x71;
x66 = x72.forward(x67, x68, x69);

// val X7564 = Convolv(1,0)(X7557,cv14_W,cv14_B)
JCudaTensor x73;
JCudaTensor x74, x75, x76;
x74 = x53;
x75 = x77;
x76 = x78;
x73 = x79.forward(x74, x75, x76);

// val X7569 = Convolv(1,0)(X7568,cv16_W,cv16_B)
JCudaTensor x80;
JCudaTensor x81, x82, x83;
x81 = x63;
x82 = x84;
x83 = x85;
x80 = x86.forward(x81, x82, x83);

// val X7561 = ReLU()(X7560)
JCudaTensor x87;
JCudaTensor x88;
x88 = x56;
x87 = x89.forward(x88);

// val X7565 = ReLU()(X7564)
JCudaTensor x90;
JCudaTensor x91;
x91 = x73;
x90 = x92.forward(x91);

// val X7562 = Convolv(1,1)(X7561,cv13_W,cv13_B)
JCudaTensor x93;
JCudaTensor x94, x95, x96;
x94 = x87;
x95 = x97;
x96 = x98;
x93 = x99.forward(x94, x95, x96);

// val X7566 = Convolv(1,2)(X7565,cv15_W,cv15_B)
JCudaTensor x100;
JCudaTensor x101, x102, x103;
x101 = x90;
x102 = x104;
x103 = x105;
x100 = x106.forward(x101, x102, x103);

// val X7559 = ReLU()(X7558)
JCudaTensor x107;
JCudaTensor x108;
x108 = x66;
x107 = x109.forward(x108);

// val X7563 = ReLU()(X7562)
JCudaTensor x110;
JCudaTensor x111;
x111 = x93;
x110 = x112.forward(x111);

// val X7567 = ReLU()(X7566)
JCudaTensor x113;
JCudaTensor x114;
x114 = x100;
x113 = x115.forward(x114);

// val X7570 = ReLU()(X7569)
JCudaTensor x116;
JCudaTensor x117;
x117 = x80;
x116 = x115.forward(x117);

// val X7571 = Concat(X7559,X7563,X7567,X7570)
JCudaTensor x118;
JCudaTensor x120, x121, x122, x123;
x120 = x107;
x121 = x110;
x122 = x113;
x123 = x116;
x118 = x119.forward(x120,x121,x122,x123);

// val X7582 = Pooling(3,1,1,true)(X7571)
JCudaTensor x124;
JCudaTensor x125;
x125 = x118;
x124 = x126.forward(x125);

// val X7578 = Convolv(1,0)(X7571,cv24_W,cv24_B)
JCudaTensor x127;
JCudaTensor x128, x129, x130;
x128 = x118;
x129 = x131;
x130 = x132;
x127 = x133.forward(x128, x129, x130);

// val X7572 = Convolv(1,0)(X7571,cv21_W,cv21_B)
JCudaTensor x134;
JCudaTensor x135, x136, x137;
x135 = x118;
x136 = x138;
x137 = x139;
x134 = x140.forward(x135, x136, x137);

// val X7574 = Convolv(1,0)(X7571,cv22_W,cv22_B)
JCudaTensor x141;
JCudaTensor x142, x143, x144;
x142 = x118;
x143 = x145;
x144 = x146;
x141 = x147.forward(x142, x143, x144);

// val X7583 = Convolv(1,0)(X7582,cv26_W,cv26_B)
JCudaTensor x148;
JCudaTensor x149, x150, x151;
x149 = x124;
x150 = x152;
x151 = x153;
x148 = x154.forward(x149, x150, x151);

// val X7579 = ReLU()(X7578)
JCudaTensor x155;
JCudaTensor x156;
x156 = x127;
x155 = x92.forward(x156);

// val X7575 = ReLU()(X7574)
JCudaTensor x157;
JCudaTensor x158;
x158 = x141;
x157 = x89.forward(x158);

// val X7580 = Convolv(1,2)(X7579,cv25_W,cv25_B)
JCudaTensor x159;
JCudaTensor x160, x161, x162;
x160 = x155;
x161 = x163;
x162 = x164;
x159 = x106.forward(x160, x161, x162);

// val X7576 = Convolv(1,1)(X7575,cv23_W,cv23_B)
JCudaTensor x165;
JCudaTensor x166, x167, x168;
x166 = x157;
x167 = x169;
x168 = x170;
x165 = x99.forward(x166, x167, x168);

// val X7573 = ReLU()(X7572)
JCudaTensor x171;
JCudaTensor x172;
x172 = x134;
x171 = x109.forward(x172);

// val X7577 = ReLU()(X7576)
JCudaTensor x173;
JCudaTensor x174;
x174 = x165;
x173 = x112.forward(x174);

// val X7581 = ReLU()(X7580)
JCudaTensor x175;
JCudaTensor x176;
x176 = x159;
x175 = x115.forward(x176);

// val X7584 = ReLU()(X7583)
JCudaTensor x177;
JCudaTensor x178;
x178 = x148;
x177 = x115.forward(x178);

// val X7585 = Concat(X7573,X7577,X7581,X7584)
JCudaTensor x179;
JCudaTensor x180, x181, x182, x183;
x180 = x171;
x181 = x173;
x182 = x175;
x183 = x177;
x179 = x119.forward(x180,x181,x182,x183);

// val X7586 = Pooling(3,2,1,true)(X7585)
JCudaTensor x184;
JCudaTensor x185;
x185 = x179;
x184 = x186.forward(x185);

// val X7597 = Pooling(3,1,1,true)(X7586)
JCudaTensor x187;
JCudaTensor x188;
x188 = x184;
x187 = x189.forward(x188);

// val X7593 = Convolv(1,0)(X7586,cv34_W,cv34_B)
JCudaTensor x190;
JCudaTensor x191, x192, x193;
x191 = x184;
x192 = x194;
x193 = x195;
x190 = x196.forward(x191, x192, x193);

// val X7589 = Convolv(1,0)(X7586,cv32_W,cv32_B)
JCudaTensor x197;
JCudaTensor x198, x199, x200;
x198 = x184;
x199 = x201;
x200 = x202;
x197 = x203.forward(x198, x199, x200);

// val X7587 = Convolv(1,0)(X7586,cv31_W,cv31_B)
JCudaTensor x204;
JCudaTensor x205, x206, x207;
x205 = x184;
x206 = x208;
x207 = x209;
x204 = x210.forward(x205, x206, x207);

// val X7594 = ReLU()(X7593)
JCudaTensor x211;
JCudaTensor x212;
x212 = x190;
x211 = x213.forward(x212);

// val X7590 = ReLU()(X7589)
JCudaTensor x214;
JCudaTensor x215;
x215 = x197;
x214 = x216.forward(x215);

// val X7598 = Convolv(1,0)(X7597,cv36_W,cv36_B)
JCudaTensor x217;
JCudaTensor x218, x219, x220;
x218 = x187;
x219 = x221;
x220 = x222;
x217 = x223.forward(x218, x219, x220);

// val X7595 = Convolv(1,2)(X7594,cv35_W,cv35_B)
JCudaTensor x224;
JCudaTensor x225, x226, x227;
x225 = x211;
x226 = x228;
x227 = x229;
x224 = x230.forward(x225, x226, x227);

// val X7591 = Convolv(1,1)(X7590,cv33_W,cv33_B)
JCudaTensor x231;
JCudaTensor x232, x233, x234;
x232 = x214;
x233 = x235;
x234 = x236;
x231 = x237.forward(x232, x233, x234);

// val X7588 = ReLU()(X7587)
JCudaTensor x238;
JCudaTensor x239;
x239 = x204;
x238 = x240.forward(x239);

// val X7592 = ReLU()(X7591)
JCudaTensor x241;
JCudaTensor x242;
x242 = x231;
x241 = x243.forward(x242);

// val X7596 = ReLU()(X7595)
JCudaTensor x244;
JCudaTensor x245;
x245 = x224;
x244 = x246.forward(x245);

// val X7599 = ReLU()(X7598)
JCudaTensor x247;
JCudaTensor x248;
x248 = x217;
x247 = x246.forward(x248);

// val X7600 = Concat(X7588,X7592,X7596,X7599)
JCudaTensor x249;
JCudaTensor x251, x252, x253, x254;
x251 = x238;
x252 = x241;
x253 = x244;
x254 = x247;
x249 = x250.forward(x251,x252,x253,x254);

// val X7607 = Convolv(1,0)(X7600,cv44_W,cv44_B)
JCudaTensor x255;
JCudaTensor x256, x257, x258;
x256 = x249;
x257 = x259;
x258 = x260;
x255 = x196.forward(x256, x257, x258);

// val X7601 = Convolv(1,0)(X7600,cv41_W,cv41_B)
JCudaTensor x261;
JCudaTensor x262, x263, x264;
x262 = x249;
x263 = x265;
x264 = x266;
x261 = x210.forward(x262, x263, x264);

// val X7706 = Pooling(5,3,0,false)(X7600)
JCudaTensor x267;
JCudaTensor x268;
x268 = x249;
x267 = x269.forward(x268);

// val X7611 = Pooling(3,1,1,true)(X7600)
JCudaTensor x270;
JCudaTensor x271;
x271 = x249;
x270 = x189.forward(x271);

// val X7603 = Convolv(1,0)(X7600,cv42_W,cv42_B)
JCudaTensor x272;
JCudaTensor x273, x274, x275;
x273 = x249;
x274 = x276;
x275 = x277;
x272 = x203.forward(x273, x274, x275);

// val X7604 = ReLU()(X7603)
JCudaTensor x278;
JCudaTensor x279;
x279 = x272;
x278 = x216.forward(x279);

// val X7612 = Convolv(1,0)(X7611,cv46_W,cv46_B)
JCudaTensor x280;
JCudaTensor x281, x282, x283;
x281 = x270;
x282 = x284;
x283 = x285;
x280 = x223.forward(x281, x282, x283);

// val X7707 = Convolv(1,0)(X7706,b1cv_W,b1cv_B)
JCudaTensor x286;
JCudaTensor x287, x288, x289;
x287 = x267;
x288 = x290;
x289 = x291;
x286 = x292.forward(x287, x288, x289);

// val X7608 = ReLU()(X7607)
JCudaTensor x293;
JCudaTensor x294;
x294 = x255;
x293 = x213.forward(x294);

// val X7609 = Convolv(1,2)(X7608,cv45_W,cv45_B)
JCudaTensor x295;
JCudaTensor x296, x297, x298;
x296 = x293;
x297 = x299;
x298 = x300;
x295 = x230.forward(x296, x297, x298);

// val X7708 = ReLU()(X7707)
JCudaTensor x301;
JCudaTensor x302;
x302 = x286;
x301 = x303.forward(x302);

// val X7605 = Convolv(1,1)(X7604,cv43_W,cv43_B)
JCudaTensor x304;
JCudaTensor x305, x306, x307;
x305 = x278;
x306 = x308;
x307 = x309;
x304 = x237.forward(x305, x306, x307);

// val X7602 = ReLU()(X7601)
JCudaTensor x310;
JCudaTensor x311;
x311 = x261;
x310 = x240.forward(x311);

// val X7610 = ReLU()(X7609)
JCudaTensor x312;
JCudaTensor x313;
x313 = x295;
x312 = x246.forward(x313);

// val X7709 = (X7708[1><3])(i21 | @) * (b1fc1_W)(i22 | @)
JCudaTensor x314;
JCudaMatrix x315;
JCudaMatrix x316;
JCudaTensor x317;
JCudaTensor x318;
x318 = x301;
x317 = x318.flatten(1, new int[]{128, 4, 4});
x315 = x317.asMatrix(1, true);
JCudaTensor x319;
x319 = x320;
x316 = x319.asMatrix(1, true);
x314 = x315.times(x316);

// val X7606 = ReLU()(X7605)
JCudaTensor x321;
JCudaTensor x322;
x322 = x304;
x321 = x243.forward(x322);

// val X7613 = ReLU()(X7612)
JCudaTensor x323;
JCudaTensor x324;
x324 = x280;
x323 = x246.forward(x324);

// val X7614 = Concat(X7602,X7606,X7610,X7613)
JCudaTensor x325;
JCudaTensor x326, x327, x328, x329;
x326 = x310;
x327 = x321;
x328 = x312;
x329 = x323;
x325 = x250.forward(x326,x327,x328,x329);

// val X7711 = (X7709 + (i21) => b1fc1_B)
JCudaTensor x330;
JCudaTensor x331, x332;
x331 = x314;
x332 = x333;
x330 = x332.copy(128, x331);

// val X7621 = Convolv(1,0)(X7614,cv54_W,cv54_B)
JCudaTensor x334;
JCudaTensor x335, x336, x337;
x335 = x325;
x336 = x338;
x337 = x339;
x334 = x196.forward(x335, x336, x337);

// val X7617 = Convolv(1,0)(X7614,cv52_W,cv52_B)
JCudaTensor x340;
JCudaTensor x341, x342, x343;
x341 = x325;
x342 = x344;
x343 = x345;
x340 = x203.forward(x341, x342, x343);

// val X7625 = Pooling(3,1,1,true)(X7614)
JCudaTensor x346;
JCudaTensor x347;
x347 = x325;
x346 = x189.forward(x347);

// val X7712 = ReLU()(X7711)
JCudaTensor x348;
JCudaTensor x349;
x349 = x330;
x348 = x350.forward(x349);

// val X7615 = Convolv(1,0)(X7614,cv51_W,cv51_B)
JCudaTensor x351;
JCudaTensor x352, x353, x354;
x352 = x325;
x353 = x355;
x354 = x356;
x351 = x210.forward(x352, x353, x354);

// val X7713 = Dropout(0.7)(X7712)
JCudaTensor x357;
JCudaTensor x358;
x358 = x348;
x357 = x359.forward(x358);

// val X7626 = Convolv(1,0)(X7625,cv56_W,cv56_B)
JCudaTensor x360;
JCudaTensor x361, x362, x363;
x361 = x346;
x362 = x364;
x363 = x365;
x360 = x223.forward(x361, x362, x363);

// val X7618 = ReLU()(X7617)
JCudaTensor x366;
JCudaTensor x367;
x367 = x340;
x366 = x216.forward(x367);

// val X7622 = ReLU()(X7621)
JCudaTensor x368;
JCudaTensor x369;
x369 = x334;
x368 = x213.forward(x369);

// val X7619 = Convolv(1,1)(X7618,cv53_W,cv53_B)
JCudaTensor x370;
JCudaTensor x371, x372, x373;
x371 = x366;
x372 = x374;
x373 = x375;
x370 = x237.forward(x371, x372, x373);

// val X7623 = Convolv(1,2)(X7622,cv55_W,cv55_B)
JCudaTensor x376;
JCudaTensor x377, x378, x379;
x377 = x368;
x378 = x380;
x379 = x381;
x376 = x230.forward(x377, x378, x379);

// val X7714 = (X7713)(i24 | @) * (b1fc2_W)(i25 | @)
JCudaTensor x382;
JCudaMatrix x383;
JCudaMatrix x384;
JCudaTensor x385;
x385 = x357;
x383 = x385.asMatrix(1, true);
JCudaTensor x386;
x386 = x387;
x384 = x386.asMatrix(1, true);
x382 = x383.times(x384);

// val X7620 = ReLU()(X7619)
JCudaTensor x388;
JCudaTensor x389;
x389 = x370;
x388 = x243.forward(x389);

// val X7616 = ReLU()(X7615)
JCudaTensor x390;
JCudaTensor x391;
x391 = x351;
x390 = x240.forward(x391);

// val X7624 = ReLU()(X7623)
JCudaTensor x392;
JCudaTensor x393;
x393 = x376;
x392 = x246.forward(x393);

// val X7627 = ReLU()(X7626)
JCudaTensor x394;
JCudaTensor x395;
x395 = x360;
x394 = x246.forward(x395);

// val X7716 = (X7714 + (i24) => b1fc2_B)
JCudaTensor x396;
JCudaTensor x397, x398;
x397 = x382;
x398 = x399;
x396 = x398.copy(128, x397);

// val X7628 = Concat(X7616,X7620,X7624,X7627)
JCudaTensor x400;
JCudaTensor x401, x402, x403, x404;
x401 = x390;
x402 = x388;
x403 = x392;
x404 = x394;
x400 = x250.forward(x401,x402,x403,x404);

// val X7717 = LogSoftmax()(X7716)
JCudaTensor x405;
JCudaTensor x406;
x406 = x396;
x405 = x407.forward(x406);

// Dealloc(X7716)
JCudaTensor x408;
x408 = x396;
x408.free();

// val X2090 = (X2089 / |128|)
JCudaTensor x409;
JCudaTensor x410;
float x411;
x410 = x18;
float x412;
x412 = 128;
x411 = 1 / x412;
x409 = x410.times_i(x411);

// val X7635 = Convolv(1,0)(X7628,cv64_W,cv64_B)
JCudaTensor x413;
JCudaTensor x414, x415, x416;
x414 = x400;
x415 = x417;
x416 = x418;
x413 = x196.forward(x414, x415, x416);

// val X7629 = Convolv(1,0)(X7628,cv61_W,cv61_B)
JCudaTensor x419;
JCudaTensor x420, x421, x422;
x420 = x400;
x421 = x423;
x422 = x424;
x419 = x210.forward(x420, x421, x422);

// val m4 = (i3357) => b1fc2_W[@, i3357]
JCudaMatrix x425;
JCudaTensor x426;
x426 = x387;
x425 = x426.asMatrix(1, false);

// val X2753 = X2090 * d_LogSoftmax()(X7717)/d_X7716
JCudaTensor x427;
JCudaTensor x428, x429;
x428 = x409;
x429 = x405;
x427 = x407.backward(x428, x429);

// val X7631 = Convolv(1,0)(X7628,cv62_W,cv62_B)
JCudaTensor x430;
JCudaTensor x431, x432, x433;
x431 = x400;
x432 = x434;
x433 = x435;
x430 = x203.forward(x431, x432, x433);

// val X7639 = Pooling(3,1,1,true)(X7628)
JCudaTensor x436;
JCudaTensor x437;
x437 = x400;
x436 = x189.forward(x437);

// val X7636 = ReLU()(X7635)
JCudaTensor x438;
JCudaTensor x439;
x439 = x413;
x438 = x213.forward(x439);

// val X7632 = ReLU()(X7631)
JCudaTensor x440;
JCudaTensor x441;
x441 = x430;
x440 = x216.forward(x441);

// val X2762 = (X2753)(i3356 | @) * m4
JCudaTensor x442;
JCudaMatrix x443;
JCudaMatrix x444;
JCudaTensor x445;
x445 = x427;
x443 = x445.asMatrix(1, true);
x444 = x425;
x442 = x443.times(x444);

// val X7640 = Convolv(1,0)(X7639,cv66_W,cv66_B)
JCudaTensor x446;
JCudaTensor x447, x448, x449;
x447 = x436;
x448 = x450;
x449 = x451;
x446 = x223.forward(x447, x448, x449);

// val m23 = (i1282) => X7713[@, i1282]
JCudaMatrix x452;
JCudaTensor x453;
x453 = x357;
x452 = x453.asMatrix(1, false);

// val m21 = (i1278) => X2753[@, i1278]
JCudaMatrix x454;
JCudaTensor x455;
x455 = x427;
x454 = x455.asMatrix(1, false);

// val X7637 = Convolv(1,2)(X7636,cv65_W,cv65_B)
JCudaTensor x456;
JCudaTensor x457, x458, x459;
x457 = x438;
x458 = x460;
x459 = x461;
x456 = x230.forward(x457, x458, x459);

// val X2763 = X2762 * d_Dropout(0.7)()/d_X7712
JCudaTensor x462;
JCudaTensor x463;
x463 = x442;
x462 = x359.backward(x463);

// Dealloc(X2762)
JCudaTensor x464;
x464 = x442;
x464.free();

// val X7633 = Convolv(1,1)(X7632,cv63_W,cv63_B)
JCudaTensor x465;
JCudaTensor x466, x467, x468;
x466 = x440;
x467 = x469;
x468 = x470;
x465 = x237.forward(x466, x467, x468);

// val X5309 = Sum(m21)
JCudaTensor x471;
JCudaMatrix x472;
x472 = x454;
x471 = x472.sum();

// val X5311 = m21 * m23
JCudaTensor x473;
JCudaMatrix x474;
JCudaMatrix x475;
x474 = x454;
x475 = x452;
x473 = x474.times(x475);

// Dealloc(X2753)
JCudaTensor x476;
x476 = x427;
x476.free();

// Dealloc(X7713)
JCudaTensor x477;
x477 = x357;
x477.free();

// val X5310 = (X5309 * loss1)
JCudaTensor x478;
JCudaTensor x479;
float x480;
x479 = x471;
x480 = loss1;
x478 = x479.times_i(x480);

// val X7638 = ReLU()(X7637)
JCudaTensor x481;
JCudaTensor x482;
x482 = x456;
x481 = x246.forward(x482);

// val m5 = (i3361) => b1fc1_W[@, i3361]
JCudaMatrix x483;
JCudaTensor x484;
x484 = x320;
x483 = x484.asMatrix(1, false);

// val X2765 = X2763 * d_ReLU()(X7712)/d_X7711
JCudaTensor x485;
JCudaTensor x486, x487;
x486 = x462;
x487 = x348;
x485 = x350.backward(x486, x487);

// Dealloc(X7712)
JCudaTensor x488;
x488 = x348;
x488.free();

// val X5312 = (X5311 * loss1)
JCudaTensor x489;
JCudaTensor x490;
float x491;
x490 = x473;
x491 = loss1;
x489 = x490.times_i(x491);

// val X7630 = ReLU()(X7629)
JCudaTensor x492;
JCudaTensor x493;
x493 = x419;
x492 = x240.forward(x493);

// val X7634 = ReLU()(X7633)
JCudaTensor x494;
JCudaTensor x495;
x495 = x465;
x494 = x243.forward(x495);

// val X7641 = ReLU()(X7640)
JCudaTensor x496;
JCudaTensor x497;
x497 = x446;
x496 = x246.forward(x497);

// V_b1fc2_B <~~ X5310
float x499, x500;
float x501;
float x502;
x501 = 2;
x502 = lrn_rate;
x499 = x501 * x502;
x500 = momentum;
JCudaTensor x503;
x503 = x478;
x498.update(x503, x499, x500);

// Dealloc(X5310)
JCudaTensor x504;
x504 = x478;
x504.free();

// val X2766 = (X2765)(i3360 | @) * m5
JCudaTensor x505;
JCudaMatrix x506;
JCudaMatrix x507;
JCudaTensor x508;
x508 = x485;
x506 = x508.asMatrix(1, true);
x507 = x483;
x505 = x506.times(x507);

// val X7642 = Concat(X7630,X7634,X7638,X7641)
JCudaTensor x509;
JCudaTensor x510, x511, x512, x513;
x510 = x492;
x511 = x494;
x512 = x481;
x513 = x496;
x509 = x250.forward(x510,x511,x512,x513);

// val m20 = (i1295) => X7708[1><3][@, i1295]
JCudaMatrix x514;
JCudaTensor x515;
JCudaTensor x516;
x516 = x301;
x515 = x516.flatten(1, new int[]{128, 4, 4});
x514 = x515.asMatrix(1, false);

// val m18 = (i1291) => X2765[@, i1291]
JCudaMatrix x517;
JCudaTensor x518;
x518 = x485;
x517 = x518.asMatrix(1, false);

// V_b1fc2_W <~~ X5312
float x520, x521;
float x522;
float x523;
x522 = 1;
x523 = lrn_rate;
x520 = x522 * x523;
x521 = momentum;
JCudaTensor x524;
x524 = x489;
x519.update(x524, x520, x521);

// Dealloc(X5312)
JCudaTensor x525;
x525 = x489;
x525.free();

// b1fc2_W <~~ V_b1fc2_W
float x526, x527;
x526 = 1;
float x528;
float x529;
x528 = 1;
float x530;
float x531;
float x532;
float x533;
x532 = 1;
x533 = decay;
x530 = x532 * x533;
float x534;
float x535;
x534 = 1;
x535 = lrn_rate;
x531 = x534 * x535;
x529 = x530 * x531;
x527 = x528 + x529;
JCudaTensor x536;
x536 = x519;
x387.update(x536, x526, x527);

// b1fc2_B <~~ V_b1fc2_B
float x537, x538;
x537 = 1;
x538 = 1;
JCudaTensor x539;
x539 = x498;
x399.update(x539, x537, x538);

// val X7643 = Convolv(1,0)(X7642,cv71_W,cv71_B)
JCudaTensor x540;
JCudaTensor x541, x542, x543;
x541 = x509;
x542 = x544;
x543 = x545;
x540 = x210.forward(x541, x542, x543);

// val X5307 = m18 * m20
JCudaTensor x546;
JCudaMatrix x547;
JCudaMatrix x548;
x547 = x517;
x548 = x514;
x546 = x547.times(x548);

// val X5305 = Sum(m18)
JCudaTensor x549;
JCudaMatrix x550;
x550 = x517;
x549 = x550.sum();

// Dealloc(X2765)
JCudaTensor x551;
x551 = x485;
x551.free();

// val X7645 = Convolv(1,0)(X7642,cv72_W,cv72_B)
JCudaTensor x552;
JCudaTensor x553, x554, x555;
x553 = x509;
x554 = x556;
x555 = x557;
x552 = x203.forward(x553, x554, x555);

// val X7649 = Convolv(1,0)(X7642,cv74_W,cv74_B)
JCudaTensor x558;
JCudaTensor x559, x560, x561;
x559 = x509;
x560 = x562;
x561 = x563;
x558 = x196.forward(x559, x560, x561);

// val X2768 = X2766[1<>3] * d_ReLU()(X7708)/d_X7707
JCudaTensor x564;
JCudaTensor x565, x566;
JCudaTensor x567;
x567 = x505;
x565 = x567.unflatten(1, new int[]{128, 4, 4});
x566 = x301;
x564 = x303.backward(x565, x566);

// Dealloc(X7708)
JCudaTensor x568;
x568 = x301;
x568.free();

// val X7653 = Pooling(3,1,1,true)(X7642)
JCudaTensor x569;
JCudaTensor x570;
x570 = x509;
x569 = x189.forward(x570);

// val X7693 = Pooling(5,3,0,false)(X7642)
JCudaTensor x571;
JCudaTensor x572;
x572 = x509;
x571 = x269.forward(x572);

// val X5301 = X2768 * d_Convolv(1,0)()/d_b1cv_B
JCudaTensor x573;
JCudaTensor x574;
x574 = x564;
x573 = x292.backward_bias(x574);

// val X7646 = ReLU()(X7645)
JCudaTensor x575;
JCudaTensor x576;
x576 = x552;
x575 = x216.forward(x576);

// val X7650 = ReLU()(X7649)
JCudaTensor x577;
JCudaTensor x578;
x578 = x558;
x577 = x213.forward(x578);

// val X5306 = (X5305 * loss1)
JCudaTensor x579;
JCudaTensor x580;
float x581;
x580 = x549;
x581 = loss1;
x579 = x580.times_i(x581);

// val X2769 = X2768 * d_Convolv(1,0)(b1cv_W)/d_X7706
JCudaTensor x582;
JCudaTensor x583, x584;
x583 = x564;
x584 = x290;
x582 = x292.backward_data(x583, x584);

// val X5308 = (X5307 * loss1)
JCudaTensor x585;
JCudaTensor x586;
float x587;
x586 = x546;
x587 = loss1;
x585 = x586.times_i(x587);

// val X7654 = Convolv(1,0)(X7653,cv76_W,cv76_B)
JCudaTensor x588;
JCudaTensor x589, x590, x591;
x589 = x569;
x590 = x592;
x591 = x593;
x588 = x223.forward(x589, x590, x591);

// val X5303 = X2768 * d_Convolv(1,0)(X7706)/d_b1cv_W
JCudaTensor x594;
JCudaTensor x595, x596;
x595 = x564;
x596 = x267;
x594 = x292.backward_filter(x595, x596);

// Dealloc(X2768)
JCudaTensor x597;
x597 = x564;
x597.free();

// val X7694 = Convolv(1,0)(X7693,b2cv_W,b2cv_B)
JCudaTensor x598;
JCudaTensor x599, x600, x601;
x599 = x571;
x600 = x602;
x601 = x603;
x598 = x292.forward(x599, x600, x601);

// val X2771 = X2769 * d_Pooling(5,3,0,false)(X7706,X7600)/d_X7600
JCudaTensor x604;
JCudaTensor x605, x606, x607;
x605 = x582;
x606 = x267;
x607 = x249;
x604 = x269.backward(x605, x606, x607);

// Dealloc(X2769)
JCudaTensor x608;
x608 = x582;
x608.free();

// Dealloc(X7706)
JCudaTensor x609;
x609 = x267;
x609.free();

// V_b1fc1_W <~~ X5308
float x611, x612;
float x613;
float x614;
x613 = 1;
x614 = lrn_rate;
x611 = x613 * x614;
x612 = momentum;
JCudaTensor x615;
x615 = x585;
x610.update(x615, x611, x612);

// Dealloc(X5308)
JCudaTensor x616;
x616 = x585;
x616.free();

// val X7651 = Convolv(1,2)(X7650,cv75_W,cv75_B)
JCudaTensor x617;
JCudaTensor x618, x619, x620;
x618 = x577;
x619 = x621;
x620 = x622;
x617 = x230.forward(x618, x619, x620);

// val X7647 = Convolv(1,1)(X7646,cv73_W,cv73_B)
JCudaTensor x623;
JCudaTensor x624, x625, x626;
x624 = x575;
x625 = x627;
x626 = x628;
x623 = x237.forward(x624, x625, x626);

// V_b1fc1_B <~~ X5306
float x630, x631;
float x632;
float x633;
x632 = 2;
x633 = lrn_rate;
x630 = x632 * x633;
x631 = momentum;
JCudaTensor x634;
x634 = x579;
x629.update(x634, x630, x631);

// Dealloc(X5306)
JCudaTensor x635;
x635 = x579;
x635.free();

// val X5304 = (X5303 * loss1)
JCudaTensor x636;
JCudaTensor x637;
float x638;
x637 = x594;
x638 = loss1;
x636 = x637.times_i(x638);

// val X5302 = (X5301 * loss1)
JCudaTensor x639;
JCudaTensor x640;
float x641;
x640 = x573;
x641 = loss1;
x639 = x640.times_i(x641);

// val X7695 = ReLU()(X7694)
JCudaTensor x642;
JCudaTensor x643;
x643 = x598;
x642 = x303.forward(x643);

// b1fc1_B <~~ V_b1fc1_B
float x644, x645;
x644 = 1;
x645 = 1;
JCudaTensor x646;
x646 = x629;
x333.update(x646, x644, x645);

// b1fc1_W <~~ V_b1fc1_W
float x647, x648;
x647 = 1;
float x649;
float x650;
x649 = 1;
float x651;
float x652;
float x653;
float x654;
x653 = 1;
x654 = decay;
x651 = x653 * x654;
float x655;
float x656;
x655 = 1;
x656 = lrn_rate;
x652 = x655 * x656;
x650 = x651 * x652;
x648 = x649 + x650;
JCudaTensor x657;
x657 = x610;
x320.update(x657, x647, x648);

// val X7644 = ReLU()(X7643)
JCudaTensor x658;
JCudaTensor x659;
x659 = x540;
x658 = x240.forward(x659);

// val X7648 = ReLU()(X7647)
JCudaTensor x660;
JCudaTensor x661;
x661 = x623;
x660 = x243.forward(x661);

// V_b1cv_B <~~ X5302
float x663, x664;
float x665;
float x666;
x665 = 2;
x666 = lrn_rate;
x663 = x665 * x666;
x664 = momentum;
JCudaTensor x667;
x667 = x639;
x662.update(x667, x663, x664);

// Dealloc(X5302)
JCudaTensor x668;
x668 = x639;
x668.free();

// val X7652 = ReLU()(X7651)
JCudaTensor x669;
JCudaTensor x670;
x670 = x617;
x669 = x246.forward(x670);

// val X7655 = ReLU()(X7654)
JCudaTensor x671;
JCudaTensor x672;
x672 = x588;
x671 = x246.forward(x672);

// val X7696 = (X7695[1><3])(i15 | @) * (b2fc1_W)(i16 | @)
JCudaTensor x673;
JCudaMatrix x674;
JCudaMatrix x675;
JCudaTensor x676;
JCudaTensor x677;
x677 = x642;
x676 = x677.flatten(1, new int[]{128, 4, 4});
x674 = x676.asMatrix(1, true);
JCudaTensor x678;
x678 = x679;
x675 = x678.asMatrix(1, true);
x673 = x674.times(x675);

// V_b1cv_W <~~ X5304
float x681, x682;
float x683;
float x684;
x683 = 1;
x684 = lrn_rate;
x681 = x683 * x684;
x682 = momentum;
JCudaTensor x685;
x685 = x636;
x680.update(x685, x681, x682);

// Dealloc(X5304)
JCudaTensor x686;
x686 = x636;
x686.free();

// b1cv_W <~~ V_b1cv_W
float x687, x688;
x687 = 1;
float x689;
float x690;
x689 = 1;
float x691;
float x692;
float x693;
float x694;
x693 = 1;
x694 = decay;
x691 = x693 * x694;
float x695;
float x696;
x695 = 1;
x696 = lrn_rate;
x692 = x695 * x696;
x690 = x691 * x692;
x688 = x689 + x690;
JCudaTensor x697;
x697 = x680;
x290.update(x697, x687, x688);

// b1cv_B <~~ V_b1cv_B
float x698, x699;
x698 = 1;
x699 = 1;
JCudaTensor x700;
x700 = x662;
x291.update(x700, x698, x699);

// val X7656 = Concat(X7644,X7648,X7652,X7655)
JCudaTensor x701;
JCudaTensor x702, x703, x704, x705;
x702 = x658;
x703 = x660;
x704 = x669;
x705 = x671;
x701 = x250.forward(x702,x703,x704,x705);

// val X7698 = (X7696 + (i15) => b2fc1_B)
JCudaTensor x706;
JCudaTensor x707, x708;
x707 = x673;
x708 = x709;
x706 = x708.copy(128, x707);

// val X7699 = ReLU()(X7698)
JCudaTensor x710;
JCudaTensor x711;
x711 = x706;
x710 = x350.forward(x711);

// val X7657 = Pooling(3,2,1,true)(X7656)
JCudaTensor x712;
JCudaTensor x713;
x713 = x701;
x712 = x714.forward(x713);

// val X7700 = Dropout(0.7)(X7699)
JCudaTensor x715;
JCudaTensor x716;
x716 = x710;
x715 = x359.forward(x716);

// val X7668 = Pooling(3,1,1,true)(X7657)
JCudaTensor x717;
JCudaTensor x718;
x718 = x712;
x717 = x719.forward(x718);

// val X7664 = Convolv(1,0)(X7657,cv84_W,cv84_B)
JCudaTensor x720;
JCudaTensor x721, x722, x723;
x721 = x712;
x722 = x724;
x723 = x725;
x720 = x726.forward(x721, x722, x723);

// val X7658 = Convolv(1,0)(X7657,cv81_W,cv81_B)
JCudaTensor x727;
JCudaTensor x728, x729, x730;
x728 = x712;
x729 = x731;
x730 = x732;
x727 = x733.forward(x728, x729, x730);

// val X7660 = Convolv(1,0)(X7657,cv82_W,cv82_B)
JCudaTensor x734;
JCudaTensor x735, x736, x737;
x735 = x712;
x736 = x738;
x737 = x739;
x734 = x740.forward(x735, x736, x737);

// val X7661 = ReLU()(X7660)
JCudaTensor x741;
JCudaTensor x742;
x742 = x734;
x741 = x743.forward(x742);

// val X7701 = (X7700)(i18 | @) * (b2fc2_W)(i19 | @)
JCudaTensor x744;
JCudaMatrix x745;
JCudaMatrix x746;
JCudaTensor x747;
x747 = x715;
x745 = x747.asMatrix(1, true);
JCudaTensor x748;
x748 = x749;
x746 = x748.asMatrix(1, true);
x744 = x745.times(x746);

// val X7669 = Convolv(1,0)(X7668,cv86_W,cv86_B)
JCudaTensor x750;
JCudaTensor x751, x752, x753;
x751 = x717;
x752 = x754;
x753 = x755;
x750 = x756.forward(x751, x752, x753);

// val X7665 = ReLU()(X7664)
JCudaTensor x757;
JCudaTensor x758;
x758 = x720;
x757 = x759.forward(x758);

// val X7662 = Convolv(1,1)(X7661,cv83_W,cv83_B)
JCudaTensor x760;
JCudaTensor x761, x762, x763;
x761 = x741;
x762 = x764;
x763 = x765;
x760 = x766.forward(x761, x762, x763);

// val X7666 = Convolv(1,2)(X7665,cv85_W,cv85_B)
JCudaTensor x767;
JCudaTensor x768, x769, x770;
x768 = x757;
x769 = x771;
x770 = x772;
x767 = x773.forward(x768, x769, x770);

// val X7703 = (X7701 + (i18) => b2fc2_B)
JCudaTensor x774;
JCudaTensor x775, x776;
x775 = x744;
x776 = x777;
x774 = x776.copy(128, x775);

// val X7663 = ReLU()(X7662)
JCudaTensor x778;
JCudaTensor x779;
x779 = x760;
x778 = x780.forward(x779);

// val X7667 = ReLU()(X7666)
JCudaTensor x781;
JCudaTensor x782;
x782 = x767;
x781 = x783.forward(x782);

// val X7670 = ReLU()(X7669)
JCudaTensor x784;
JCudaTensor x785;
x785 = x750;
x784 = x783.forward(x785);

// val X7659 = ReLU()(X7658)
JCudaTensor x786;
JCudaTensor x787;
x787 = x727;
x786 = x788.forward(x787);

// val X7704 = LogSoftmax()(X7703)
JCudaTensor x789;
JCudaTensor x790;
x790 = x774;
x789 = x407.forward(x790);

// Dealloc(X7703)
JCudaTensor x791;
x791 = x774;
x791.free();

// val X7671 = Concat(X7659,X7663,X7667,X7670)
JCudaTensor x792;
JCudaTensor x794, x795, x796, x797;
x794 = x786;
x795 = x778;
x796 = x781;
x797 = x784;
x792 = x793.forward(x794,x795,x796,x797);

// val X2424 = X2090 * d_LogSoftmax()(X7704)/d_X7703
JCudaTensor x798;
JCudaTensor x799, x800;
x799 = x409;
x800 = x789;
x798 = x407.backward(x799, x800);

// val m2 = (i3173) => b2fc2_W[@, i3173]
JCudaMatrix x801;
JCudaTensor x802;
x802 = x749;
x801 = x802.asMatrix(1, false);

// val X7672 = Convolv(1,0)(X7671,cv91_W,cv91_B)
JCudaTensor x803;
JCudaTensor x804, x805, x806;
x804 = x792;
x805 = x807;
x806 = x808;
x803 = x733.forward(x804, x805, x806);

// val X2433 = (X2424)(i3172 | @) * m2
JCudaTensor x809;
JCudaMatrix x810;
JCudaMatrix x811;
JCudaTensor x812;
x812 = x798;
x810 = x812.asMatrix(1, true);
x811 = x801;
x809 = x810.times(x811);

// val X7678 = Convolv(1,0)(X7671,cv94_W,cv94_B)
JCudaTensor x813;
JCudaTensor x814, x815, x816;
x814 = x792;
x815 = x817;
x816 = x818;
x813 = x726.forward(x814, x815, x816);

// val m29 = (i506) => X7700[@, i506]
JCudaMatrix x819;
JCudaTensor x820;
x820 = x715;
x819 = x820.asMatrix(1, false);

// val X7682 = Pooling(3,1,1,true)(X7671)
JCudaTensor x821;
JCudaTensor x822;
x822 = x792;
x821 = x719.forward(x822);

// val X7674 = Convolv(1,0)(X7671,cv92_W,cv92_B)
JCudaTensor x823;
JCudaTensor x824, x825, x826;
x824 = x792;
x825 = x827;
x826 = x828;
x823 = x740.forward(x824, x825, x826);

// val m27 = (i502) => X2424[@, i502]
JCudaMatrix x829;
JCudaTensor x830;
x830 = x798;
x829 = x830.asMatrix(1, false);

// val X2434 = X2433 * d_Dropout(0.7)()/d_X7699
JCudaTensor x831;
JCudaTensor x832;
x832 = x809;
x831 = x359.backward(x832);

// Dealloc(X2433)
JCudaTensor x833;
x833 = x809;
x833.free();

// val X7679 = ReLU()(X7678)
JCudaTensor x834;
JCudaTensor x835;
x835 = x813;
x834 = x759.forward(x835);

// val X7675 = ReLU()(X7674)
JCudaTensor x836;
JCudaTensor x837;
x837 = x823;
x836 = x743.forward(x837);

// val X5321 = Sum(m27)
JCudaTensor x838;
JCudaMatrix x839;
x839 = x829;
x838 = x839.sum();

// val X7683 = Convolv(1,0)(X7682,cv96_W,cv96_B)
JCudaTensor x840;
JCudaTensor x841, x842, x843;
x841 = x821;
x842 = x844;
x843 = x845;
x840 = x756.forward(x841, x842, x843);

// val X5323 = m27 * m29
JCudaTensor x846;
JCudaMatrix x847;
JCudaMatrix x848;
x847 = x829;
x848 = x819;
x846 = x847.times(x848);

// Dealloc(X2424)
JCudaTensor x849;
x849 = x798;
x849.free();

// Dealloc(X7700)
JCudaTensor x850;
x850 = x715;
x850.free();

// val X7676 = Convolv(1,1)(X7675,cv93_W,cv93_B)
JCudaTensor x851;
JCudaTensor x852, x853, x854;
x852 = x836;
x853 = x855;
x854 = x856;
x851 = x766.forward(x852, x853, x854);

// val m3 = (i3177) => b2fc1_W[@, i3177]
JCudaMatrix x857;
JCudaTensor x858;
x858 = x679;
x857 = x858.asMatrix(1, false);

// val X5324 = (X5323 * loss2)
JCudaTensor x859;
JCudaTensor x860;
float x861;
x860 = x846;
x861 = loss2;
x859 = x860.times_i(x861);

// val X2436 = X2434 * d_ReLU()(X7699)/d_X7698
JCudaTensor x862;
JCudaTensor x863, x864;
x863 = x831;
x864 = x710;
x862 = x350.backward(x863, x864);

// Dealloc(X7699)
JCudaTensor x865;
x865 = x710;
x865.free();

// val X7680 = Convolv(1,2)(X7679,cv95_W,cv95_B)
JCudaTensor x866;
JCudaTensor x867, x868, x869;
x867 = x834;
x868 = x870;
x869 = x871;
x866 = x773.forward(x867, x868, x869);

// val X5322 = (X5321 * loss2)
JCudaTensor x872;
JCudaTensor x873;
float x874;
x873 = x838;
x874 = loss2;
x872 = x873.times_i(x874);

// val m26 = (i519) => X7695[1><3][@, i519]
JCudaMatrix x875;
JCudaTensor x876;
JCudaTensor x877;
x877 = x642;
x876 = x877.flatten(1, new int[]{128, 4, 4});
x875 = x876.asMatrix(1, false);

// val X7681 = ReLU()(X7680)
JCudaTensor x878;
JCudaTensor x879;
x879 = x866;
x878 = x783.forward(x879);

// V_b2fc2_W <~~ X5324
float x881, x882;
float x883;
float x884;
x883 = 1;
x884 = lrn_rate;
x881 = x883 * x884;
x882 = momentum;
JCudaTensor x885;
x885 = x859;
x880.update(x885, x881, x882);

// Dealloc(X5324)
JCudaTensor x886;
x886 = x859;
x886.free();

// V_b2fc2_B <~~ X5322
float x888, x889;
float x890;
float x891;
x890 = 2;
x891 = lrn_rate;
x888 = x890 * x891;
x889 = momentum;
JCudaTensor x892;
x892 = x872;
x887.update(x892, x888, x889);

// Dealloc(X5322)
JCudaTensor x893;
x893 = x872;
x893.free();

// val X7684 = ReLU()(X7683)
JCudaTensor x894;
JCudaTensor x895;
x895 = x840;
x894 = x783.forward(x895);

// val m24 = (i515) => X2436[@, i515]
JCudaMatrix x896;
JCudaTensor x897;
x897 = x862;
x896 = x897.asMatrix(1, false);

// val X7673 = ReLU()(X7672)
JCudaTensor x898;
JCudaTensor x899;
x899 = x803;
x898 = x788.forward(x899);

// val X7677 = ReLU()(X7676)
JCudaTensor x900;
JCudaTensor x901;
x901 = x851;
x900 = x780.forward(x901);

// val X2437 = (X2436)(i3176 | @) * m3
JCudaTensor x902;
JCudaMatrix x903;
JCudaMatrix x904;
JCudaTensor x905;
x905 = x862;
x903 = x905.asMatrix(1, true);
x904 = x857;
x902 = x903.times(x904);

// b2fc2_B <~~ V_b2fc2_B
float x906, x907;
x906 = 1;
x907 = 1;
JCudaTensor x908;
x908 = x887;
x777.update(x908, x906, x907);

// b2fc2_W <~~ V_b2fc2_W
float x909, x910;
x909 = 1;
float x911;
float x912;
x911 = 1;
float x913;
float x914;
float x915;
float x916;
x915 = 1;
x916 = decay;
x913 = x915 * x916;
float x917;
float x918;
x917 = 1;
x918 = lrn_rate;
x914 = x917 * x918;
x912 = x913 * x914;
x910 = x911 + x912;
JCudaTensor x919;
x919 = x880;
x749.update(x919, x909, x910);

// val X7685 = Concat(X7673,X7677,X7681,X7684)
JCudaTensor x920;
JCudaTensor x921, x922, x923, x924;
x921 = x898;
x922 = x900;
x923 = x878;
x924 = x894;
x920 = x793.forward(x921,x922,x923,x924);

// val X5319 = m24 * m26
JCudaTensor x925;
JCudaMatrix x926;
JCudaMatrix x927;
x926 = x896;
x927 = x875;
x925 = x926.times(x927);

// val X5317 = Sum(m24)
JCudaTensor x928;
JCudaMatrix x929;
x929 = x896;
x928 = x929.sum();

// Dealloc(X2436)
JCudaTensor x930;
x930 = x862;
x930.free();

// val X2439 = X2437[1<>3] * d_ReLU()(X7695)/d_X7694
JCudaTensor x931;
JCudaTensor x932, x933;
JCudaTensor x934;
x934 = x902;
x932 = x934.unflatten(1, new int[]{128, 4, 4});
x933 = x642;
x931 = x303.backward(x932, x933);

// Dealloc(X7695)
JCudaTensor x935;
x935 = x642;
x935.free();

// val X7686 = Pooling(7,1,0,false)(X7685)
JCudaTensor x936;
JCudaTensor x937;
x937 = x920;
x936 = x938.forward(x937);

// val X5320 = (X5319 * loss2)
JCudaTensor x939;
JCudaTensor x940;
float x941;
x940 = x925;
x941 = loss2;
x939 = x940.times_i(x941);

// val X5318 = (X5317 * loss2)
JCudaTensor x942;
JCudaTensor x943;
float x944;
x943 = x928;
x944 = loss2;
x942 = x943.times_i(x944);

// val X5315 = X2439 * d_Convolv(1,0)(X7693)/d_b2cv_W
JCudaTensor x945;
JCudaTensor x946, x947;
x946 = x931;
x947 = x571;
x945 = x292.backward_filter(x946, x947);

// val X5313 = X2439 * d_Convolv(1,0)()/d_b2cv_B
JCudaTensor x948;
JCudaTensor x949;
x949 = x931;
x948 = x292.backward_bias(x949);

// val X2440 = X2439 * d_Convolv(1,0)(b2cv_W)/d_X7693
JCudaTensor x950;
JCudaTensor x951, x952;
x951 = x931;
x952 = x602;
x950 = x292.backward_data(x951, x952);

// Dealloc(X2439)
JCudaTensor x953;
x953 = x931;
x953.free();

// val X5316 = (X5315 * loss2)
JCudaTensor x954;
JCudaTensor x955;
float x956;
x955 = x945;
x956 = loss2;
x954 = x955.times_i(x956);

// V_b2fc1_W <~~ X5320
float x958, x959;
float x960;
float x961;
x960 = 1;
x961 = lrn_rate;
x958 = x960 * x961;
x959 = momentum;
JCudaTensor x962;
x962 = x939;
x957.update(x962, x958, x959);

// Dealloc(X5320)
JCudaTensor x963;
x963 = x939;
x963.free();

// val X5314 = (X5313 * loss2)
JCudaTensor x964;
JCudaTensor x965;
float x966;
x965 = x948;
x966 = loss2;
x964 = x965.times_i(x966);

// val X2442 = X2440 * d_Pooling(5,3,0,false)(X7693,X7642)/d_X7642
JCudaTensor x967;
JCudaTensor x968, x969, x970;
x968 = x950;
x969 = x571;
x970 = x509;
x967 = x269.backward(x968, x969, x970);

// Dealloc(X2440)
JCudaTensor x971;
x971 = x950;
x971.free();

// Dealloc(X7693)
JCudaTensor x972;
x972 = x571;
x972.free();

// val X7687 = Dropout(0.4)(X7686)
JCudaTensor x973;
JCudaTensor x974;
x974 = x936;
x973 = x975.forward(x974);

// V_b2fc1_B <~~ X5318
float x977, x978;
float x979;
float x980;
x979 = 2;
x980 = lrn_rate;
x977 = x979 * x980;
x978 = momentum;
JCudaTensor x981;
x981 = x942;
x976.update(x981, x977, x978);

// Dealloc(X5318)
JCudaTensor x982;
x982 = x942;
x982.free();

// b2fc1_B <~~ V_b2fc1_B
float x983, x984;
x983 = 1;
x984 = 1;
JCudaTensor x985;
x985 = x976;
x709.update(x985, x983, x984);

// b2fc1_W <~~ V_b2fc1_W
float x986, x987;
x986 = 1;
float x988;
float x989;
x988 = 1;
float x990;
float x991;
float x992;
float x993;
x992 = 1;
x993 = decay;
x990 = x992 * x993;
float x994;
float x995;
x994 = 1;
x995 = lrn_rate;
x991 = x994 * x995;
x989 = x990 * x991;
x987 = x988 + x989;
JCudaTensor x996;
x996 = x957;
x679.update(x996, x986, x987);

// V_b2cv_W <~~ X5316
float x998, x999;
float x1000;
float x1001;
x1000 = 1;
x1001 = lrn_rate;
x998 = x1000 * x1001;
x999 = momentum;
JCudaTensor x1002;
x1002 = x954;
x997.update(x1002, x998, x999);

// Dealloc(X5316)
JCudaTensor x1003;
x1003 = x954;
x1003.free();

// val X7688 = (X7687[1><3])(i12 | @) * (fc_W)(i13 | @)
JCudaTensor x1004;
JCudaMatrix x1005;
JCudaMatrix x1006;
JCudaTensor x1007;
JCudaTensor x1008;
x1008 = x973;
x1007 = x1008.flatten(1, new int[]{256, 1, 1});
x1005 = x1007.asMatrix(1, true);
JCudaTensor x1009;
x1009 = x1010;
x1006 = x1009.asMatrix(1, true);
x1004 = x1005.times(x1006);

// V_b2cv_B <~~ X5314
float x1012, x1013;
float x1014;
float x1015;
x1014 = 2;
x1015 = lrn_rate;
x1012 = x1014 * x1015;
x1013 = momentum;
JCudaTensor x1016;
x1016 = x964;
x1011.update(x1016, x1012, x1013);

// Dealloc(X5314)
JCudaTensor x1017;
x1017 = x964;
x1017.free();

// b2cv_W <~~ V_b2cv_W
float x1018, x1019;
x1018 = 1;
float x1020;
float x1021;
x1020 = 1;
float x1022;
float x1023;
float x1024;
float x1025;
x1024 = 1;
x1025 = decay;
x1022 = x1024 * x1025;
float x1026;
float x1027;
x1026 = 1;
x1027 = lrn_rate;
x1023 = x1026 * x1027;
x1021 = x1022 * x1023;
x1019 = x1020 + x1021;
JCudaTensor x1028;
x1028 = x997;
x602.update(x1028, x1018, x1019);

// b2cv_B <~~ V_b2cv_B
float x1029, x1030;
x1029 = 1;
x1030 = 1;
JCudaTensor x1031;
x1031 = x1011;
x603.update(x1031, x1029, x1030);

// val X7690 = (X7688 + (i12) => fc_B)
JCudaTensor x1032;
JCudaTensor x1033, x1034;
x1033 = x1004;
x1034 = x1035;
x1032 = x1034.copy(128, x1033);

// val X7691 = LogSoftmax()(X7690)
JCudaTensor x1036;
JCudaTensor x1037;
x1037 = x1032;
x1036 = x407.forward(x1037);

// Dealloc(X7690)
JCudaTensor x1038;
x1038 = x1032;
x1038.free();

// Cost(((((0 - (X7692 . X7691)) / |128|) + (((0 - (X7692 . X7704)) / |128|) * loss2)) + (((0 - (X7692 . X7717)) / |128|) * loss1)))
float x1039;
float x1040;
float x1041;
float x1042;
float x1043;
float x1044;
float x1045;
float x1046;
JCudaTensor x1047, x1048;
x1047 = x9;
x1048 = x1036;
x1046 = x1047.dot(x1048);
x1044 = - x1046;
x1045 = 128;
x1042 = x1044 / x1045;
float x1049;
float x1050;
float x1051;
float x1052;
float x1053;
JCudaTensor x1054, x1055;
x1054 = x9;
x1055 = x789;
x1053 = x1054.dot(x1055);
x1051 = - x1053;
x1052 = 128;
x1049 = x1051 / x1052;
x1050 = loss2;
x1043 = x1049 * x1050;
x1040 = x1042 + x1043;
float x1056;
float x1057;
float x1058;
float x1059;
float x1060;
JCudaTensor x1061, x1062;
x1061 = x9;
x1062 = x405;
x1060 = x1061.dot(x1062);
x1058 = - x1060;
x1059 = 128;
x1056 = x1058 / x1059;
x1057 = loss1;
x1041 = x1056 * x1057;
x1039 = x1040 + x1041;
System.out.println(x5 + " " + x1039);
if (Float.isNaN(x1039)) { System.exit(-1); }

// Dealloc(X7704)
JCudaTensor x1063;
x1063 = x789;
x1063.free();

// Dealloc(X7717)
JCudaTensor x1064;
x1064 = x405;
x1064.free();

// Dealloc(X7692)
JCudaTensor x1065;
x1065 = x9;
x1065.free();

// val X2092 = X2090 * d_LogSoftmax()(X7691)/d_X7690
JCudaTensor x1066;
JCudaTensor x1067, x1068;
x1067 = x409;
x1068 = x1036;
x1066 = x407.backward(x1067, x1068);

// Dealloc(X7691)
JCudaTensor x1069;
x1069 = x1036;
x1069.free();

// Dealloc(X2090)
JCudaTensor x1070;
x1070 = x409;
x1070.free();

// val m1 = (i3005) => fc_W[@, i3005]
JCudaMatrix x1071;
JCudaTensor x1072;
x1072 = x1010;
x1071 = x1072.asMatrix(1, false);

// val X2138 = (X2092)(i3004 | @) * m1
JCudaTensor x1073;
JCudaMatrix x1074;
JCudaMatrix x1075;
JCudaTensor x1076;
x1076 = x1066;
x1074 = x1076.asMatrix(1, true);
x1075 = x1071;
x1073 = x1074.times(x1075);

// val m30 = (i31) => X2092[@, i31]
JCudaMatrix x1077;
JCudaTensor x1078;
x1078 = x1066;
x1077 = x1078.asMatrix(1, false);

// val m32 = (i35) => X7687[1><3][@, i35]
JCudaMatrix x1079;
JCudaTensor x1080;
JCudaTensor x1081;
x1081 = x973;
x1080 = x1081.flatten(1, new int[]{256, 1, 1});
x1079 = x1080.asMatrix(1, false);

// val X2139 = X2138[1<>3] * d_Dropout(0.4)()/d_X7686
JCudaTensor x1082;
JCudaTensor x1083;
JCudaTensor x1084;
x1084 = x1073;
x1083 = x1084.unflatten(1, new int[]{256, 1, 1});
x1082 = x975.backward(x1083);

// Dealloc(X2138)
JCudaTensor x1085;
x1085 = x1073;
x1085.free();

// V_fc_W <~~ m30 * m32
float x1087, x1088;
float x1089;
float x1090;
x1089 = 1;
x1090 = lrn_rate;
x1087 = x1089 * x1090;
x1088 = momentum;
JCudaMatrix x1091;
JCudaMatrix x1092;
x1091 = x1077;
x1092 = x1079;
x1091.times(x1092, x1086, x1087, x1088);

// Dealloc(X7687)
JCudaTensor x1093;
x1093 = x973;
x1093.free();

// V_fc_B <~~ Sum(m30)
float x1095, x1096;
float x1097;
float x1098;
x1097 = 2;
x1098 = lrn_rate;
x1095 = x1097 * x1098;
x1096 = momentum;
JCudaMatrix x1099;
x1099 = x1077;
x1099.sum(x1094, x1095, x1096);

// Dealloc(X2092)
JCudaTensor x1100;
x1100 = x1066;
x1100.free();

// fc_W <~~ V_fc_W
float x1101, x1102;
x1101 = 1;
float x1103;
float x1104;
x1103 = 1;
float x1105;
float x1106;
float x1107;
float x1108;
x1107 = 1;
x1108 = decay;
x1105 = x1107 * x1108;
float x1109;
float x1110;
x1109 = 1;
x1110 = lrn_rate;
x1106 = x1109 * x1110;
x1104 = x1105 * x1106;
x1102 = x1103 + x1104;
JCudaTensor x1111;
x1111 = x1086;
x1010.update(x1111, x1101, x1102);

// fc_B <~~ V_fc_B
float x1112, x1113;
x1112 = 1;
x1113 = 1;
JCudaTensor x1114;
x1114 = x1094;
x1035.update(x1114, x1112, x1113);

// val X2141 = X2139 * d_Pooling(7,1,0,false)(X7686,X7685)/d_X7685
JCudaTensor x1115;
JCudaTensor x1116, x1117, x1118;
x1116 = x1082;
x1117 = x936;
x1118 = x920;
x1115 = x938.backward(x1116, x1117, x1118);

// Dealloc(X2139)
JCudaTensor x1119;
x1119 = x1082;
x1119.free();

// Dealloc(X7686)
JCudaTensor x1120;
x1120 = x936;
x1120.free();

// Dealloc(X7685)
JCudaTensor x1121;
x1121 = x920;
x1121.free();

// val X2155 = Proj(X2141, X7673,X7677,X7681,X7684, 0)
JCudaTensor x1122;
JCudaTensor x1124;
x1124 = x1115;
JCudaTensor[] x1123 = x793.backward(x1124);
x1122 = x1123[0];

// val X2197 = Proj(X2141, X7673,X7677,X7681,X7684, 2)
JCudaTensor x1125;
x1125 = x1123[2];

// val X2173 = Proj(X2141, X7673,X7677,X7681,X7684, 1)
JCudaTensor x1126;
x1126 = x1123[1];

// val X2221 = Proj(X2141, X7673,X7677,X7681,X7684, 3)
JCudaTensor x1127;
x1127 = x1123[3];

// Dealloc(X2141)
JCudaTensor x1128;
x1128 = x1115;
x1128.free();

// val X2202 = X2197 * d_ReLU()(X7681)/d_X7680
JCudaTensor x1129;
JCudaTensor x1130, x1131;
x1130 = x1125;
x1131 = x878;
x1129 = x783.backward(x1130, x1131);

// Dealloc(X7681)
JCudaTensor x1132;
x1132 = x878;
x1132.free();

// val X2158 = X2155 * d_ReLU()(X7673)/d_X7672
JCudaTensor x1133;
JCudaTensor x1134, x1135;
x1134 = x1122;
x1135 = x898;
x1133 = x788.backward(x1134, x1135);

// Dealloc(X7673)
JCudaTensor x1136;
x1136 = x898;
x1136.free();

// val X2178 = X2173 * d_ReLU()(X7677)/d_X7676
JCudaTensor x1137;
JCudaTensor x1138, x1139;
x1138 = x1126;
x1139 = x900;
x1137 = x780.backward(x1138, x1139);

// Dealloc(X7677)
JCudaTensor x1140;
x1140 = x900;
x1140.free();

// val X2225 = X2221 * d_ReLU()(X7684)/d_X7683
JCudaTensor x1141;
JCudaTensor x1142, x1143;
x1142 = x1127;
x1143 = x894;
x1141 = x783.backward(x1142, x1143);

// Dealloc(X7684)
JCudaTensor x1144;
x1144 = x894;
x1144.free();

// V_cv95_W <~~ X2202 * d_Convolv(1,2)(X7679)/d_cv95_W
float x1146, x1147;
float x1148;
float x1149;
x1148 = 1;
x1149 = lrn_rate;
x1146 = x1148 * x1149;
x1147 = momentum;
JCudaTensor x1150, x1151;
x1150 = x1129;
x1151 = x834;
x773.backward_filter(x1150, x1151, x1145, x1146, x1147);

// val X2159 = X2158 * d_Convolv(1,0)(cv91_W)/d_X7671
JCudaTensor x1152;
JCudaTensor x1153, x1154;
x1153 = x1133;
x1154 = x807;
x1152 = x733.backward_data(x1153, x1154);

// val X2179 = X2178 * d_Convolv(1,1)(cv93_W)/d_X7675
JCudaTensor x1155;
JCudaTensor x1156, x1157;
x1156 = x1137;
x1157 = x855;
x1155 = x766.backward_data(x1156, x1157);

// val X2203 = X2202 * d_Convolv(1,2)(cv95_W)/d_X7679
JCudaTensor x1158;
JCudaTensor x1159, x1160;
x1159 = x1129;
x1160 = x870;
x1158 = x773.backward_data(x1159, x1160);

// V_cv91_B <~~ X2158 * d_Convolv(1,0)()/d_cv91_B
float x1162, x1163;
float x1164;
float x1165;
x1164 = 2;
x1165 = lrn_rate;
x1162 = x1164 * x1165;
x1163 = momentum;
JCudaTensor x1166;
x1166 = x1133;
x733.backward_bias(x1166, x1161, x1162, x1163);

// V_cv96_W <~~ X2225 * d_Convolv(1,0)(X7682)/d_cv96_W
float x1168, x1169;
float x1170;
float x1171;
x1170 = 1;
x1171 = lrn_rate;
x1168 = x1170 * x1171;
x1169 = momentum;
JCudaTensor x1172, x1173;
x1172 = x1141;
x1173 = x821;
x756.backward_filter(x1172, x1173, x1167, x1168, x1169);

// V_cv95_B <~~ X2202 * d_Convolv(1,2)()/d_cv95_B
float x1175, x1176;
float x1177;
float x1178;
x1177 = 2;
x1178 = lrn_rate;
x1175 = x1177 * x1178;
x1176 = momentum;
JCudaTensor x1179;
x1179 = x1129;
x773.backward_bias(x1179, x1174, x1175, x1176);

// Dealloc(X2202)
JCudaTensor x1180;
x1180 = x1129;
x1180.free();

// V_cv96_B <~~ X2225 * d_Convolv(1,0)()/d_cv96_B
float x1182, x1183;
float x1184;
float x1185;
x1184 = 2;
x1185 = lrn_rate;
x1182 = x1184 * x1185;
x1183 = momentum;
JCudaTensor x1186;
x1186 = x1141;
x756.backward_bias(x1186, x1181, x1182, x1183);

// val X2226 = X2225 * d_Convolv(1,0)(cv96_W)/d_X7682
JCudaTensor x1187;
JCudaTensor x1188, x1189;
x1188 = x1141;
x1189 = x844;
x1187 = x756.backward_data(x1188, x1189);

// Dealloc(X2225)
JCudaTensor x1190;
x1190 = x1141;
x1190.free();

// V_cv91_W <~~ X2158 * d_Convolv(1,0)(X7671)/d_cv91_W
float x1192, x1193;
float x1194;
float x1195;
x1194 = 1;
x1195 = lrn_rate;
x1192 = x1194 * x1195;
x1193 = momentum;
JCudaTensor x1196, x1197;
x1196 = x1133;
x1197 = x792;
x733.backward_filter(x1196, x1197, x1191, x1192, x1193);

// Dealloc(X2158)
JCudaTensor x1198;
x1198 = x1133;
x1198.free();

// V_cv93_B <~~ X2178 * d_Convolv(1,1)()/d_cv93_B
float x1200, x1201;
float x1202;
float x1203;
x1202 = 2;
x1203 = lrn_rate;
x1200 = x1202 * x1203;
x1201 = momentum;
JCudaTensor x1204;
x1204 = x1137;
x766.backward_bias(x1204, x1199, x1200, x1201);

// V_cv93_W <~~ X2178 * d_Convolv(1,1)(X7675)/d_cv93_W
float x1206, x1207;
float x1208;
float x1209;
x1208 = 1;
x1209 = lrn_rate;
x1206 = x1208 * x1209;
x1207 = momentum;
JCudaTensor x1210, x1211;
x1210 = x1137;
x1211 = x836;
x766.backward_filter(x1210, x1211, x1205, x1206, x1207);

// Dealloc(X2178)
JCudaTensor x1212;
x1212 = x1137;
x1212.free();

// cv96_W <~~ V_cv96_W
float x1213, x1214;
x1213 = 1;
float x1215;
float x1216;
x1215 = 1;
float x1217;
float x1218;
float x1219;
float x1220;
x1219 = 1;
x1220 = decay;
x1217 = x1219 * x1220;
float x1221;
float x1222;
x1221 = 1;
x1222 = lrn_rate;
x1218 = x1221 * x1222;
x1216 = x1217 * x1218;
x1214 = x1215 + x1216;
JCudaTensor x1223;
x1223 = x1167;
x844.update(x1223, x1213, x1214);

// cv91_B <~~ V_cv91_B
float x1224, x1225;
x1224 = 1;
x1225 = 1;
JCudaTensor x1226;
x1226 = x1161;
x808.update(x1226, x1224, x1225);

// cv95_W <~~ V_cv95_W
float x1227, x1228;
x1227 = 1;
float x1229;
float x1230;
x1229 = 1;
float x1231;
float x1232;
float x1233;
float x1234;
x1233 = 1;
x1234 = decay;
x1231 = x1233 * x1234;
float x1235;
float x1236;
x1235 = 1;
x1236 = lrn_rate;
x1232 = x1235 * x1236;
x1230 = x1231 * x1232;
x1228 = x1229 + x1230;
JCudaTensor x1237;
x1237 = x1145;
x870.update(x1237, x1227, x1228);

// cv91_W <~~ V_cv91_W
float x1238, x1239;
x1238 = 1;
float x1240;
float x1241;
x1240 = 1;
float x1242;
float x1243;
float x1244;
float x1245;
x1244 = 1;
x1245 = decay;
x1242 = x1244 * x1245;
float x1246;
float x1247;
x1246 = 1;
x1247 = lrn_rate;
x1243 = x1246 * x1247;
x1241 = x1242 * x1243;
x1239 = x1240 + x1241;
JCudaTensor x1248;
x1248 = x1191;
x807.update(x1248, x1238, x1239);

// cv96_B <~~ V_cv96_B
float x1249, x1250;
x1249 = 1;
x1250 = 1;
JCudaTensor x1251;
x1251 = x1181;
x845.update(x1251, x1249, x1250);

// cv95_B <~~ V_cv95_B
float x1252, x1253;
x1252 = 1;
x1253 = 1;
JCudaTensor x1254;
x1254 = x1174;
x871.update(x1254, x1252, x1253);

// cv93_B <~~ V_cv93_B
float x1255, x1256;
x1255 = 1;
x1256 = 1;
JCudaTensor x1257;
x1257 = x1199;
x856.update(x1257, x1255, x1256);

// cv93_W <~~ V_cv93_W
float x1258, x1259;
x1258 = 1;
float x1260;
float x1261;
x1260 = 1;
float x1262;
float x1263;
float x1264;
float x1265;
x1264 = 1;
x1265 = decay;
x1262 = x1264 * x1265;
float x1266;
float x1267;
x1266 = 1;
x1267 = lrn_rate;
x1263 = x1266 * x1267;
x1261 = x1262 * x1263;
x1259 = x1260 + x1261;
JCudaTensor x1268;
x1268 = x1205;
x855.update(x1268, x1258, x1259);

// val X2181 = X2179 * d_ReLU()(X7675)/d_X7674
JCudaTensor x1269;
JCudaTensor x1270, x1271;
x1270 = x1155;
x1271 = x836;
x1269 = x743.backward(x1270, x1271);

// Dealloc(X7675)
JCudaTensor x1272;
x1272 = x836;
x1272.free();

// val X2205 = X2203 * d_ReLU()(X7679)/d_X7678
JCudaTensor x1273;
JCudaTensor x1274, x1275;
x1274 = x1158;
x1275 = x834;
x1273 = x759.backward(x1274, x1275);

// Dealloc(X7679)
JCudaTensor x1276;
x1276 = x834;
x1276.free();

// V_cv94_W <~~ X2205 * d_Convolv(1,0)(X7671)/d_cv94_W
float x1278, x1279;
float x1280;
float x1281;
x1280 = 1;
x1281 = lrn_rate;
x1278 = x1280 * x1281;
x1279 = momentum;
JCudaTensor x1282, x1283;
x1282 = x1273;
x1283 = x792;
x726.backward_filter(x1282, x1283, x1277, x1278, x1279);

// V_cv94_B <~~ X2205 * d_Convolv(1,0)()/d_cv94_B
float x1285, x1286;
float x1287;
float x1288;
x1287 = 2;
x1288 = lrn_rate;
x1285 = x1287 * x1288;
x1286 = momentum;
JCudaTensor x1289;
x1289 = x1273;
x726.backward_bias(x1289, x1284, x1285, x1286);

// val X2183 = (X2159 + X2181 * d_Convolv(1,0)(cv92_W)/d_X7671)
JCudaTensor x1290;
JCudaTensor x1291;
x1291 = x1152;
JCudaTensor x1292, x1293;
x1292 = x1269;
x1293 = x827;
x1290 = x740.backward_data(x1292,x1293, x1291);

// V_cv92_W <~~ X2181 * d_Convolv(1,0)(X7671)/d_cv92_W
float x1295, x1296;
float x1297;
float x1298;
x1297 = 1;
x1298 = lrn_rate;
x1295 = x1297 * x1298;
x1296 = momentum;
JCudaTensor x1299, x1300;
x1299 = x1269;
x1300 = x792;
x740.backward_filter(x1299, x1300, x1294, x1295, x1296);

// V_cv92_B <~~ X2181 * d_Convolv(1,0)()/d_cv92_B
float x1302, x1303;
float x1304;
float x1305;
x1304 = 2;
x1305 = lrn_rate;
x1302 = x1304 * x1305;
x1303 = momentum;
JCudaTensor x1306;
x1306 = x1269;
x740.backward_bias(x1306, x1301, x1302, x1303);

// Dealloc(X2181)
JCudaTensor x1307;
x1307 = x1269;
x1307.free();

// cv94_B <~~ V_cv94_B
float x1308, x1309;
x1308 = 1;
x1309 = 1;
JCudaTensor x1310;
x1310 = x1284;
x818.update(x1310, x1308, x1309);

// cv92_W <~~ V_cv92_W
float x1311, x1312;
x1311 = 1;
float x1313;
float x1314;
x1313 = 1;
float x1315;
float x1316;
float x1317;
float x1318;
x1317 = 1;
x1318 = decay;
x1315 = x1317 * x1318;
float x1319;
float x1320;
x1319 = 1;
x1320 = lrn_rate;
x1316 = x1319 * x1320;
x1314 = x1315 * x1316;
x1312 = x1313 + x1314;
JCudaTensor x1321;
x1321 = x1294;
x827.update(x1321, x1311, x1312);

// cv92_B <~~ V_cv92_B
float x1322, x1323;
x1322 = 1;
x1323 = 1;
JCudaTensor x1324;
x1324 = x1301;
x828.update(x1324, x1322, x1323);

// val X2207 = (X2183 + X2205 * d_Convolv(1,0)(cv94_W)/d_X7671)
JCudaTensor x1325;
JCudaTensor x1326;
x1326 = x1290;
JCudaTensor x1327, x1328;
x1327 = x1273;
x1328 = x817;
x1325 = x726.backward_data(x1327,x1328, x1326);

// Dealloc(X2205)
JCudaTensor x1329;
x1329 = x1273;
x1329.free();

// cv94_W <~~ V_cv94_W
float x1330, x1331;
x1330 = 1;
float x1332;
float x1333;
x1332 = 1;
float x1334;
float x1335;
float x1336;
float x1337;
x1336 = 1;
x1337 = decay;
x1334 = x1336 * x1337;
float x1338;
float x1339;
x1338 = 1;
x1339 = lrn_rate;
x1335 = x1338 * x1339;
x1333 = x1334 * x1335;
x1331 = x1332 + x1333;
JCudaTensor x1340;
x1340 = x1277;
x817.update(x1340, x1330, x1331);

// val X2229 = (X2207 + X2226 * d_Pooling(3,1,1,true)(X7682,X7671)/d_X7671)
JCudaTensor x1341;
JCudaTensor x1342;
x1342 = x1325;
JCudaTensor x1343, x1344, x1345;
x1343 = x1187;
x1344 = x821;
x1345 = x792;
x1341 = x719.backward(x1343,x1344,x1345, x1342);

// Dealloc(X2226)
JCudaTensor x1346;
x1346 = x1187;
x1346.free();

// Dealloc(X7682)
JCudaTensor x1347;
x1347 = x821;
x1347.free();

// Dealloc(X7671)
JCudaTensor x1348;
x1348 = x792;
x1348.free();

// val X2261 = Proj(X2229, X7659,X7663,X7667,X7670, 1)
JCudaTensor x1349;
JCudaTensor x1351;
x1351 = x1341;
JCudaTensor[] x1350 = x793.backward(x1351);
x1349 = x1350[1];

// val X2243 = Proj(X2229, X7659,X7663,X7667,X7670, 0)
JCudaTensor x1352;
x1352 = x1350[0];

// val X2309 = Proj(X2229, X7659,X7663,X7667,X7670, 3)
JCudaTensor x1353;
x1353 = x1350[3];

// val X2285 = Proj(X2229, X7659,X7663,X7667,X7670, 2)
JCudaTensor x1354;
x1354 = x1350[2];

// Dealloc(X2229)
JCudaTensor x1355;
x1355 = x1341;
x1355.free();

// val X2290 = X2285 * d_ReLU()(X7667)/d_X7666
JCudaTensor x1356;
JCudaTensor x1357, x1358;
x1357 = x1354;
x1358 = x781;
x1356 = x783.backward(x1357, x1358);

// Dealloc(X7667)
JCudaTensor x1359;
x1359 = x781;
x1359.free();

// val X2246 = X2243 * d_ReLU()(X7659)/d_X7658
JCudaTensor x1360;
JCudaTensor x1361, x1362;
x1361 = x1352;
x1362 = x786;
x1360 = x788.backward(x1361, x1362);

// Dealloc(X7659)
JCudaTensor x1363;
x1363 = x786;
x1363.free();

// val X2266 = X2261 * d_ReLU()(X7663)/d_X7662
JCudaTensor x1364;
JCudaTensor x1365, x1366;
x1365 = x1349;
x1366 = x778;
x1364 = x780.backward(x1365, x1366);

// Dealloc(X7663)
JCudaTensor x1367;
x1367 = x778;
x1367.free();

// val X2313 = X2309 * d_ReLU()(X7670)/d_X7669
JCudaTensor x1368;
JCudaTensor x1369, x1370;
x1369 = x1353;
x1370 = x784;
x1368 = x783.backward(x1369, x1370);

// Dealloc(X7670)
JCudaTensor x1371;
x1371 = x784;
x1371.free();

// val X2291 = X2290 * d_Convolv(1,2)(cv85_W)/d_X7665
JCudaTensor x1372;
JCudaTensor x1373, x1374;
x1373 = x1356;
x1374 = x771;
x1372 = x773.backward_data(x1373, x1374);

// V_cv85_B <~~ X2290 * d_Convolv(1,2)()/d_cv85_B
float x1376, x1377;
float x1378;
float x1379;
x1378 = 2;
x1379 = lrn_rate;
x1376 = x1378 * x1379;
x1377 = momentum;
JCudaTensor x1380;
x1380 = x1356;
x773.backward_bias(x1380, x1375, x1376, x1377);

// V_cv81_W <~~ X2246 * d_Convolv(1,0)(X7657)/d_cv81_W
float x1382, x1383;
float x1384;
float x1385;
x1384 = 1;
x1385 = lrn_rate;
x1382 = x1384 * x1385;
x1383 = momentum;
JCudaTensor x1386, x1387;
x1386 = x1360;
x1387 = x712;
x733.backward_filter(x1386, x1387, x1381, x1382, x1383);

// V_cv83_B <~~ X2266 * d_Convolv(1,1)()/d_cv83_B
float x1389, x1390;
float x1391;
float x1392;
x1391 = 2;
x1392 = lrn_rate;
x1389 = x1391 * x1392;
x1390 = momentum;
JCudaTensor x1393;
x1393 = x1364;
x766.backward_bias(x1393, x1388, x1389, x1390);

// V_cv83_W <~~ X2266 * d_Convolv(1,1)(X7661)/d_cv83_W
float x1395, x1396;
float x1397;
float x1398;
x1397 = 1;
x1398 = lrn_rate;
x1395 = x1397 * x1398;
x1396 = momentum;
JCudaTensor x1399, x1400;
x1399 = x1364;
x1400 = x741;
x766.backward_filter(x1399, x1400, x1394, x1395, x1396);

// V_cv86_W <~~ X2313 * d_Convolv(1,0)(X7668)/d_cv86_W
float x1402, x1403;
float x1404;
float x1405;
x1404 = 1;
x1405 = lrn_rate;
x1402 = x1404 * x1405;
x1403 = momentum;
JCudaTensor x1406, x1407;
x1406 = x1368;
x1407 = x717;
x756.backward_filter(x1406, x1407, x1401, x1402, x1403);

// V_cv81_B <~~ X2246 * d_Convolv(1,0)()/d_cv81_B
float x1409, x1410;
float x1411;
float x1412;
x1411 = 2;
x1412 = lrn_rate;
x1409 = x1411 * x1412;
x1410 = momentum;
JCudaTensor x1413;
x1413 = x1360;
x733.backward_bias(x1413, x1408, x1409, x1410);

// V_cv85_W <~~ X2290 * d_Convolv(1,2)(X7665)/d_cv85_W
float x1415, x1416;
float x1417;
float x1418;
x1417 = 1;
x1418 = lrn_rate;
x1415 = x1417 * x1418;
x1416 = momentum;
JCudaTensor x1419, x1420;
x1419 = x1356;
x1420 = x757;
x773.backward_filter(x1419, x1420, x1414, x1415, x1416);

// Dealloc(X2290)
JCudaTensor x1421;
x1421 = x1356;
x1421.free();

// val X2267 = X2266 * d_Convolv(1,1)(cv83_W)/d_X7661
JCudaTensor x1422;
JCudaTensor x1423, x1424;
x1423 = x1364;
x1424 = x764;
x1422 = x766.backward_data(x1423, x1424);

// Dealloc(X2266)
JCudaTensor x1425;
x1425 = x1364;
x1425.free();

// V_cv86_B <~~ X2313 * d_Convolv(1,0)()/d_cv86_B
float x1427, x1428;
float x1429;
float x1430;
x1429 = 2;
x1430 = lrn_rate;
x1427 = x1429 * x1430;
x1428 = momentum;
JCudaTensor x1431;
x1431 = x1368;
x756.backward_bias(x1431, x1426, x1427, x1428);

// val X2314 = X2313 * d_Convolv(1,0)(cv86_W)/d_X7668
JCudaTensor x1432;
JCudaTensor x1433, x1434;
x1433 = x1368;
x1434 = x754;
x1432 = x756.backward_data(x1433, x1434);

// Dealloc(X2313)
JCudaTensor x1435;
x1435 = x1368;
x1435.free();

// val X2247 = X2246 * d_Convolv(1,0)(cv81_W)/d_X7657
JCudaTensor x1436;
JCudaTensor x1437, x1438;
x1437 = x1360;
x1438 = x731;
x1436 = x733.backward_data(x1437, x1438);

// Dealloc(X2246)
JCudaTensor x1439;
x1439 = x1360;
x1439.free();

// cv81_W <~~ V_cv81_W
float x1440, x1441;
x1440 = 1;
float x1442;
float x1443;
x1442 = 1;
float x1444;
float x1445;
float x1446;
float x1447;
x1446 = 1;
x1447 = decay;
x1444 = x1446 * x1447;
float x1448;
float x1449;
x1448 = 1;
x1449 = lrn_rate;
x1445 = x1448 * x1449;
x1443 = x1444 * x1445;
x1441 = x1442 + x1443;
JCudaTensor x1450;
x1450 = x1381;
x731.update(x1450, x1440, x1441);

// cv86_B <~~ V_cv86_B
float x1451, x1452;
x1451 = 1;
x1452 = 1;
JCudaTensor x1453;
x1453 = x1426;
x755.update(x1453, x1451, x1452);

// cv83_W <~~ V_cv83_W
float x1454, x1455;
x1454 = 1;
float x1456;
float x1457;
x1456 = 1;
float x1458;
float x1459;
float x1460;
float x1461;
x1460 = 1;
x1461 = decay;
x1458 = x1460 * x1461;
float x1462;
float x1463;
x1462 = 1;
x1463 = lrn_rate;
x1459 = x1462 * x1463;
x1457 = x1458 * x1459;
x1455 = x1456 + x1457;
JCudaTensor x1464;
x1464 = x1394;
x764.update(x1464, x1454, x1455);

// cv85_W <~~ V_cv85_W
float x1465, x1466;
x1465 = 1;
float x1467;
float x1468;
x1467 = 1;
float x1469;
float x1470;
float x1471;
float x1472;
x1471 = 1;
x1472 = decay;
x1469 = x1471 * x1472;
float x1473;
float x1474;
x1473 = 1;
x1474 = lrn_rate;
x1470 = x1473 * x1474;
x1468 = x1469 * x1470;
x1466 = x1467 + x1468;
JCudaTensor x1475;
x1475 = x1414;
x771.update(x1475, x1465, x1466);

// cv83_B <~~ V_cv83_B
float x1476, x1477;
x1476 = 1;
x1477 = 1;
JCudaTensor x1478;
x1478 = x1388;
x765.update(x1478, x1476, x1477);

// cv85_B <~~ V_cv85_B
float x1479, x1480;
x1479 = 1;
x1480 = 1;
JCudaTensor x1481;
x1481 = x1375;
x772.update(x1481, x1479, x1480);

// cv86_W <~~ V_cv86_W
float x1482, x1483;
x1482 = 1;
float x1484;
float x1485;
x1484 = 1;
float x1486;
float x1487;
float x1488;
float x1489;
x1488 = 1;
x1489 = decay;
x1486 = x1488 * x1489;
float x1490;
float x1491;
x1490 = 1;
x1491 = lrn_rate;
x1487 = x1490 * x1491;
x1485 = x1486 * x1487;
x1483 = x1484 + x1485;
JCudaTensor x1492;
x1492 = x1401;
x754.update(x1492, x1482, x1483);

// cv81_B <~~ V_cv81_B
float x1493, x1494;
x1493 = 1;
x1494 = 1;
JCudaTensor x1495;
x1495 = x1408;
x732.update(x1495, x1493, x1494);

// val X2293 = X2291 * d_ReLU()(X7665)/d_X7664
JCudaTensor x1496;
JCudaTensor x1497, x1498;
x1497 = x1372;
x1498 = x757;
x1496 = x759.backward(x1497, x1498);

// Dealloc(X7665)
JCudaTensor x1499;
x1499 = x757;
x1499.free();

// val X2269 = X2267 * d_ReLU()(X7661)/d_X7660
JCudaTensor x1500;
JCudaTensor x1501, x1502;
x1501 = x1422;
x1502 = x741;
x1500 = x743.backward(x1501, x1502);

// Dealloc(X7661)
JCudaTensor x1503;
x1503 = x741;
x1503.free();

// V_cv84_W <~~ X2293 * d_Convolv(1,0)(X7657)/d_cv84_W
float x1505, x1506;
float x1507;
float x1508;
x1507 = 1;
x1508 = lrn_rate;
x1505 = x1507 * x1508;
x1506 = momentum;
JCudaTensor x1509, x1510;
x1509 = x1496;
x1510 = x712;
x726.backward_filter(x1509, x1510, x1504, x1505, x1506);

// V_cv82_B <~~ X2269 * d_Convolv(1,0)()/d_cv82_B
float x1512, x1513;
float x1514;
float x1515;
x1514 = 2;
x1515 = lrn_rate;
x1512 = x1514 * x1515;
x1513 = momentum;
JCudaTensor x1516;
x1516 = x1500;
x740.backward_bias(x1516, x1511, x1512, x1513);

// val X2271 = (X2247 + X2269 * d_Convolv(1,0)(cv82_W)/d_X7657)
JCudaTensor x1517;
JCudaTensor x1518;
x1518 = x1436;
JCudaTensor x1519, x1520;
x1519 = x1500;
x1520 = x738;
x1517 = x740.backward_data(x1519,x1520, x1518);

// V_cv82_W <~~ X2269 * d_Convolv(1,0)(X7657)/d_cv82_W
float x1522, x1523;
float x1524;
float x1525;
x1524 = 1;
x1525 = lrn_rate;
x1522 = x1524 * x1525;
x1523 = momentum;
JCudaTensor x1526, x1527;
x1526 = x1500;
x1527 = x712;
x740.backward_filter(x1526, x1527, x1521, x1522, x1523);

// Dealloc(X2269)
JCudaTensor x1528;
x1528 = x1500;
x1528.free();

// V_cv84_B <~~ X2293 * d_Convolv(1,0)()/d_cv84_B
float x1530, x1531;
float x1532;
float x1533;
x1532 = 2;
x1533 = lrn_rate;
x1530 = x1532 * x1533;
x1531 = momentum;
JCudaTensor x1534;
x1534 = x1496;
x726.backward_bias(x1534, x1529, x1530, x1531);

// cv82_B <~~ V_cv82_B
float x1535, x1536;
x1535 = 1;
x1536 = 1;
JCudaTensor x1537;
x1537 = x1511;
x739.update(x1537, x1535, x1536);

// cv82_W <~~ V_cv82_W
float x1538, x1539;
x1538 = 1;
float x1540;
float x1541;
x1540 = 1;
float x1542;
float x1543;
float x1544;
float x1545;
x1544 = 1;
x1545 = decay;
x1542 = x1544 * x1545;
float x1546;
float x1547;
x1546 = 1;
x1547 = lrn_rate;
x1543 = x1546 * x1547;
x1541 = x1542 * x1543;
x1539 = x1540 + x1541;
JCudaTensor x1548;
x1548 = x1521;
x738.update(x1548, x1538, x1539);

// cv84_B <~~ V_cv84_B
float x1549, x1550;
x1549 = 1;
x1550 = 1;
JCudaTensor x1551;
x1551 = x1529;
x725.update(x1551, x1549, x1550);

// val X2295 = (X2271 + X2293 * d_Convolv(1,0)(cv84_W)/d_X7657)
JCudaTensor x1552;
JCudaTensor x1553;
x1553 = x1517;
JCudaTensor x1554, x1555;
x1554 = x1496;
x1555 = x724;
x1552 = x726.backward_data(x1554,x1555, x1553);

// Dealloc(X2293)
JCudaTensor x1556;
x1556 = x1496;
x1556.free();

// cv84_W <~~ V_cv84_W
float x1557, x1558;
x1557 = 1;
float x1559;
float x1560;
x1559 = 1;
float x1561;
float x1562;
float x1563;
float x1564;
x1563 = 1;
x1564 = decay;
x1561 = x1563 * x1564;
float x1565;
float x1566;
x1565 = 1;
x1566 = lrn_rate;
x1562 = x1565 * x1566;
x1560 = x1561 * x1562;
x1558 = x1559 + x1560;
JCudaTensor x1567;
x1567 = x1504;
x724.update(x1567, x1557, x1558);

// val X2317 = (X2295 + X2314 * d_Pooling(3,1,1,true)(X7668,X7657)/d_X7657)
JCudaTensor x1568;
JCudaTensor x1569;
x1569 = x1552;
JCudaTensor x1570, x1571, x1572;
x1570 = x1432;
x1571 = x717;
x1572 = x712;
x1568 = x719.backward(x1570,x1571,x1572, x1569);

// Dealloc(X2314)
JCudaTensor x1573;
x1573 = x1432;
x1573.free();

// Dealloc(X7668)
JCudaTensor x1574;
x1574 = x717;
x1574.free();

// val X2319 = X2317 * d_Pooling(3,2,1,true)(X7657,X7656)/d_X7656
JCudaTensor x1575;
JCudaTensor x1576, x1577, x1578;
x1576 = x1568;
x1577 = x712;
x1578 = x701;
x1575 = x714.backward(x1576, x1577, x1578);

// Dealloc(X2317)
JCudaTensor x1579;
x1579 = x1568;
x1579.free();

// Dealloc(X7657)
JCudaTensor x1580;
x1580 = x712;
x1580.free();

// Dealloc(X7656)
JCudaTensor x1581;
x1581 = x701;
x1581.free();

// val X2351 = Proj(X2319, X7644,X7648,X7652,X7655, 1)
JCudaTensor x1582;
JCudaTensor x1584;
x1584 = x1575;
JCudaTensor[] x1583 = x250.backward(x1584);
x1582 = x1583[1];

// val X2333 = Proj(X2319, X7644,X7648,X7652,X7655, 0)
JCudaTensor x1585;
x1585 = x1583[0];

// val X2399 = Proj(X2319, X7644,X7648,X7652,X7655, 3)
JCudaTensor x1586;
x1586 = x1583[3];

// val X2375 = Proj(X2319, X7644,X7648,X7652,X7655, 2)
JCudaTensor x1587;
x1587 = x1583[2];

// Dealloc(X2319)
JCudaTensor x1588;
x1588 = x1575;
x1588.free();

// val X2380 = X2375 * d_ReLU()(X7652)/d_X7651
JCudaTensor x1589;
JCudaTensor x1590, x1591;
x1590 = x1587;
x1591 = x669;
x1589 = x246.backward(x1590, x1591);

// Dealloc(X7652)
JCudaTensor x1592;
x1592 = x669;
x1592.free();

// val X2336 = X2333 * d_ReLU()(X7644)/d_X7643
JCudaTensor x1593;
JCudaTensor x1594, x1595;
x1594 = x1585;
x1595 = x658;
x1593 = x240.backward(x1594, x1595);

// Dealloc(X7644)
JCudaTensor x1596;
x1596 = x658;
x1596.free();

// val X2356 = X2351 * d_ReLU()(X7648)/d_X7647
JCudaTensor x1597;
JCudaTensor x1598, x1599;
x1598 = x1582;
x1599 = x660;
x1597 = x243.backward(x1598, x1599);

// Dealloc(X7648)
JCudaTensor x1600;
x1600 = x660;
x1600.free();

// val X2403 = X2399 * d_ReLU()(X7655)/d_X7654
JCudaTensor x1601;
JCudaTensor x1602, x1603;
x1602 = x1586;
x1603 = x671;
x1601 = x246.backward(x1602, x1603);

// Dealloc(X7655)
JCudaTensor x1604;
x1604 = x671;
x1604.free();

// V_cv75_W <~~ X2380 * d_Convolv(1,2)(X7650)/d_cv75_W
float x1606, x1607;
float x1608;
float x1609;
x1608 = 1;
x1609 = lrn_rate;
x1606 = x1608 * x1609;
x1607 = momentum;
JCudaTensor x1610, x1611;
x1610 = x1589;
x1611 = x577;
x230.backward_filter(x1610, x1611, x1605, x1606, x1607);

// val X2337 = X2336 * d_Convolv(1,0)(cv71_W)/d_X7642
JCudaTensor x1612;
JCudaTensor x1613, x1614;
x1613 = x1593;
x1614 = x544;
x1612 = x210.backward_data(x1613, x1614);

// val X2357 = X2356 * d_Convolv(1,1)(cv73_W)/d_X7646
JCudaTensor x1615;
JCudaTensor x1616, x1617;
x1616 = x1597;
x1617 = x627;
x1615 = x237.backward_data(x1616, x1617);

// V_cv73_B <~~ X2356 * d_Convolv(1,1)()/d_cv73_B
float x1619, x1620;
float x1621;
float x1622;
x1621 = 2;
x1622 = lrn_rate;
x1619 = x1621 * x1622;
x1620 = momentum;
JCudaTensor x1623;
x1623 = x1597;
x237.backward_bias(x1623, x1618, x1619, x1620);

// val X2404 = X2403 * d_Convolv(1,0)(cv76_W)/d_X7653
JCudaTensor x1624;
JCudaTensor x1625, x1626;
x1625 = x1601;
x1626 = x592;
x1624 = x223.backward_data(x1625, x1626);

// V_cv76_B <~~ X2403 * d_Convolv(1,0)()/d_cv76_B
float x1628, x1629;
float x1630;
float x1631;
x1630 = 2;
x1631 = lrn_rate;
x1628 = x1630 * x1631;
x1629 = momentum;
JCudaTensor x1632;
x1632 = x1601;
x223.backward_bias(x1632, x1627, x1628, x1629);

// V_cv71_W <~~ X2336 * d_Convolv(1,0)(X7642)/d_cv71_W
float x1634, x1635;
float x1636;
float x1637;
x1636 = 1;
x1637 = lrn_rate;
x1634 = x1636 * x1637;
x1635 = momentum;
JCudaTensor x1638, x1639;
x1638 = x1593;
x1639 = x509;
x210.backward_filter(x1638, x1639, x1633, x1634, x1635);

// V_cv76_W <~~ X2403 * d_Convolv(1,0)(X7653)/d_cv76_W
float x1641, x1642;
float x1643;
float x1644;
x1643 = 1;
x1644 = lrn_rate;
x1641 = x1643 * x1644;
x1642 = momentum;
JCudaTensor x1645, x1646;
x1645 = x1601;
x1646 = x569;
x223.backward_filter(x1645, x1646, x1640, x1641, x1642);

// Dealloc(X2403)
JCudaTensor x1647;
x1647 = x1601;
x1647.free();

// V_cv71_B <~~ X2336 * d_Convolv(1,0)()/d_cv71_B
float x1649, x1650;
float x1651;
float x1652;
x1651 = 2;
x1652 = lrn_rate;
x1649 = x1651 * x1652;
x1650 = momentum;
JCudaTensor x1653;
x1653 = x1593;
x210.backward_bias(x1653, x1648, x1649, x1650);

// Dealloc(X2336)
JCudaTensor x1654;
x1654 = x1593;
x1654.free();

// V_cv73_W <~~ X2356 * d_Convolv(1,1)(X7646)/d_cv73_W
float x1656, x1657;
float x1658;
float x1659;
x1658 = 1;
x1659 = lrn_rate;
x1656 = x1658 * x1659;
x1657 = momentum;
JCudaTensor x1660, x1661;
x1660 = x1597;
x1661 = x575;
x237.backward_filter(x1660, x1661, x1655, x1656, x1657);

// Dealloc(X2356)
JCudaTensor x1662;
x1662 = x1597;
x1662.free();

// V_cv75_B <~~ X2380 * d_Convolv(1,2)()/d_cv75_B
float x1664, x1665;
float x1666;
float x1667;
x1666 = 2;
x1667 = lrn_rate;
x1664 = x1666 * x1667;
x1665 = momentum;
JCudaTensor x1668;
x1668 = x1589;
x230.backward_bias(x1668, x1663, x1664, x1665);

// val X2381 = X2380 * d_Convolv(1,2)(cv75_W)/d_X7650
JCudaTensor x1669;
JCudaTensor x1670, x1671;
x1670 = x1589;
x1671 = x621;
x1669 = x230.backward_data(x1670, x1671);

// Dealloc(X2380)
JCudaTensor x1672;
x1672 = x1589;
x1672.free();

// cv71_B <~~ V_cv71_B
float x1673, x1674;
x1673 = 1;
x1674 = 1;
JCudaTensor x1675;
x1675 = x1648;
x545.update(x1675, x1673, x1674);

// cv75_B <~~ V_cv75_B
float x1676, x1677;
x1676 = 1;
x1677 = 1;
JCudaTensor x1678;
x1678 = x1663;
x622.update(x1678, x1676, x1677);

// cv73_B <~~ V_cv73_B
float x1679, x1680;
x1679 = 1;
x1680 = 1;
JCudaTensor x1681;
x1681 = x1618;
x628.update(x1681, x1679, x1680);

// cv76_W <~~ V_cv76_W
float x1682, x1683;
x1682 = 1;
float x1684;
float x1685;
x1684 = 1;
float x1686;
float x1687;
float x1688;
float x1689;
x1688 = 1;
x1689 = decay;
x1686 = x1688 * x1689;
float x1690;
float x1691;
x1690 = 1;
x1691 = lrn_rate;
x1687 = x1690 * x1691;
x1685 = x1686 * x1687;
x1683 = x1684 + x1685;
JCudaTensor x1692;
x1692 = x1640;
x592.update(x1692, x1682, x1683);

// cv73_W <~~ V_cv73_W
float x1693, x1694;
x1693 = 1;
float x1695;
float x1696;
x1695 = 1;
float x1697;
float x1698;
float x1699;
float x1700;
x1699 = 1;
x1700 = decay;
x1697 = x1699 * x1700;
float x1701;
float x1702;
x1701 = 1;
x1702 = lrn_rate;
x1698 = x1701 * x1702;
x1696 = x1697 * x1698;
x1694 = x1695 + x1696;
JCudaTensor x1703;
x1703 = x1655;
x627.update(x1703, x1693, x1694);

// cv71_W <~~ V_cv71_W
float x1704, x1705;
x1704 = 1;
float x1706;
float x1707;
x1706 = 1;
float x1708;
float x1709;
float x1710;
float x1711;
x1710 = 1;
x1711 = decay;
x1708 = x1710 * x1711;
float x1712;
float x1713;
x1712 = 1;
x1713 = lrn_rate;
x1709 = x1712 * x1713;
x1707 = x1708 * x1709;
x1705 = x1706 + x1707;
JCudaTensor x1714;
x1714 = x1633;
x544.update(x1714, x1704, x1705);

// cv76_B <~~ V_cv76_B
float x1715, x1716;
x1715 = 1;
x1716 = 1;
JCudaTensor x1717;
x1717 = x1627;
x593.update(x1717, x1715, x1716);

// cv75_W <~~ V_cv75_W
float x1718, x1719;
x1718 = 1;
float x1720;
float x1721;
x1720 = 1;
float x1722;
float x1723;
float x1724;
float x1725;
x1724 = 1;
x1725 = decay;
x1722 = x1724 * x1725;
float x1726;
float x1727;
x1726 = 1;
x1727 = lrn_rate;
x1723 = x1726 * x1727;
x1721 = x1722 * x1723;
x1719 = x1720 + x1721;
JCudaTensor x1728;
x1728 = x1605;
x621.update(x1728, x1718, x1719);

// val X2383 = X2381 * d_ReLU()(X7650)/d_X7649
JCudaTensor x1729;
JCudaTensor x1730, x1731;
x1730 = x1669;
x1731 = x577;
x1729 = x213.backward(x1730, x1731);

// Dealloc(X7650)
JCudaTensor x1732;
x1732 = x577;
x1732.free();

// val X2359 = X2357 * d_ReLU()(X7646)/d_X7645
JCudaTensor x1733;
JCudaTensor x1734, x1735;
x1734 = x1615;
x1735 = x575;
x1733 = x216.backward(x1734, x1735);

// Dealloc(X7646)
JCudaTensor x1736;
x1736 = x575;
x1736.free();

// V_cv74_W <~~ X2383 * d_Convolv(1,0)(X7642)/d_cv74_W
float x1738, x1739;
float x1740;
float x1741;
x1740 = 1;
x1741 = lrn_rate;
x1738 = x1740 * x1741;
x1739 = momentum;
JCudaTensor x1742, x1743;
x1742 = x1729;
x1743 = x509;
x196.backward_filter(x1742, x1743, x1737, x1738, x1739);

// val X2361 = (X2337 + X2359 * d_Convolv(1,0)(cv72_W)/d_X7642)
JCudaTensor x1744;
JCudaTensor x1745;
x1745 = x1612;
JCudaTensor x1746, x1747;
x1746 = x1733;
x1747 = x556;
x1744 = x203.backward_data(x1746,x1747, x1745);

// V_cv72_W <~~ X2359 * d_Convolv(1,0)(X7642)/d_cv72_W
float x1749, x1750;
float x1751;
float x1752;
x1751 = 1;
x1752 = lrn_rate;
x1749 = x1751 * x1752;
x1750 = momentum;
JCudaTensor x1753, x1754;
x1753 = x1733;
x1754 = x509;
x203.backward_filter(x1753, x1754, x1748, x1749, x1750);

// V_cv72_B <~~ X2359 * d_Convolv(1,0)()/d_cv72_B
float x1756, x1757;
float x1758;
float x1759;
x1758 = 2;
x1759 = lrn_rate;
x1756 = x1758 * x1759;
x1757 = momentum;
JCudaTensor x1760;
x1760 = x1733;
x203.backward_bias(x1760, x1755, x1756, x1757);

// Dealloc(X2359)
JCudaTensor x1761;
x1761 = x1733;
x1761.free();

// V_cv74_B <~~ X2383 * d_Convolv(1,0)()/d_cv74_B
float x1763, x1764;
float x1765;
float x1766;
x1765 = 2;
x1766 = lrn_rate;
x1763 = x1765 * x1766;
x1764 = momentum;
JCudaTensor x1767;
x1767 = x1729;
x196.backward_bias(x1767, x1762, x1763, x1764);

// cv72_W <~~ V_cv72_W
float x1768, x1769;
x1768 = 1;
float x1770;
float x1771;
x1770 = 1;
float x1772;
float x1773;
float x1774;
float x1775;
x1774 = 1;
x1775 = decay;
x1772 = x1774 * x1775;
float x1776;
float x1777;
x1776 = 1;
x1777 = lrn_rate;
x1773 = x1776 * x1777;
x1771 = x1772 * x1773;
x1769 = x1770 + x1771;
JCudaTensor x1778;
x1778 = x1748;
x556.update(x1778, x1768, x1769);

// cv72_B <~~ V_cv72_B
float x1779, x1780;
x1779 = 1;
x1780 = 1;
JCudaTensor x1781;
x1781 = x1755;
x557.update(x1781, x1779, x1780);

// cv74_B <~~ V_cv74_B
float x1782, x1783;
x1782 = 1;
x1783 = 1;
JCudaTensor x1784;
x1784 = x1762;
x563.update(x1784, x1782, x1783);

// val X2385 = (X2361 + X2383 * d_Convolv(1,0)(cv74_W)/d_X7642)
JCudaTensor x1785;
JCudaTensor x1786;
x1786 = x1744;
JCudaTensor x1787, x1788;
x1787 = x1729;
x1788 = x562;
x1785 = x196.backward_data(x1787,x1788, x1786);

// Dealloc(X2383)
JCudaTensor x1789;
x1789 = x1729;
x1789.free();

// cv74_W <~~ V_cv74_W
float x1790, x1791;
x1790 = 1;
float x1792;
float x1793;
x1792 = 1;
float x1794;
float x1795;
float x1796;
float x1797;
x1796 = 1;
x1797 = decay;
x1794 = x1796 * x1797;
float x1798;
float x1799;
x1798 = 1;
x1799 = lrn_rate;
x1795 = x1798 * x1799;
x1793 = x1794 * x1795;
x1791 = x1792 + x1793;
JCudaTensor x1800;
x1800 = x1737;
x562.update(x1800, x1790, x1791);

// val X2407 = (X2385 + X2404 * d_Pooling(3,1,1,true)(X7653,X7642)/d_X7642)
JCudaTensor x1801;
JCudaTensor x1802;
x1802 = x1785;
JCudaTensor x1803, x1804, x1805;
x1803 = x1624;
x1804 = x569;
x1805 = x509;
x1801 = x189.backward(x1803,x1804,x1805, x1802);

// Dealloc(X2404)
JCudaTensor x1806;
x1806 = x1624;
x1806.free();

// Dealloc(X7653)
JCudaTensor x1807;
x1807 = x569;
x1807.free();

// Dealloc(X7642)
JCudaTensor x1808;
x1808 = x509;
x1808.free();

// val X2443 = (X2442 * loss2)
JCudaTensor x1809;
JCudaTensor x1810;
float x1811;
x1810 = x967;
x1811 = loss2;
x1809 = x1810.times_i(x1811);

// val X2444 = (X2407 + X2443)
JCudaTensor x1812;
JCudaTensor x1813, x1814;
x1813 = x1801;
x1814 = x1809;
x1812 = x1813.plus_i(x1814);

// Dealloc(X2443)
JCudaTensor x1815;
x1815 = x1809;
x1815.free();

// val X2552 = Proj(X2444, X7630,X7634,X7638,X7641, 3)
JCudaTensor x1816;
JCudaTensor x1818;
x1818 = x1812;
JCudaTensor[] x1817 = x250.backward(x1818);
x1816 = x1817[3];

// val X2528 = Proj(X2444, X7630,X7634,X7638,X7641, 2)
JCudaTensor x1819;
x1819 = x1817[2];

// val X2504 = Proj(X2444, X7630,X7634,X7638,X7641, 1)
JCudaTensor x1820;
x1820 = x1817[1];

// val X2486 = Proj(X2444, X7630,X7634,X7638,X7641, 0)
JCudaTensor x1821;
x1821 = x1817[0];

// Dealloc(X2444)
JCudaTensor x1822;
x1822 = x1812;
x1822.free();

// val X2556 = X2552 * d_ReLU()(X7641)/d_X7640
JCudaTensor x1823;
JCudaTensor x1824, x1825;
x1824 = x1816;
x1825 = x496;
x1823 = x246.backward(x1824, x1825);

// Dealloc(X7641)
JCudaTensor x1826;
x1826 = x496;
x1826.free();

// val X2533 = X2528 * d_ReLU()(X7638)/d_X7637
JCudaTensor x1827;
JCudaTensor x1828, x1829;
x1828 = x1819;
x1829 = x481;
x1827 = x246.backward(x1828, x1829);

// Dealloc(X7638)
JCudaTensor x1830;
x1830 = x481;
x1830.free();

// val X2509 = X2504 * d_ReLU()(X7634)/d_X7633
JCudaTensor x1831;
JCudaTensor x1832, x1833;
x1832 = x1820;
x1833 = x494;
x1831 = x243.backward(x1832, x1833);

// Dealloc(X7634)
JCudaTensor x1834;
x1834 = x494;
x1834.free();

// val X2489 = X2486 * d_ReLU()(X7630)/d_X7629
JCudaTensor x1835;
JCudaTensor x1836, x1837;
x1836 = x1821;
x1837 = x492;
x1835 = x240.backward(x1836, x1837);

// Dealloc(X7630)
JCudaTensor x1838;
x1838 = x492;
x1838.free();

// V_cv66_W <~~ X2556 * d_Convolv(1,0)(X7639)/d_cv66_W
float x1840, x1841;
float x1842;
float x1843;
x1842 = 1;
x1843 = lrn_rate;
x1840 = x1842 * x1843;
x1841 = momentum;
JCudaTensor x1844, x1845;
x1844 = x1823;
x1845 = x436;
x223.backward_filter(x1844, x1845, x1839, x1840, x1841);

// V_cv65_B <~~ X2533 * d_Convolv(1,2)()/d_cv65_B
float x1847, x1848;
float x1849;
float x1850;
x1849 = 2;
x1850 = lrn_rate;
x1847 = x1849 * x1850;
x1848 = momentum;
JCudaTensor x1851;
x1851 = x1827;
x230.backward_bias(x1851, x1846, x1847, x1848);

// val X2534 = X2533 * d_Convolv(1,2)(cv65_W)/d_X7636
JCudaTensor x1852;
JCudaTensor x1853, x1854;
x1853 = x1827;
x1854 = x460;
x1852 = x230.backward_data(x1853, x1854);

// V_cv63_B <~~ X2509 * d_Convolv(1,1)()/d_cv63_B
float x1856, x1857;
float x1858;
float x1859;
x1858 = 2;
x1859 = lrn_rate;
x1856 = x1858 * x1859;
x1857 = momentum;
JCudaTensor x1860;
x1860 = x1831;
x237.backward_bias(x1860, x1855, x1856, x1857);

// V_cv61_W <~~ X2489 * d_Convolv(1,0)(X7628)/d_cv61_W
float x1862, x1863;
float x1864;
float x1865;
x1864 = 1;
x1865 = lrn_rate;
x1862 = x1864 * x1865;
x1863 = momentum;
JCudaTensor x1866, x1867;
x1866 = x1835;
x1867 = x400;
x210.backward_filter(x1866, x1867, x1861, x1862, x1863);

// V_cv65_W <~~ X2533 * d_Convolv(1,2)(X7636)/d_cv65_W
float x1869, x1870;
float x1871;
float x1872;
x1871 = 1;
x1872 = lrn_rate;
x1869 = x1871 * x1872;
x1870 = momentum;
JCudaTensor x1873, x1874;
x1873 = x1827;
x1874 = x438;
x230.backward_filter(x1873, x1874, x1868, x1869, x1870);

// Dealloc(X2533)
JCudaTensor x1875;
x1875 = x1827;
x1875.free();

// V_cv63_W <~~ X2509 * d_Convolv(1,1)(X7632)/d_cv63_W
float x1877, x1878;
float x1879;
float x1880;
x1879 = 1;
x1880 = lrn_rate;
x1877 = x1879 * x1880;
x1878 = momentum;
JCudaTensor x1881, x1882;
x1881 = x1831;
x1882 = x440;
x237.backward_filter(x1881, x1882, x1876, x1877, x1878);

// V_cv66_B <~~ X2556 * d_Convolv(1,0)()/d_cv66_B
float x1884, x1885;
float x1886;
float x1887;
x1886 = 2;
x1887 = lrn_rate;
x1884 = x1886 * x1887;
x1885 = momentum;
JCudaTensor x1888;
x1888 = x1823;
x223.backward_bias(x1888, x1883, x1884, x1885);

// val X2557 = X2556 * d_Convolv(1,0)(cv66_W)/d_X7639
JCudaTensor x1889;
JCudaTensor x1890, x1891;
x1890 = x1823;
x1891 = x450;
x1889 = x223.backward_data(x1890, x1891);

// Dealloc(X2556)
JCudaTensor x1892;
x1892 = x1823;
x1892.free();

// val X2510 = X2509 * d_Convolv(1,1)(cv63_W)/d_X7632
JCudaTensor x1893;
JCudaTensor x1894, x1895;
x1894 = x1831;
x1895 = x469;
x1893 = x237.backward_data(x1894, x1895);

// Dealloc(X2509)
JCudaTensor x1896;
x1896 = x1831;
x1896.free();

// val X2490 = X2489 * d_Convolv(1,0)(cv61_W)/d_X7628
JCudaTensor x1897;
JCudaTensor x1898, x1899;
x1898 = x1835;
x1899 = x423;
x1897 = x210.backward_data(x1898, x1899);

// V_cv61_B <~~ X2489 * d_Convolv(1,0)()/d_cv61_B
float x1901, x1902;
float x1903;
float x1904;
x1903 = 2;
x1904 = lrn_rate;
x1901 = x1903 * x1904;
x1902 = momentum;
JCudaTensor x1905;
x1905 = x1835;
x210.backward_bias(x1905, x1900, x1901, x1902);

// Dealloc(X2489)
JCudaTensor x1906;
x1906 = x1835;
x1906.free();

// cv61_W <~~ V_cv61_W
float x1907, x1908;
x1907 = 1;
float x1909;
float x1910;
x1909 = 1;
float x1911;
float x1912;
float x1913;
float x1914;
x1913 = 1;
x1914 = decay;
x1911 = x1913 * x1914;
float x1915;
float x1916;
x1915 = 1;
x1916 = lrn_rate;
x1912 = x1915 * x1916;
x1910 = x1911 * x1912;
x1908 = x1909 + x1910;
JCudaTensor x1917;
x1917 = x1861;
x423.update(x1917, x1907, x1908);

// cv61_B <~~ V_cv61_B
float x1918, x1919;
x1918 = 1;
x1919 = 1;
JCudaTensor x1920;
x1920 = x1900;
x424.update(x1920, x1918, x1919);

// cv65_W <~~ V_cv65_W
float x1921, x1922;
x1921 = 1;
float x1923;
float x1924;
x1923 = 1;
float x1925;
float x1926;
float x1927;
float x1928;
x1927 = 1;
x1928 = decay;
x1925 = x1927 * x1928;
float x1929;
float x1930;
x1929 = 1;
x1930 = lrn_rate;
x1926 = x1929 * x1930;
x1924 = x1925 * x1926;
x1922 = x1923 + x1924;
JCudaTensor x1931;
x1931 = x1868;
x460.update(x1931, x1921, x1922);

// cv63_B <~~ V_cv63_B
float x1932, x1933;
x1932 = 1;
x1933 = 1;
JCudaTensor x1934;
x1934 = x1855;
x470.update(x1934, x1932, x1933);

// cv63_W <~~ V_cv63_W
float x1935, x1936;
x1935 = 1;
float x1937;
float x1938;
x1937 = 1;
float x1939;
float x1940;
float x1941;
float x1942;
x1941 = 1;
x1942 = decay;
x1939 = x1941 * x1942;
float x1943;
float x1944;
x1943 = 1;
x1944 = lrn_rate;
x1940 = x1943 * x1944;
x1938 = x1939 * x1940;
x1936 = x1937 + x1938;
JCudaTensor x1945;
x1945 = x1876;
x469.update(x1945, x1935, x1936);

// cv66_W <~~ V_cv66_W
float x1946, x1947;
x1946 = 1;
float x1948;
float x1949;
x1948 = 1;
float x1950;
float x1951;
float x1952;
float x1953;
x1952 = 1;
x1953 = decay;
x1950 = x1952 * x1953;
float x1954;
float x1955;
x1954 = 1;
x1955 = lrn_rate;
x1951 = x1954 * x1955;
x1949 = x1950 * x1951;
x1947 = x1948 + x1949;
JCudaTensor x1956;
x1956 = x1839;
x450.update(x1956, x1946, x1947);

// cv65_B <~~ V_cv65_B
float x1957, x1958;
x1957 = 1;
x1958 = 1;
JCudaTensor x1959;
x1959 = x1846;
x461.update(x1959, x1957, x1958);

// cv66_B <~~ V_cv66_B
float x1960, x1961;
x1960 = 1;
x1961 = 1;
JCudaTensor x1962;
x1962 = x1883;
x451.update(x1962, x1960, x1961);

// val X2536 = X2534 * d_ReLU()(X7636)/d_X7635
JCudaTensor x1963;
JCudaTensor x1964, x1965;
x1964 = x1852;
x1965 = x438;
x1963 = x213.backward(x1964, x1965);

// Dealloc(X7636)
JCudaTensor x1966;
x1966 = x438;
x1966.free();

// val X2512 = X2510 * d_ReLU()(X7632)/d_X7631
JCudaTensor x1967;
JCudaTensor x1968, x1969;
x1968 = x1893;
x1969 = x440;
x1967 = x216.backward(x1968, x1969);

// Dealloc(X7632)
JCudaTensor x1970;
x1970 = x440;
x1970.free();

// V_cv64_W <~~ X2536 * d_Convolv(1,0)(X7628)/d_cv64_W
float x1972, x1973;
float x1974;
float x1975;
x1974 = 1;
x1975 = lrn_rate;
x1972 = x1974 * x1975;
x1973 = momentum;
JCudaTensor x1976, x1977;
x1976 = x1963;
x1977 = x400;
x196.backward_filter(x1976, x1977, x1971, x1972, x1973);

// val X2514 = (X2490 + X2512 * d_Convolv(1,0)(cv62_W)/d_X7628)
JCudaTensor x1978;
JCudaTensor x1979;
x1979 = x1897;
JCudaTensor x1980, x1981;
x1980 = x1967;
x1981 = x434;
x1978 = x203.backward_data(x1980,x1981, x1979);

// V_cv62_W <~~ X2512 * d_Convolv(1,0)(X7628)/d_cv62_W
float x1983, x1984;
float x1985;
float x1986;
x1985 = 1;
x1986 = lrn_rate;
x1983 = x1985 * x1986;
x1984 = momentum;
JCudaTensor x1987, x1988;
x1987 = x1967;
x1988 = x400;
x203.backward_filter(x1987, x1988, x1982, x1983, x1984);

// V_cv62_B <~~ X2512 * d_Convolv(1,0)()/d_cv62_B
float x1990, x1991;
float x1992;
float x1993;
x1992 = 2;
x1993 = lrn_rate;
x1990 = x1992 * x1993;
x1991 = momentum;
JCudaTensor x1994;
x1994 = x1967;
x203.backward_bias(x1994, x1989, x1990, x1991);

// Dealloc(X2512)
JCudaTensor x1995;
x1995 = x1967;
x1995.free();

// V_cv64_B <~~ X2536 * d_Convolv(1,0)()/d_cv64_B
float x1997, x1998;
float x1999;
float x2000;
x1999 = 2;
x2000 = lrn_rate;
x1997 = x1999 * x2000;
x1998 = momentum;
JCudaTensor x2001;
x2001 = x1963;
x196.backward_bias(x2001, x1996, x1997, x1998);

// cv62_W <~~ V_cv62_W
float x2002, x2003;
x2002 = 1;
float x2004;
float x2005;
x2004 = 1;
float x2006;
float x2007;
float x2008;
float x2009;
x2008 = 1;
x2009 = decay;
x2006 = x2008 * x2009;
float x2010;
float x2011;
x2010 = 1;
x2011 = lrn_rate;
x2007 = x2010 * x2011;
x2005 = x2006 * x2007;
x2003 = x2004 + x2005;
JCudaTensor x2012;
x2012 = x1982;
x434.update(x2012, x2002, x2003);

// cv62_B <~~ V_cv62_B
float x2013, x2014;
x2013 = 1;
x2014 = 1;
JCudaTensor x2015;
x2015 = x1989;
x435.update(x2015, x2013, x2014);

// cv64_B <~~ V_cv64_B
float x2016, x2017;
x2016 = 1;
x2017 = 1;
JCudaTensor x2018;
x2018 = x1996;
x418.update(x2018, x2016, x2017);

// val X2538 = (X2514 + X2536 * d_Convolv(1,0)(cv64_W)/d_X7628)
JCudaTensor x2019;
JCudaTensor x2020;
x2020 = x1978;
JCudaTensor x2021, x2022;
x2021 = x1963;
x2022 = x417;
x2019 = x196.backward_data(x2021,x2022, x2020);

// Dealloc(X2536)
JCudaTensor x2023;
x2023 = x1963;
x2023.free();

// cv64_W <~~ V_cv64_W
float x2024, x2025;
x2024 = 1;
float x2026;
float x2027;
x2026 = 1;
float x2028;
float x2029;
float x2030;
float x2031;
x2030 = 1;
x2031 = decay;
x2028 = x2030 * x2031;
float x2032;
float x2033;
x2032 = 1;
x2033 = lrn_rate;
x2029 = x2032 * x2033;
x2027 = x2028 * x2029;
x2025 = x2026 + x2027;
JCudaTensor x2034;
x2034 = x1971;
x417.update(x2034, x2024, x2025);

// val X2560 = (X2538 + X2557 * d_Pooling(3,1,1,true)(X7639,X7628)/d_X7628)
JCudaTensor x2035;
JCudaTensor x2036;
x2036 = x2019;
JCudaTensor x2037, x2038, x2039;
x2037 = x1889;
x2038 = x436;
x2039 = x400;
x2035 = x189.backward(x2037,x2038,x2039, x2036);

// Dealloc(X2557)
JCudaTensor x2040;
x2040 = x1889;
x2040.free();

// Dealloc(X7639)
JCudaTensor x2041;
x2041 = x436;
x2041.free();

// Dealloc(X7628)
JCudaTensor x2042;
x2042 = x400;
x2042.free();

// val X2640 = Proj(X2560, X7616,X7620,X7624,X7627, 3)
JCudaTensor x2043;
JCudaTensor x2045;
x2045 = x2035;
JCudaTensor[] x2044 = x250.backward(x2045);
x2043 = x2044[3];

// val X2592 = Proj(X2560, X7616,X7620,X7624,X7627, 1)
JCudaTensor x2046;
x2046 = x2044[1];

// val X2574 = Proj(X2560, X7616,X7620,X7624,X7627, 0)
JCudaTensor x2047;
x2047 = x2044[0];

// val X2616 = Proj(X2560, X7616,X7620,X7624,X7627, 2)
JCudaTensor x2048;
x2048 = x2044[2];

// Dealloc(X2560)
JCudaTensor x2049;
x2049 = x2035;
x2049.free();

// val X2621 = X2616 * d_ReLU()(X7624)/d_X7623
JCudaTensor x2050;
JCudaTensor x2051, x2052;
x2051 = x2048;
x2052 = x392;
x2050 = x246.backward(x2051, x2052);

// Dealloc(X7624)
JCudaTensor x2053;
x2053 = x392;
x2053.free();

// val X2597 = X2592 * d_ReLU()(X7620)/d_X7619
JCudaTensor x2054;
JCudaTensor x2055, x2056;
x2055 = x2046;
x2056 = x388;
x2054 = x243.backward(x2055, x2056);

// Dealloc(X7620)
JCudaTensor x2057;
x2057 = x388;
x2057.free();

// val X2577 = X2574 * d_ReLU()(X7616)/d_X7615
JCudaTensor x2058;
JCudaTensor x2059, x2060;
x2059 = x2047;
x2060 = x390;
x2058 = x240.backward(x2059, x2060);

// Dealloc(X7616)
JCudaTensor x2061;
x2061 = x390;
x2061.free();

// val X2644 = X2640 * d_ReLU()(X7627)/d_X7626
JCudaTensor x2062;
JCudaTensor x2063, x2064;
x2063 = x2043;
x2064 = x394;
x2062 = x246.backward(x2063, x2064);

// Dealloc(X7627)
JCudaTensor x2065;
x2065 = x394;
x2065.free();

// V_cv55_W <~~ X2621 * d_Convolv(1,2)(X7622)/d_cv55_W
float x2067, x2068;
float x2069;
float x2070;
x2069 = 1;
x2070 = lrn_rate;
x2067 = x2069 * x2070;
x2068 = momentum;
JCudaTensor x2071, x2072;
x2071 = x2050;
x2072 = x368;
x230.backward_filter(x2071, x2072, x2066, x2067, x2068);

// val X2598 = X2597 * d_Convolv(1,1)(cv53_W)/d_X7618
JCudaTensor x2073;
JCudaTensor x2074, x2075;
x2074 = x2054;
x2075 = x374;
x2073 = x237.backward_data(x2074, x2075);

// val X2578 = X2577 * d_Convolv(1,0)(cv51_W)/d_X7614
JCudaTensor x2076;
JCudaTensor x2077, x2078;
x2077 = x2058;
x2078 = x355;
x2076 = x210.backward_data(x2077, x2078);

// V_cv53_W <~~ X2597 * d_Convolv(1,1)(X7618)/d_cv53_W
float x2080, x2081;
float x2082;
float x2083;
x2082 = 1;
x2083 = lrn_rate;
x2080 = x2082 * x2083;
x2081 = momentum;
JCudaTensor x2084, x2085;
x2084 = x2054;
x2085 = x366;
x237.backward_filter(x2084, x2085, x2079, x2080, x2081);

// val X2645 = X2644 * d_Convolv(1,0)(cv56_W)/d_X7625
JCudaTensor x2086;
JCudaTensor x2087, x2088;
x2087 = x2062;
x2088 = x364;
x2086 = x223.backward_data(x2087, x2088);

// val X2622 = X2621 * d_Convolv(1,2)(cv55_W)/d_X7622
JCudaTensor x2089;
JCudaTensor x2090, x2091;
x2090 = x2050;
x2091 = x380;
x2089 = x230.backward_data(x2090, x2091);

// V_cv53_B <~~ X2597 * d_Convolv(1,1)()/d_cv53_B
float x2093, x2094;
float x2095;
float x2096;
x2095 = 2;
x2096 = lrn_rate;
x2093 = x2095 * x2096;
x2094 = momentum;
JCudaTensor x2097;
x2097 = x2054;
x237.backward_bias(x2097, x2092, x2093, x2094);

// Dealloc(X2597)
JCudaTensor x2098;
x2098 = x2054;
x2098.free();

// V_cv51_B <~~ X2577 * d_Convolv(1,0)()/d_cv51_B
float x2100, x2101;
float x2102;
float x2103;
x2102 = 2;
x2103 = lrn_rate;
x2100 = x2102 * x2103;
x2101 = momentum;
JCudaTensor x2104;
x2104 = x2058;
x210.backward_bias(x2104, x2099, x2100, x2101);

// V_cv56_B <~~ X2644 * d_Convolv(1,0)()/d_cv56_B
float x2106, x2107;
float x2108;
float x2109;
x2108 = 2;
x2109 = lrn_rate;
x2106 = x2108 * x2109;
x2107 = momentum;
JCudaTensor x2110;
x2110 = x2062;
x223.backward_bias(x2110, x2105, x2106, x2107);

// V_cv51_W <~~ X2577 * d_Convolv(1,0)(X7614)/d_cv51_W
float x2112, x2113;
float x2114;
float x2115;
x2114 = 1;
x2115 = lrn_rate;
x2112 = x2114 * x2115;
x2113 = momentum;
JCudaTensor x2116, x2117;
x2116 = x2058;
x2117 = x325;
x210.backward_filter(x2116, x2117, x2111, x2112, x2113);

// Dealloc(X2577)
JCudaTensor x2118;
x2118 = x2058;
x2118.free();

// V_cv56_W <~~ X2644 * d_Convolv(1,0)(X7625)/d_cv56_W
float x2120, x2121;
float x2122;
float x2123;
x2122 = 1;
x2123 = lrn_rate;
x2120 = x2122 * x2123;
x2121 = momentum;
JCudaTensor x2124, x2125;
x2124 = x2062;
x2125 = x346;
x223.backward_filter(x2124, x2125, x2119, x2120, x2121);

// Dealloc(X2644)
JCudaTensor x2126;
x2126 = x2062;
x2126.free();

// V_cv55_B <~~ X2621 * d_Convolv(1,2)()/d_cv55_B
float x2128, x2129;
float x2130;
float x2131;
x2130 = 2;
x2131 = lrn_rate;
x2128 = x2130 * x2131;
x2129 = momentum;
JCudaTensor x2132;
x2132 = x2050;
x230.backward_bias(x2132, x2127, x2128, x2129);

// Dealloc(X2621)
JCudaTensor x2133;
x2133 = x2050;
x2133.free();

// cv56_W <~~ V_cv56_W
float x2134, x2135;
x2134 = 1;
float x2136;
float x2137;
x2136 = 1;
float x2138;
float x2139;
float x2140;
float x2141;
x2140 = 1;
x2141 = decay;
x2138 = x2140 * x2141;
float x2142;
float x2143;
x2142 = 1;
x2143 = lrn_rate;
x2139 = x2142 * x2143;
x2137 = x2138 * x2139;
x2135 = x2136 + x2137;
JCudaTensor x2144;
x2144 = x2119;
x364.update(x2144, x2134, x2135);

// cv55_B <~~ V_cv55_B
float x2145, x2146;
x2145 = 1;
x2146 = 1;
JCudaTensor x2147;
x2147 = x2127;
x381.update(x2147, x2145, x2146);

// cv53_W <~~ V_cv53_W
float x2148, x2149;
x2148 = 1;
float x2150;
float x2151;
x2150 = 1;
float x2152;
float x2153;
float x2154;
float x2155;
x2154 = 1;
x2155 = decay;
x2152 = x2154 * x2155;
float x2156;
float x2157;
x2156 = 1;
x2157 = lrn_rate;
x2153 = x2156 * x2157;
x2151 = x2152 * x2153;
x2149 = x2150 + x2151;
JCudaTensor x2158;
x2158 = x2079;
x374.update(x2158, x2148, x2149);

// cv53_B <~~ V_cv53_B
float x2159, x2160;
x2159 = 1;
x2160 = 1;
JCudaTensor x2161;
x2161 = x2092;
x375.update(x2161, x2159, x2160);

// cv51_B <~~ V_cv51_B
float x2162, x2163;
x2162 = 1;
x2163 = 1;
JCudaTensor x2164;
x2164 = x2099;
x356.update(x2164, x2162, x2163);

// cv51_W <~~ V_cv51_W
float x2165, x2166;
x2165 = 1;
float x2167;
float x2168;
x2167 = 1;
float x2169;
float x2170;
float x2171;
float x2172;
x2171 = 1;
x2172 = decay;
x2169 = x2171 * x2172;
float x2173;
float x2174;
x2173 = 1;
x2174 = lrn_rate;
x2170 = x2173 * x2174;
x2168 = x2169 * x2170;
x2166 = x2167 + x2168;
JCudaTensor x2175;
x2175 = x2111;
x355.update(x2175, x2165, x2166);

// cv55_W <~~ V_cv55_W
float x2176, x2177;
x2176 = 1;
float x2178;
float x2179;
x2178 = 1;
float x2180;
float x2181;
float x2182;
float x2183;
x2182 = 1;
x2183 = decay;
x2180 = x2182 * x2183;
float x2184;
float x2185;
x2184 = 1;
x2185 = lrn_rate;
x2181 = x2184 * x2185;
x2179 = x2180 * x2181;
x2177 = x2178 + x2179;
JCudaTensor x2186;
x2186 = x2066;
x380.update(x2186, x2176, x2177);

// cv56_B <~~ V_cv56_B
float x2187, x2188;
x2187 = 1;
x2188 = 1;
JCudaTensor x2189;
x2189 = x2105;
x365.update(x2189, x2187, x2188);

// val X2624 = X2622 * d_ReLU()(X7622)/d_X7621
JCudaTensor x2190;
JCudaTensor x2191, x2192;
x2191 = x2089;
x2192 = x368;
x2190 = x213.backward(x2191, x2192);

// Dealloc(X7622)
JCudaTensor x2193;
x2193 = x368;
x2193.free();

// val X2600 = X2598 * d_ReLU()(X7618)/d_X7617
JCudaTensor x2194;
JCudaTensor x2195, x2196;
x2195 = x2073;
x2196 = x366;
x2194 = x216.backward(x2195, x2196);

// Dealloc(X7618)
JCudaTensor x2197;
x2197 = x366;
x2197.free();

// V_cv54_W <~~ X2624 * d_Convolv(1,0)(X7614)/d_cv54_W
float x2199, x2200;
float x2201;
float x2202;
x2201 = 1;
x2202 = lrn_rate;
x2199 = x2201 * x2202;
x2200 = momentum;
JCudaTensor x2203, x2204;
x2203 = x2190;
x2204 = x325;
x196.backward_filter(x2203, x2204, x2198, x2199, x2200);

// V_cv52_B <~~ X2600 * d_Convolv(1,0)()/d_cv52_B
float x2206, x2207;
float x2208;
float x2209;
x2208 = 2;
x2209 = lrn_rate;
x2206 = x2208 * x2209;
x2207 = momentum;
JCudaTensor x2210;
x2210 = x2194;
x203.backward_bias(x2210, x2205, x2206, x2207);

// V_cv54_B <~~ X2624 * d_Convolv(1,0)()/d_cv54_B
float x2212, x2213;
float x2214;
float x2215;
x2214 = 2;
x2215 = lrn_rate;
x2212 = x2214 * x2215;
x2213 = momentum;
JCudaTensor x2216;
x2216 = x2190;
x196.backward_bias(x2216, x2211, x2212, x2213);

// val X2602 = (X2578 + X2600 * d_Convolv(1,0)(cv52_W)/d_X7614)
JCudaTensor x2217;
JCudaTensor x2218;
x2218 = x2076;
JCudaTensor x2219, x2220;
x2219 = x2194;
x2220 = x344;
x2217 = x203.backward_data(x2219,x2220, x2218);

// V_cv52_W <~~ X2600 * d_Convolv(1,0)(X7614)/d_cv52_W
float x2222, x2223;
float x2224;
float x2225;
x2224 = 1;
x2225 = lrn_rate;
x2222 = x2224 * x2225;
x2223 = momentum;
JCudaTensor x2226, x2227;
x2226 = x2194;
x2227 = x325;
x203.backward_filter(x2226, x2227, x2221, x2222, x2223);

// Dealloc(X2600)
JCudaTensor x2228;
x2228 = x2194;
x2228.free();

// cv52_B <~~ V_cv52_B
float x2229, x2230;
x2229 = 1;
x2230 = 1;
JCudaTensor x2231;
x2231 = x2205;
x345.update(x2231, x2229, x2230);

// cv54_B <~~ V_cv54_B
float x2232, x2233;
x2232 = 1;
x2233 = 1;
JCudaTensor x2234;
x2234 = x2211;
x339.update(x2234, x2232, x2233);

// cv52_W <~~ V_cv52_W
float x2235, x2236;
x2235 = 1;
float x2237;
float x2238;
x2237 = 1;
float x2239;
float x2240;
float x2241;
float x2242;
x2241 = 1;
x2242 = decay;
x2239 = x2241 * x2242;
float x2243;
float x2244;
x2243 = 1;
x2244 = lrn_rate;
x2240 = x2243 * x2244;
x2238 = x2239 * x2240;
x2236 = x2237 + x2238;
JCudaTensor x2245;
x2245 = x2221;
x344.update(x2245, x2235, x2236);

// val X2626 = (X2602 + X2624 * d_Convolv(1,0)(cv54_W)/d_X7614)
JCudaTensor x2246;
JCudaTensor x2247;
x2247 = x2217;
JCudaTensor x2248, x2249;
x2248 = x2190;
x2249 = x338;
x2246 = x196.backward_data(x2248,x2249, x2247);

// Dealloc(X2624)
JCudaTensor x2250;
x2250 = x2190;
x2250.free();

// cv54_W <~~ V_cv54_W
float x2251, x2252;
x2251 = 1;
float x2253;
float x2254;
x2253 = 1;
float x2255;
float x2256;
float x2257;
float x2258;
x2257 = 1;
x2258 = decay;
x2255 = x2257 * x2258;
float x2259;
float x2260;
x2259 = 1;
x2260 = lrn_rate;
x2256 = x2259 * x2260;
x2254 = x2255 * x2256;
x2252 = x2253 + x2254;
JCudaTensor x2261;
x2261 = x2198;
x338.update(x2261, x2251, x2252);

// val X2648 = (X2626 + X2645 * d_Pooling(3,1,1,true)(X7625,X7614)/d_X7614)
JCudaTensor x2262;
JCudaTensor x2263;
x2263 = x2246;
JCudaTensor x2264, x2265, x2266;
x2264 = x2086;
x2265 = x346;
x2266 = x325;
x2262 = x189.backward(x2264,x2265,x2266, x2263);

// Dealloc(X2645)
JCudaTensor x2267;
x2267 = x2086;
x2267.free();

// Dealloc(X7625)
JCudaTensor x2268;
x2268 = x346;
x2268.free();

// Dealloc(X7614)
JCudaTensor x2269;
x2269 = x325;
x2269.free();

// val X2662 = Proj(X2648, X7602,X7606,X7610,X7613, 0)
JCudaTensor x2270;
JCudaTensor x2272;
x2272 = x2262;
JCudaTensor[] x2271 = x250.backward(x2272);
x2270 = x2271[0];

// val X2728 = Proj(X2648, X7602,X7606,X7610,X7613, 3)
JCudaTensor x2273;
x2273 = x2271[3];

// val X2680 = Proj(X2648, X7602,X7606,X7610,X7613, 1)
JCudaTensor x2274;
x2274 = x2271[1];

// val X2704 = Proj(X2648, X7602,X7606,X7610,X7613, 2)
JCudaTensor x2275;
x2275 = x2271[2];

// Dealloc(X2648)
JCudaTensor x2276;
x2276 = x2262;
x2276.free();

// val X2685 = X2680 * d_ReLU()(X7606)/d_X7605
JCudaTensor x2277;
JCudaTensor x2278, x2279;
x2278 = x2274;
x2279 = x321;
x2277 = x243.backward(x2278, x2279);

// Dealloc(X7606)
JCudaTensor x2280;
x2280 = x321;
x2280.free();

// val X2732 = X2728 * d_ReLU()(X7613)/d_X7612
JCudaTensor x2281;
JCudaTensor x2282, x2283;
x2282 = x2273;
x2283 = x323;
x2281 = x246.backward(x2282, x2283);

// Dealloc(X7613)
JCudaTensor x2284;
x2284 = x323;
x2284.free();

// val X2709 = X2704 * d_ReLU()(X7610)/d_X7609
JCudaTensor x2285;
JCudaTensor x2286, x2287;
x2286 = x2275;
x2287 = x312;
x2285 = x246.backward(x2286, x2287);

// Dealloc(X7610)
JCudaTensor x2288;
x2288 = x312;
x2288.free();

// val X2665 = X2662 * d_ReLU()(X7602)/d_X7601
JCudaTensor x2289;
JCudaTensor x2290, x2291;
x2290 = x2270;
x2291 = x310;
x2289 = x240.backward(x2290, x2291);

// Dealloc(X7602)
JCudaTensor x2292;
x2292 = x310;
x2292.free();

// V_cv43_B <~~ X2685 * d_Convolv(1,1)()/d_cv43_B
float x2294, x2295;
float x2296;
float x2297;
x2296 = 2;
x2297 = lrn_rate;
x2294 = x2296 * x2297;
x2295 = momentum;
JCudaTensor x2298;
x2298 = x2277;
x237.backward_bias(x2298, x2293, x2294, x2295);

// val X2686 = X2685 * d_Convolv(1,1)(cv43_W)/d_X7604
JCudaTensor x2299;
JCudaTensor x2300, x2301;
x2300 = x2277;
x2301 = x308;
x2299 = x237.backward_data(x2300, x2301);

// val X2733 = X2732 * d_Convolv(1,0)(cv46_W)/d_X7611
JCudaTensor x2302;
JCudaTensor x2303, x2304;
x2303 = x2281;
x2304 = x284;
x2302 = x223.backward_data(x2303, x2304);

// V_cv43_W <~~ X2685 * d_Convolv(1,1)(X7604)/d_cv43_W
float x2306, x2307;
float x2308;
float x2309;
x2308 = 1;
x2309 = lrn_rate;
x2306 = x2308 * x2309;
x2307 = momentum;
JCudaTensor x2310, x2311;
x2310 = x2277;
x2311 = x278;
x237.backward_filter(x2310, x2311, x2305, x2306, x2307);

// Dealloc(X2685)
JCudaTensor x2312;
x2312 = x2277;
x2312.free();

// V_cv45_W <~~ X2709 * d_Convolv(1,2)(X7608)/d_cv45_W
float x2314, x2315;
float x2316;
float x2317;
x2316 = 1;
x2317 = lrn_rate;
x2314 = x2316 * x2317;
x2315 = momentum;
JCudaTensor x2318, x2319;
x2318 = x2285;
x2319 = x293;
x230.backward_filter(x2318, x2319, x2313, x2314, x2315);

// V_cv45_B <~~ X2709 * d_Convolv(1,2)()/d_cv45_B
float x2321, x2322;
float x2323;
float x2324;
x2323 = 2;
x2324 = lrn_rate;
x2321 = x2323 * x2324;
x2322 = momentum;
JCudaTensor x2325;
x2325 = x2285;
x230.backward_bias(x2325, x2320, x2321, x2322);

// val X2666 = X2665 * d_Convolv(1,0)(cv41_W)/d_X7600
JCudaTensor x2326;
JCudaTensor x2327, x2328;
x2327 = x2289;
x2328 = x265;
x2326 = x210.backward_data(x2327, x2328);

// val X2710 = X2709 * d_Convolv(1,2)(cv45_W)/d_X7608
JCudaTensor x2329;
JCudaTensor x2330, x2331;
x2330 = x2285;
x2331 = x299;
x2329 = x230.backward_data(x2330, x2331);

// Dealloc(X2709)
JCudaTensor x2332;
x2332 = x2285;
x2332.free();

// V_cv41_W <~~ X2665 * d_Convolv(1,0)(X7600)/d_cv41_W
float x2334, x2335;
float x2336;
float x2337;
x2336 = 1;
x2337 = lrn_rate;
x2334 = x2336 * x2337;
x2335 = momentum;
JCudaTensor x2338, x2339;
x2338 = x2289;
x2339 = x249;
x210.backward_filter(x2338, x2339, x2333, x2334, x2335);

// V_cv41_B <~~ X2665 * d_Convolv(1,0)()/d_cv41_B
float x2341, x2342;
float x2343;
float x2344;
x2343 = 2;
x2344 = lrn_rate;
x2341 = x2343 * x2344;
x2342 = momentum;
JCudaTensor x2345;
x2345 = x2289;
x210.backward_bias(x2345, x2340, x2341, x2342);

// Dealloc(X2665)
JCudaTensor x2346;
x2346 = x2289;
x2346.free();

// V_cv46_W <~~ X2732 * d_Convolv(1,0)(X7611)/d_cv46_W
float x2348, x2349;
float x2350;
float x2351;
x2350 = 1;
x2351 = lrn_rate;
x2348 = x2350 * x2351;
x2349 = momentum;
JCudaTensor x2352, x2353;
x2352 = x2281;
x2353 = x270;
x223.backward_filter(x2352, x2353, x2347, x2348, x2349);

// V_cv46_B <~~ X2732 * d_Convolv(1,0)()/d_cv46_B
float x2355, x2356;
float x2357;
float x2358;
x2357 = 2;
x2358 = lrn_rate;
x2355 = x2357 * x2358;
x2356 = momentum;
JCudaTensor x2359;
x2359 = x2281;
x223.backward_bias(x2359, x2354, x2355, x2356);

// Dealloc(X2732)
JCudaTensor x2360;
x2360 = x2281;
x2360.free();

// cv41_W <~~ V_cv41_W
float x2361, x2362;
x2361 = 1;
float x2363;
float x2364;
x2363 = 1;
float x2365;
float x2366;
float x2367;
float x2368;
x2367 = 1;
x2368 = decay;
x2365 = x2367 * x2368;
float x2369;
float x2370;
x2369 = 1;
x2370 = lrn_rate;
x2366 = x2369 * x2370;
x2364 = x2365 * x2366;
x2362 = x2363 + x2364;
JCudaTensor x2371;
x2371 = x2333;
x265.update(x2371, x2361, x2362);

// cv43_B <~~ V_cv43_B
float x2372, x2373;
x2372 = 1;
x2373 = 1;
JCudaTensor x2374;
x2374 = x2293;
x309.update(x2374, x2372, x2373);

// cv41_B <~~ V_cv41_B
float x2375, x2376;
x2375 = 1;
x2376 = 1;
JCudaTensor x2377;
x2377 = x2340;
x266.update(x2377, x2375, x2376);

// cv45_W <~~ V_cv45_W
float x2378, x2379;
x2378 = 1;
float x2380;
float x2381;
x2380 = 1;
float x2382;
float x2383;
float x2384;
float x2385;
x2384 = 1;
x2385 = decay;
x2382 = x2384 * x2385;
float x2386;
float x2387;
x2386 = 1;
x2387 = lrn_rate;
x2383 = x2386 * x2387;
x2381 = x2382 * x2383;
x2379 = x2380 + x2381;
JCudaTensor x2388;
x2388 = x2313;
x299.update(x2388, x2378, x2379);

// cv46_B <~~ V_cv46_B
float x2389, x2390;
x2389 = 1;
x2390 = 1;
JCudaTensor x2391;
x2391 = x2354;
x285.update(x2391, x2389, x2390);

// cv43_W <~~ V_cv43_W
float x2392, x2393;
x2392 = 1;
float x2394;
float x2395;
x2394 = 1;
float x2396;
float x2397;
float x2398;
float x2399;
x2398 = 1;
x2399 = decay;
x2396 = x2398 * x2399;
float x2400;
float x2401;
x2400 = 1;
x2401 = lrn_rate;
x2397 = x2400 * x2401;
x2395 = x2396 * x2397;
x2393 = x2394 + x2395;
JCudaTensor x2402;
x2402 = x2305;
x308.update(x2402, x2392, x2393);

// cv45_B <~~ V_cv45_B
float x2403, x2404;
x2403 = 1;
x2404 = 1;
JCudaTensor x2405;
x2405 = x2320;
x300.update(x2405, x2403, x2404);

// cv46_W <~~ V_cv46_W
float x2406, x2407;
x2406 = 1;
float x2408;
float x2409;
x2408 = 1;
float x2410;
float x2411;
float x2412;
float x2413;
x2412 = 1;
x2413 = decay;
x2410 = x2412 * x2413;
float x2414;
float x2415;
x2414 = 1;
x2415 = lrn_rate;
x2411 = x2414 * x2415;
x2409 = x2410 * x2411;
x2407 = x2408 + x2409;
JCudaTensor x2416;
x2416 = x2347;
x284.update(x2416, x2406, x2407);

// val X2688 = X2686 * d_ReLU()(X7604)/d_X7603
JCudaTensor x2417;
JCudaTensor x2418, x2419;
x2418 = x2299;
x2419 = x278;
x2417 = x216.backward(x2418, x2419);

// Dealloc(X7604)
JCudaTensor x2420;
x2420 = x278;
x2420.free();

// val X2712 = X2710 * d_ReLU()(X7608)/d_X7607
JCudaTensor x2421;
JCudaTensor x2422, x2423;
x2422 = x2329;
x2423 = x293;
x2421 = x213.backward(x2422, x2423);

// Dealloc(X7608)
JCudaTensor x2424;
x2424 = x293;
x2424.free();

// V_cv44_W <~~ X2712 * d_Convolv(1,0)(X7600)/d_cv44_W
float x2426, x2427;
float x2428;
float x2429;
x2428 = 1;
x2429 = lrn_rate;
x2426 = x2428 * x2429;
x2427 = momentum;
JCudaTensor x2430, x2431;
x2430 = x2421;
x2431 = x249;
x196.backward_filter(x2430, x2431, x2425, x2426, x2427);

// val X2690 = (X2666 + X2688 * d_Convolv(1,0)(cv42_W)/d_X7600)
JCudaTensor x2432;
JCudaTensor x2433;
x2433 = x2326;
JCudaTensor x2434, x2435;
x2434 = x2417;
x2435 = x276;
x2432 = x203.backward_data(x2434,x2435, x2433);

// V_cv42_B <~~ X2688 * d_Convolv(1,0)()/d_cv42_B
float x2437, x2438;
float x2439;
float x2440;
x2439 = 2;
x2440 = lrn_rate;
x2437 = x2439 * x2440;
x2438 = momentum;
JCudaTensor x2441;
x2441 = x2417;
x203.backward_bias(x2441, x2436, x2437, x2438);

// V_cv42_W <~~ X2688 * d_Convolv(1,0)(X7600)/d_cv42_W
float x2443, x2444;
float x2445;
float x2446;
x2445 = 1;
x2446 = lrn_rate;
x2443 = x2445 * x2446;
x2444 = momentum;
JCudaTensor x2447, x2448;
x2447 = x2417;
x2448 = x249;
x203.backward_filter(x2447, x2448, x2442, x2443, x2444);

// Dealloc(X2688)
JCudaTensor x2449;
x2449 = x2417;
x2449.free();

// V_cv44_B <~~ X2712 * d_Convolv(1,0)()/d_cv44_B
float x2451, x2452;
float x2453;
float x2454;
x2453 = 2;
x2454 = lrn_rate;
x2451 = x2453 * x2454;
x2452 = momentum;
JCudaTensor x2455;
x2455 = x2421;
x196.backward_bias(x2455, x2450, x2451, x2452);

// cv42_B <~~ V_cv42_B
float x2456, x2457;
x2456 = 1;
x2457 = 1;
JCudaTensor x2458;
x2458 = x2436;
x277.update(x2458, x2456, x2457);

// cv42_W <~~ V_cv42_W
float x2459, x2460;
x2459 = 1;
float x2461;
float x2462;
x2461 = 1;
float x2463;
float x2464;
float x2465;
float x2466;
x2465 = 1;
x2466 = decay;
x2463 = x2465 * x2466;
float x2467;
float x2468;
x2467 = 1;
x2468 = lrn_rate;
x2464 = x2467 * x2468;
x2462 = x2463 * x2464;
x2460 = x2461 + x2462;
JCudaTensor x2469;
x2469 = x2442;
x276.update(x2469, x2459, x2460);

// cv44_B <~~ V_cv44_B
float x2470, x2471;
x2470 = 1;
x2471 = 1;
JCudaTensor x2472;
x2472 = x2450;
x260.update(x2472, x2470, x2471);

// val X2714 = (X2690 + X2712 * d_Convolv(1,0)(cv44_W)/d_X7600)
JCudaTensor x2473;
JCudaTensor x2474;
x2474 = x2432;
JCudaTensor x2475, x2476;
x2475 = x2421;
x2476 = x259;
x2473 = x196.backward_data(x2475,x2476, x2474);

// Dealloc(X2712)
JCudaTensor x2477;
x2477 = x2421;
x2477.free();

// cv44_W <~~ V_cv44_W
float x2478, x2479;
x2478 = 1;
float x2480;
float x2481;
x2480 = 1;
float x2482;
float x2483;
float x2484;
float x2485;
x2484 = 1;
x2485 = decay;
x2482 = x2484 * x2485;
float x2486;
float x2487;
x2486 = 1;
x2487 = lrn_rate;
x2483 = x2486 * x2487;
x2481 = x2482 * x2483;
x2479 = x2480 + x2481;
JCudaTensor x2488;
x2488 = x2425;
x259.update(x2488, x2478, x2479);

// val X2736 = (X2714 + X2733 * d_Pooling(3,1,1,true)(X7611,X7600)/d_X7600)
JCudaTensor x2489;
JCudaTensor x2490;
x2490 = x2473;
JCudaTensor x2491, x2492, x2493;
x2491 = x2302;
x2492 = x270;
x2493 = x249;
x2489 = x189.backward(x2491,x2492,x2493, x2490);

// Dealloc(X2733)
JCudaTensor x2494;
x2494 = x2302;
x2494.free();

// Dealloc(X7611)
JCudaTensor x2495;
x2495 = x270;
x2495.free();

// Dealloc(X7600)
JCudaTensor x2496;
x2496 = x249;
x2496.free();

// val X2772 = (X2771 * loss1)
JCudaTensor x2497;
JCudaTensor x2498;
float x2499;
x2498 = x604;
x2499 = loss1;
x2497 = x2498.times_i(x2499);

// val X2773 = (X2736 + X2772)
JCudaTensor x2500;
JCudaTensor x2501, x2502;
x2501 = x2489;
x2502 = x2497;
x2500 = x2501.plus_i(x2502);

// Dealloc(X2772)
JCudaTensor x2503;
x2503 = x2497;
x2503.free();

// val X4115 = Proj(X2773, X7588,X7592,X7596,X7599, 1)
JCudaTensor x2504;
JCudaTensor x2506;
x2506 = x2500;
JCudaTensor[] x2505 = x250.backward(x2506);
x2504 = x2505[1];

// val X4097 = Proj(X2773, X7588,X7592,X7596,X7599, 0)
JCudaTensor x2507;
x2507 = x2505[0];

// val X4139 = Proj(X2773, X7588,X7592,X7596,X7599, 2)
JCudaTensor x2508;
x2508 = x2505[2];

// val X4163 = Proj(X2773, X7588,X7592,X7596,X7599, 3)
JCudaTensor x2509;
x2509 = x2505[3];

// Dealloc(X2773)
JCudaTensor x2510;
x2510 = x2500;
x2510.free();

// val X4167 = X4163 * d_ReLU()(X7599)/d_X7598
JCudaTensor x2511;
JCudaTensor x2512, x2513;
x2512 = x2509;
x2513 = x247;
x2511 = x246.backward(x2512, x2513);

// Dealloc(X7599)
JCudaTensor x2514;
x2514 = x247;
x2514.free();

// val X4100 = X4097 * d_ReLU()(X7588)/d_X7587
JCudaTensor x2515;
JCudaTensor x2516, x2517;
x2516 = x2507;
x2517 = x238;
x2515 = x240.backward(x2516, x2517);

// Dealloc(X7588)
JCudaTensor x2518;
x2518 = x238;
x2518.free();

// val X4120 = X4115 * d_ReLU()(X7592)/d_X7591
JCudaTensor x2519;
JCudaTensor x2520, x2521;
x2520 = x2504;
x2521 = x241;
x2519 = x243.backward(x2520, x2521);

// Dealloc(X7592)
JCudaTensor x2522;
x2522 = x241;
x2522.free();

// val X4144 = X4139 * d_ReLU()(X7596)/d_X7595
JCudaTensor x2523;
JCudaTensor x2524, x2525;
x2524 = x2508;
x2525 = x244;
x2523 = x246.backward(x2524, x2525);

// Dealloc(X7596)
JCudaTensor x2526;
x2526 = x244;
x2526.free();

// val X4168 = X4167 * d_Convolv(1,0)(cv36_W)/d_X7597
JCudaTensor x2527;
JCudaTensor x2528, x2529;
x2528 = x2511;
x2529 = x221;
x2527 = x223.backward_data(x2528, x2529);

// V_cv31_B <~~ X4100 * d_Convolv(1,0)()/d_cv31_B
float x2531, x2532;
float x2533;
float x2534;
x2533 = 2;
x2534 = lrn_rate;
x2531 = x2533 * x2534;
x2532 = momentum;
JCudaTensor x2535;
x2535 = x2515;
x210.backward_bias(x2535, x2530, x2531, x2532);

// V_cv36_B <~~ X4167 * d_Convolv(1,0)()/d_cv36_B
float x2537, x2538;
float x2539;
float x2540;
x2539 = 2;
x2540 = lrn_rate;
x2537 = x2539 * x2540;
x2538 = momentum;
JCudaTensor x2541;
x2541 = x2511;
x223.backward_bias(x2541, x2536, x2537, x2538);

// V_cv31_W <~~ X4100 * d_Convolv(1,0)(X7586)/d_cv31_W
float x2543, x2544;
float x2545;
float x2546;
x2545 = 1;
x2546 = lrn_rate;
x2543 = x2545 * x2546;
x2544 = momentum;
JCudaTensor x2547, x2548;
x2547 = x2515;
x2548 = x184;
x210.backward_filter(x2547, x2548, x2542, x2543, x2544);

// V_cv33_W <~~ X4120 * d_Convolv(1,1)(X7590)/d_cv33_W
float x2550, x2551;
float x2552;
float x2553;
x2552 = 1;
x2553 = lrn_rate;
x2550 = x2552 * x2553;
x2551 = momentum;
JCudaTensor x2554, x2555;
x2554 = x2519;
x2555 = x214;
x237.backward_filter(x2554, x2555, x2549, x2550, x2551);

// V_cv36_W <~~ X4167 * d_Convolv(1,0)(X7597)/d_cv36_W
float x2557, x2558;
float x2559;
float x2560;
x2559 = 1;
x2560 = lrn_rate;
x2557 = x2559 * x2560;
x2558 = momentum;
JCudaTensor x2561, x2562;
x2561 = x2511;
x2562 = x187;
x223.backward_filter(x2561, x2562, x2556, x2557, x2558);

// Dealloc(X4167)
JCudaTensor x2563;
x2563 = x2511;
x2563.free();

// V_cv35_B <~~ X4144 * d_Convolv(1,2)()/d_cv35_B
float x2565, x2566;
float x2567;
float x2568;
x2567 = 2;
x2568 = lrn_rate;
x2565 = x2567 * x2568;
x2566 = momentum;
JCudaTensor x2569;
x2569 = x2523;
x230.backward_bias(x2569, x2564, x2565, x2566);

// V_cv35_W <~~ X4144 * d_Convolv(1,2)(X7594)/d_cv35_W
float x2571, x2572;
float x2573;
float x2574;
x2573 = 1;
x2574 = lrn_rate;
x2571 = x2573 * x2574;
x2572 = momentum;
JCudaTensor x2575, x2576;
x2575 = x2523;
x2576 = x211;
x230.backward_filter(x2575, x2576, x2570, x2571, x2572);

// val X4121 = X4120 * d_Convolv(1,1)(cv33_W)/d_X7590
JCudaTensor x2577;
JCudaTensor x2578, x2579;
x2578 = x2519;
x2579 = x235;
x2577 = x237.backward_data(x2578, x2579);

// val X4145 = X4144 * d_Convolv(1,2)(cv35_W)/d_X7594
JCudaTensor x2580;
JCudaTensor x2581, x2582;
x2581 = x2523;
x2582 = x228;
x2580 = x230.backward_data(x2581, x2582);

// Dealloc(X4144)
JCudaTensor x2583;
x2583 = x2523;
x2583.free();

// val X4101 = X4100 * d_Convolv(1,0)(cv31_W)/d_X7586
JCudaTensor x2584;
JCudaTensor x2585, x2586;
x2585 = x2515;
x2586 = x208;
x2584 = x210.backward_data(x2585, x2586);

// Dealloc(X4100)
JCudaTensor x2587;
x2587 = x2515;
x2587.free();

// V_cv33_B <~~ X4120 * d_Convolv(1,1)()/d_cv33_B
float x2589, x2590;
float x2591;
float x2592;
x2591 = 2;
x2592 = lrn_rate;
x2589 = x2591 * x2592;
x2590 = momentum;
JCudaTensor x2593;
x2593 = x2519;
x237.backward_bias(x2593, x2588, x2589, x2590);

// Dealloc(X4120)
JCudaTensor x2594;
x2594 = x2519;
x2594.free();

// cv33_B <~~ V_cv33_B
float x2595, x2596;
x2595 = 1;
x2596 = 1;
JCudaTensor x2597;
x2597 = x2588;
x236.update(x2597, x2595, x2596);

// cv33_W <~~ V_cv33_W
float x2598, x2599;
x2598 = 1;
float x2600;
float x2601;
x2600 = 1;
float x2602;
float x2603;
float x2604;
float x2605;
x2604 = 1;
x2605 = decay;
x2602 = x2604 * x2605;
float x2606;
float x2607;
x2606 = 1;
x2607 = lrn_rate;
x2603 = x2606 * x2607;
x2601 = x2602 * x2603;
x2599 = x2600 + x2601;
JCudaTensor x2608;
x2608 = x2549;
x235.update(x2608, x2598, x2599);

// cv31_B <~~ V_cv31_B
float x2609, x2610;
x2609 = 1;
x2610 = 1;
JCudaTensor x2611;
x2611 = x2530;
x209.update(x2611, x2609, x2610);

// cv31_W <~~ V_cv31_W
float x2612, x2613;
x2612 = 1;
float x2614;
float x2615;
x2614 = 1;
float x2616;
float x2617;
float x2618;
float x2619;
x2618 = 1;
x2619 = decay;
x2616 = x2618 * x2619;
float x2620;
float x2621;
x2620 = 1;
x2621 = lrn_rate;
x2617 = x2620 * x2621;
x2615 = x2616 * x2617;
x2613 = x2614 + x2615;
JCudaTensor x2622;
x2622 = x2542;
x208.update(x2622, x2612, x2613);

// cv36_B <~~ V_cv36_B
float x2623, x2624;
x2623 = 1;
x2624 = 1;
JCudaTensor x2625;
x2625 = x2536;
x222.update(x2625, x2623, x2624);

// cv35_B <~~ V_cv35_B
float x2626, x2627;
x2626 = 1;
x2627 = 1;
JCudaTensor x2628;
x2628 = x2564;
x229.update(x2628, x2626, x2627);

// cv35_W <~~ V_cv35_W
float x2629, x2630;
x2629 = 1;
float x2631;
float x2632;
x2631 = 1;
float x2633;
float x2634;
float x2635;
float x2636;
x2635 = 1;
x2636 = decay;
x2633 = x2635 * x2636;
float x2637;
float x2638;
x2637 = 1;
x2638 = lrn_rate;
x2634 = x2637 * x2638;
x2632 = x2633 * x2634;
x2630 = x2631 + x2632;
JCudaTensor x2639;
x2639 = x2570;
x228.update(x2639, x2629, x2630);

// cv36_W <~~ V_cv36_W
float x2640, x2641;
x2640 = 1;
float x2642;
float x2643;
x2642 = 1;
float x2644;
float x2645;
float x2646;
float x2647;
x2646 = 1;
x2647 = decay;
x2644 = x2646 * x2647;
float x2648;
float x2649;
x2648 = 1;
x2649 = lrn_rate;
x2645 = x2648 * x2649;
x2643 = x2644 * x2645;
x2641 = x2642 + x2643;
JCudaTensor x2650;
x2650 = x2556;
x221.update(x2650, x2640, x2641);

// val X4123 = X4121 * d_ReLU()(X7590)/d_X7589
JCudaTensor x2651;
JCudaTensor x2652, x2653;
x2652 = x2577;
x2653 = x214;
x2651 = x216.backward(x2652, x2653);

// Dealloc(X7590)
JCudaTensor x2654;
x2654 = x214;
x2654.free();

// val X4147 = X4145 * d_ReLU()(X7594)/d_X7593
JCudaTensor x2655;
JCudaTensor x2656, x2657;
x2656 = x2580;
x2657 = x211;
x2655 = x213.backward(x2656, x2657);

// Dealloc(X7594)
JCudaTensor x2658;
x2658 = x211;
x2658.free();

// V_cv34_W <~~ X4147 * d_Convolv(1,0)(X7586)/d_cv34_W
float x2660, x2661;
float x2662;
float x2663;
x2662 = 1;
x2663 = lrn_rate;
x2660 = x2662 * x2663;
x2661 = momentum;
JCudaTensor x2664, x2665;
x2664 = x2655;
x2665 = x184;
x196.backward_filter(x2664, x2665, x2659, x2660, x2661);

// V_cv34_B <~~ X4147 * d_Convolv(1,0)()/d_cv34_B
float x2667, x2668;
float x2669;
float x2670;
x2669 = 2;
x2670 = lrn_rate;
x2667 = x2669 * x2670;
x2668 = momentum;
JCudaTensor x2671;
x2671 = x2655;
x196.backward_bias(x2671, x2666, x2667, x2668);

// V_cv32_B <~~ X4123 * d_Convolv(1,0)()/d_cv32_B
float x2673, x2674;
float x2675;
float x2676;
x2675 = 2;
x2676 = lrn_rate;
x2673 = x2675 * x2676;
x2674 = momentum;
JCudaTensor x2677;
x2677 = x2651;
x203.backward_bias(x2677, x2672, x2673, x2674);

// val X4125 = (X4101 + X4123 * d_Convolv(1,0)(cv32_W)/d_X7586)
JCudaTensor x2678;
JCudaTensor x2679;
x2679 = x2584;
JCudaTensor x2680, x2681;
x2680 = x2651;
x2681 = x201;
x2678 = x203.backward_data(x2680,x2681, x2679);

// V_cv32_W <~~ X4123 * d_Convolv(1,0)(X7586)/d_cv32_W
float x2683, x2684;
float x2685;
float x2686;
x2685 = 1;
x2686 = lrn_rate;
x2683 = x2685 * x2686;
x2684 = momentum;
JCudaTensor x2687, x2688;
x2687 = x2651;
x2688 = x184;
x203.backward_filter(x2687, x2688, x2682, x2683, x2684);

// Dealloc(X4123)
JCudaTensor x2689;
x2689 = x2651;
x2689.free();

// cv34_B <~~ V_cv34_B
float x2690, x2691;
x2690 = 1;
x2691 = 1;
JCudaTensor x2692;
x2692 = x2666;
x195.update(x2692, x2690, x2691);

// cv32_B <~~ V_cv32_B
float x2693, x2694;
x2693 = 1;
x2694 = 1;
JCudaTensor x2695;
x2695 = x2672;
x202.update(x2695, x2693, x2694);

// cv32_W <~~ V_cv32_W
float x2696, x2697;
x2696 = 1;
float x2698;
float x2699;
x2698 = 1;
float x2700;
float x2701;
float x2702;
float x2703;
x2702 = 1;
x2703 = decay;
x2700 = x2702 * x2703;
float x2704;
float x2705;
x2704 = 1;
x2705 = lrn_rate;
x2701 = x2704 * x2705;
x2699 = x2700 * x2701;
x2697 = x2698 + x2699;
JCudaTensor x2706;
x2706 = x2682;
x201.update(x2706, x2696, x2697);

// val X4149 = (X4125 + X4147 * d_Convolv(1,0)(cv34_W)/d_X7586)
JCudaTensor x2707;
JCudaTensor x2708;
x2708 = x2678;
JCudaTensor x2709, x2710;
x2709 = x2655;
x2710 = x194;
x2707 = x196.backward_data(x2709,x2710, x2708);

// Dealloc(X4147)
JCudaTensor x2711;
x2711 = x2655;
x2711.free();

// cv34_W <~~ V_cv34_W
float x2712, x2713;
x2712 = 1;
float x2714;
float x2715;
x2714 = 1;
float x2716;
float x2717;
float x2718;
float x2719;
x2718 = 1;
x2719 = decay;
x2716 = x2718 * x2719;
float x2720;
float x2721;
x2720 = 1;
x2721 = lrn_rate;
x2717 = x2720 * x2721;
x2715 = x2716 * x2717;
x2713 = x2714 + x2715;
JCudaTensor x2722;
x2722 = x2659;
x194.update(x2722, x2712, x2713);

// val X4171 = (X4149 + X4168 * d_Pooling(3,1,1,true)(X7597,X7586)/d_X7586)
JCudaTensor x2723;
JCudaTensor x2724;
x2724 = x2707;
JCudaTensor x2725, x2726, x2727;
x2725 = x2527;
x2726 = x187;
x2727 = x184;
x2723 = x189.backward(x2725,x2726,x2727, x2724);

// Dealloc(X4168)
JCudaTensor x2728;
x2728 = x2527;
x2728.free();

// Dealloc(X7597)
JCudaTensor x2729;
x2729 = x187;
x2729.free();

// val X4173 = X4171 * d_Pooling(3,2,1,true)(X7586,X7585)/d_X7585
JCudaTensor x2730;
JCudaTensor x2731, x2732, x2733;
x2731 = x2723;
x2732 = x184;
x2733 = x179;
x2730 = x186.backward(x2731, x2732, x2733);

// Dealloc(X4171)
JCudaTensor x2734;
x2734 = x2723;
x2734.free();

// Dealloc(X7586)
JCudaTensor x2735;
x2735 = x184;
x2735.free();

// Dealloc(X7585)
JCudaTensor x2736;
x2736 = x179;
x2736.free();

// val X4229 = Proj(X4173, X7573,X7577,X7581,X7584, 2)
JCudaTensor x2737;
JCudaTensor x2739;
x2739 = x2730;
JCudaTensor[] x2738 = x119.backward(x2739);
x2737 = x2738[2];

// val X4187 = Proj(X4173, X7573,X7577,X7581,X7584, 0)
JCudaTensor x2740;
x2740 = x2738[0];

// val X4205 = Proj(X4173, X7573,X7577,X7581,X7584, 1)
JCudaTensor x2741;
x2741 = x2738[1];

// val X4253 = Proj(X4173, X7573,X7577,X7581,X7584, 3)
JCudaTensor x2742;
x2742 = x2738[3];

// Dealloc(X4173)
JCudaTensor x2743;
x2743 = x2730;
x2743.free();

// val X4257 = X4253 * d_ReLU()(X7584)/d_X7583
JCudaTensor x2744;
JCudaTensor x2745, x2746;
x2745 = x2742;
x2746 = x177;
x2744 = x115.backward(x2745, x2746);

// Dealloc(X7584)
JCudaTensor x2747;
x2747 = x177;
x2747.free();

// val X4190 = X4187 * d_ReLU()(X7573)/d_X7572
JCudaTensor x2748;
JCudaTensor x2749, x2750;
x2749 = x2740;
x2750 = x171;
x2748 = x109.backward(x2749, x2750);

// Dealloc(X7573)
JCudaTensor x2751;
x2751 = x171;
x2751.free();

// val X4234 = X4229 * d_ReLU()(X7581)/d_X7580
JCudaTensor x2752;
JCudaTensor x2753, x2754;
x2753 = x2737;
x2754 = x175;
x2752 = x115.backward(x2753, x2754);

// Dealloc(X7581)
JCudaTensor x2755;
x2755 = x175;
x2755.free();

// val X4210 = X4205 * d_ReLU()(X7577)/d_X7576
JCudaTensor x2756;
JCudaTensor x2757, x2758;
x2757 = x2741;
x2758 = x173;
x2756 = x112.backward(x2757, x2758);

// Dealloc(X7577)
JCudaTensor x2759;
x2759 = x173;
x2759.free();

// val X4258 = X4257 * d_Convolv(1,0)(cv26_W)/d_X7582
JCudaTensor x2760;
JCudaTensor x2761, x2762;
x2761 = x2744;
x2762 = x152;
x2760 = x154.backward_data(x2761, x2762);

// V_cv21_W <~~ X4190 * d_Convolv(1,0)(X7571)/d_cv21_W
float x2764, x2765;
float x2766;
float x2767;
x2766 = 1;
x2767 = lrn_rate;
x2764 = x2766 * x2767;
x2765 = momentum;
JCudaTensor x2768, x2769;
x2768 = x2748;
x2769 = x118;
x140.backward_filter(x2768, x2769, x2763, x2764, x2765);

// V_cv25_B <~~ X4234 * d_Convolv(1,2)()/d_cv25_B
float x2771, x2772;
float x2773;
float x2774;
x2773 = 2;
x2774 = lrn_rate;
x2771 = x2773 * x2774;
x2772 = momentum;
JCudaTensor x2775;
x2775 = x2752;
x106.backward_bias(x2775, x2770, x2771, x2772);

// V_cv26_W <~~ X4257 * d_Convolv(1,0)(X7582)/d_cv26_W
float x2777, x2778;
float x2779;
float x2780;
x2779 = 1;
x2780 = lrn_rate;
x2777 = x2779 * x2780;
x2778 = momentum;
JCudaTensor x2781, x2782;
x2781 = x2744;
x2782 = x124;
x154.backward_filter(x2781, x2782, x2776, x2777, x2778);

// V_cv23_B <~~ X4210 * d_Convolv(1,1)()/d_cv23_B
float x2784, x2785;
float x2786;
float x2787;
x2786 = 2;
x2787 = lrn_rate;
x2784 = x2786 * x2787;
x2785 = momentum;
JCudaTensor x2788;
x2788 = x2756;
x99.backward_bias(x2788, x2783, x2784, x2785);

// val X4211 = X4210 * d_Convolv(1,1)(cv23_W)/d_X7575
JCudaTensor x2789;
JCudaTensor x2790, x2791;
x2790 = x2756;
x2791 = x169;
x2789 = x99.backward_data(x2790, x2791);

// V_cv23_W <~~ X4210 * d_Convolv(1,1)(X7575)/d_cv23_W
float x2793, x2794;
float x2795;
float x2796;
x2795 = 1;
x2796 = lrn_rate;
x2793 = x2795 * x2796;
x2794 = momentum;
JCudaTensor x2797, x2798;
x2797 = x2756;
x2798 = x157;
x99.backward_filter(x2797, x2798, x2792, x2793, x2794);

// Dealloc(X4210)
JCudaTensor x2799;
x2799 = x2756;
x2799.free();

// val X4235 = X4234 * d_Convolv(1,2)(cv25_W)/d_X7579
JCudaTensor x2800;
JCudaTensor x2801, x2802;
x2801 = x2752;
x2802 = x163;
x2800 = x106.backward_data(x2801, x2802);

// V_cv26_B <~~ X4257 * d_Convolv(1,0)()/d_cv26_B
float x2804, x2805;
float x2806;
float x2807;
x2806 = 2;
x2807 = lrn_rate;
x2804 = x2806 * x2807;
x2805 = momentum;
JCudaTensor x2808;
x2808 = x2744;
x154.backward_bias(x2808, x2803, x2804, x2805);

// Dealloc(X4257)
JCudaTensor x2809;
x2809 = x2744;
x2809.free();

// V_cv25_W <~~ X4234 * d_Convolv(1,2)(X7579)/d_cv25_W
float x2811, x2812;
float x2813;
float x2814;
x2813 = 1;
x2814 = lrn_rate;
x2811 = x2813 * x2814;
x2812 = momentum;
JCudaTensor x2815, x2816;
x2815 = x2752;
x2816 = x155;
x106.backward_filter(x2815, x2816, x2810, x2811, x2812);

// Dealloc(X4234)
JCudaTensor x2817;
x2817 = x2752;
x2817.free();

// val X4191 = X4190 * d_Convolv(1,0)(cv21_W)/d_X7571
JCudaTensor x2818;
JCudaTensor x2819, x2820;
x2819 = x2748;
x2820 = x138;
x2818 = x140.backward_data(x2819, x2820);

// V_cv21_B <~~ X4190 * d_Convolv(1,0)()/d_cv21_B
float x2822, x2823;
float x2824;
float x2825;
x2824 = 2;
x2825 = lrn_rate;
x2822 = x2824 * x2825;
x2823 = momentum;
JCudaTensor x2826;
x2826 = x2748;
x140.backward_bias(x2826, x2821, x2822, x2823);

// Dealloc(X4190)
JCudaTensor x2827;
x2827 = x2748;
x2827.free();

// cv23_W <~~ V_cv23_W
float x2828, x2829;
x2828 = 1;
float x2830;
float x2831;
x2830 = 1;
float x2832;
float x2833;
float x2834;
float x2835;
x2834 = 1;
x2835 = decay;
x2832 = x2834 * x2835;
float x2836;
float x2837;
x2836 = 1;
x2837 = lrn_rate;
x2833 = x2836 * x2837;
x2831 = x2832 * x2833;
x2829 = x2830 + x2831;
JCudaTensor x2838;
x2838 = x2792;
x169.update(x2838, x2828, x2829);

// cv25_W <~~ V_cv25_W
float x2839, x2840;
x2839 = 1;
float x2841;
float x2842;
x2841 = 1;
float x2843;
float x2844;
float x2845;
float x2846;
x2845 = 1;
x2846 = decay;
x2843 = x2845 * x2846;
float x2847;
float x2848;
x2847 = 1;
x2848 = lrn_rate;
x2844 = x2847 * x2848;
x2842 = x2843 * x2844;
x2840 = x2841 + x2842;
JCudaTensor x2849;
x2849 = x2810;
x163.update(x2849, x2839, x2840);

// cv23_B <~~ V_cv23_B
float x2850, x2851;
x2850 = 1;
x2851 = 1;
JCudaTensor x2852;
x2852 = x2783;
x170.update(x2852, x2850, x2851);

// cv25_B <~~ V_cv25_B
float x2853, x2854;
x2853 = 1;
x2854 = 1;
JCudaTensor x2855;
x2855 = x2770;
x164.update(x2855, x2853, x2854);

// cv26_W <~~ V_cv26_W
float x2856, x2857;
x2856 = 1;
float x2858;
float x2859;
x2858 = 1;
float x2860;
float x2861;
float x2862;
float x2863;
x2862 = 1;
x2863 = decay;
x2860 = x2862 * x2863;
float x2864;
float x2865;
x2864 = 1;
x2865 = lrn_rate;
x2861 = x2864 * x2865;
x2859 = x2860 * x2861;
x2857 = x2858 + x2859;
JCudaTensor x2866;
x2866 = x2776;
x152.update(x2866, x2856, x2857);

// cv26_B <~~ V_cv26_B
float x2867, x2868;
x2867 = 1;
x2868 = 1;
JCudaTensor x2869;
x2869 = x2803;
x153.update(x2869, x2867, x2868);

// cv21_W <~~ V_cv21_W
float x2870, x2871;
x2870 = 1;
float x2872;
float x2873;
x2872 = 1;
float x2874;
float x2875;
float x2876;
float x2877;
x2876 = 1;
x2877 = decay;
x2874 = x2876 * x2877;
float x2878;
float x2879;
x2878 = 1;
x2879 = lrn_rate;
x2875 = x2878 * x2879;
x2873 = x2874 * x2875;
x2871 = x2872 + x2873;
JCudaTensor x2880;
x2880 = x2763;
x138.update(x2880, x2870, x2871);

// cv21_B <~~ V_cv21_B
float x2881, x2882;
x2881 = 1;
x2882 = 1;
JCudaTensor x2883;
x2883 = x2821;
x139.update(x2883, x2881, x2882);

// val X4213 = X4211 * d_ReLU()(X7575)/d_X7574
JCudaTensor x2884;
JCudaTensor x2885, x2886;
x2885 = x2789;
x2886 = x157;
x2884 = x89.backward(x2885, x2886);

// Dealloc(X7575)
JCudaTensor x2887;
x2887 = x157;
x2887.free();

// val X4237 = X4235 * d_ReLU()(X7579)/d_X7578
JCudaTensor x2888;
JCudaTensor x2889, x2890;
x2889 = x2800;
x2890 = x155;
x2888 = x92.backward(x2889, x2890);

// Dealloc(X7579)
JCudaTensor x2891;
x2891 = x155;
x2891.free();

// V_cv24_W <~~ X4237 * d_Convolv(1,0)(X7571)/d_cv24_W
float x2893, x2894;
float x2895;
float x2896;
x2895 = 1;
x2896 = lrn_rate;
x2893 = x2895 * x2896;
x2894 = momentum;
JCudaTensor x2897, x2898;
x2897 = x2888;
x2898 = x118;
x133.backward_filter(x2897, x2898, x2892, x2893, x2894);

// V_cv24_B <~~ X4237 * d_Convolv(1,0)()/d_cv24_B
float x2900, x2901;
float x2902;
float x2903;
x2902 = 2;
x2903 = lrn_rate;
x2900 = x2902 * x2903;
x2901 = momentum;
JCudaTensor x2904;
x2904 = x2888;
x133.backward_bias(x2904, x2899, x2900, x2901);

// V_cv22_B <~~ X4213 * d_Convolv(1,0)()/d_cv22_B
float x2906, x2907;
float x2908;
float x2909;
x2908 = 2;
x2909 = lrn_rate;
x2906 = x2908 * x2909;
x2907 = momentum;
JCudaTensor x2910;
x2910 = x2884;
x147.backward_bias(x2910, x2905, x2906, x2907);

// val X4215 = (X4191 + X4213 * d_Convolv(1,0)(cv22_W)/d_X7571)
JCudaTensor x2911;
JCudaTensor x2912;
x2912 = x2818;
JCudaTensor x2913, x2914;
x2913 = x2884;
x2914 = x145;
x2911 = x147.backward_data(x2913,x2914, x2912);

// V_cv22_W <~~ X4213 * d_Convolv(1,0)(X7571)/d_cv22_W
float x2916, x2917;
float x2918;
float x2919;
x2918 = 1;
x2919 = lrn_rate;
x2916 = x2918 * x2919;
x2917 = momentum;
JCudaTensor x2920, x2921;
x2920 = x2884;
x2921 = x118;
x147.backward_filter(x2920, x2921, x2915, x2916, x2917);

// Dealloc(X4213)
JCudaTensor x2922;
x2922 = x2884;
x2922.free();

// cv24_B <~~ V_cv24_B
float x2923, x2924;
x2923 = 1;
x2924 = 1;
JCudaTensor x2925;
x2925 = x2899;
x132.update(x2925, x2923, x2924);

// cv22_B <~~ V_cv22_B
float x2926, x2927;
x2926 = 1;
x2927 = 1;
JCudaTensor x2928;
x2928 = x2905;
x146.update(x2928, x2926, x2927);

// cv22_W <~~ V_cv22_W
float x2929, x2930;
x2929 = 1;
float x2931;
float x2932;
x2931 = 1;
float x2933;
float x2934;
float x2935;
float x2936;
x2935 = 1;
x2936 = decay;
x2933 = x2935 * x2936;
float x2937;
float x2938;
x2937 = 1;
x2938 = lrn_rate;
x2934 = x2937 * x2938;
x2932 = x2933 * x2934;
x2930 = x2931 + x2932;
JCudaTensor x2939;
x2939 = x2915;
x145.update(x2939, x2929, x2930);

// val X4239 = (X4215 + X4237 * d_Convolv(1,0)(cv24_W)/d_X7571)
JCudaTensor x2940;
JCudaTensor x2941;
x2941 = x2911;
JCudaTensor x2942, x2943;
x2942 = x2888;
x2943 = x131;
x2940 = x133.backward_data(x2942,x2943, x2941);

// Dealloc(X4237)
JCudaTensor x2944;
x2944 = x2888;
x2944.free();

// cv24_W <~~ V_cv24_W
float x2945, x2946;
x2945 = 1;
float x2947;
float x2948;
x2947 = 1;
float x2949;
float x2950;
float x2951;
float x2952;
x2951 = 1;
x2952 = decay;
x2949 = x2951 * x2952;
float x2953;
float x2954;
x2953 = 1;
x2954 = lrn_rate;
x2950 = x2953 * x2954;
x2948 = x2949 * x2950;
x2946 = x2947 + x2948;
JCudaTensor x2955;
x2955 = x2892;
x131.update(x2955, x2945, x2946);

// val X4261 = (X4239 + X4258 * d_Pooling(3,1,1,true)(X7582,X7571)/d_X7571)
JCudaTensor x2956;
JCudaTensor x2957;
x2957 = x2940;
JCudaTensor x2958, x2959, x2960;
x2958 = x2760;
x2959 = x124;
x2960 = x118;
x2956 = x126.backward(x2958,x2959,x2960, x2957);

// Dealloc(X4258)
JCudaTensor x2961;
x2961 = x2760;
x2961.free();

// Dealloc(X7582)
JCudaTensor x2962;
x2962 = x124;
x2962.free();

// Dealloc(X7571)
JCudaTensor x2963;
x2963 = x118;
x2963.free();

// val X4275 = Proj(X4261, X7559,X7563,X7567,X7570, 0)
JCudaTensor x2964;
JCudaTensor x2966;
x2966 = x2956;
JCudaTensor[] x2965 = x119.backward(x2966);
x2964 = x2965[0];

// val X4341 = Proj(X4261, X7559,X7563,X7567,X7570, 3)
JCudaTensor x2967;
x2967 = x2965[3];

// val X4293 = Proj(X4261, X7559,X7563,X7567,X7570, 1)
JCudaTensor x2968;
x2968 = x2965[1];

// val X4317 = Proj(X4261, X7559,X7563,X7567,X7570, 2)
JCudaTensor x2969;
x2969 = x2965[2];

// Dealloc(X4261)
JCudaTensor x2970;
x2970 = x2956;
x2970.free();

// val X4298 = X4293 * d_ReLU()(X7563)/d_X7562
JCudaTensor x2971;
JCudaTensor x2972, x2973;
x2972 = x2968;
x2973 = x110;
x2971 = x112.backward(x2972, x2973);

// Dealloc(X7563)
JCudaTensor x2974;
x2974 = x110;
x2974.free();

// val X4278 = X4275 * d_ReLU()(X7559)/d_X7558
JCudaTensor x2975;
JCudaTensor x2976, x2977;
x2976 = x2964;
x2977 = x107;
x2975 = x109.backward(x2976, x2977);

// Dealloc(X7559)
JCudaTensor x2978;
x2978 = x107;
x2978.free();

// val X4345 = X4341 * d_ReLU()(X7570)/d_X7569
JCudaTensor x2979;
JCudaTensor x2980, x2981;
x2980 = x2967;
x2981 = x116;
x2979 = x115.backward(x2980, x2981);

// Dealloc(X7570)
JCudaTensor x2982;
x2982 = x116;
x2982.free();

// val X4322 = X4317 * d_ReLU()(X7567)/d_X7566
JCudaTensor x2983;
JCudaTensor x2984, x2985;
x2984 = x2969;
x2985 = x113;
x2983 = x115.backward(x2984, x2985);

// Dealloc(X7567)
JCudaTensor x2986;
x2986 = x113;
x2986.free();

// V_cv13_B <~~ X4298 * d_Convolv(1,1)()/d_cv13_B
float x2988, x2989;
float x2990;
float x2991;
x2990 = 2;
x2991 = lrn_rate;
x2988 = x2990 * x2991;
x2989 = momentum;
JCudaTensor x2992;
x2992 = x2971;
x99.backward_bias(x2992, x2987, x2988, x2989);

// V_cv11_B <~~ X4278 * d_Convolv(1,0)()/d_cv11_B
float x2994, x2995;
float x2996;
float x2997;
x2996 = 2;
x2997 = lrn_rate;
x2994 = x2996 * x2997;
x2995 = momentum;
JCudaTensor x2998;
x2998 = x2975;
x72.backward_bias(x2998, x2993, x2994, x2995);

// V_cv16_W <~~ X4345 * d_Convolv(1,0)(X7568)/d_cv16_W
float x3000, x3001;
float x3002;
float x3003;
x3002 = 1;
x3003 = lrn_rate;
x3000 = x3002 * x3003;
x3001 = momentum;
JCudaTensor x3004, x3005;
x3004 = x2979;
x3005 = x63;
x86.backward_filter(x3004, x3005, x2999, x3000, x3001);

// V_cv16_B <~~ X4345 * d_Convolv(1,0)()/d_cv16_B
float x3007, x3008;
float x3009;
float x3010;
x3009 = 2;
x3010 = lrn_rate;
x3007 = x3009 * x3010;
x3008 = momentum;
JCudaTensor x3011;
x3011 = x2979;
x86.backward_bias(x3011, x3006, x3007, x3008);

// V_cv15_W <~~ X4322 * d_Convolv(1,2)(X7565)/d_cv15_W
float x3013, x3014;
float x3015;
float x3016;
x3015 = 1;
x3016 = lrn_rate;
x3013 = x3015 * x3016;
x3014 = momentum;
JCudaTensor x3017, x3018;
x3017 = x2983;
x3018 = x90;
x106.backward_filter(x3017, x3018, x3012, x3013, x3014);

// val X4323 = X4322 * d_Convolv(1,2)(cv15_W)/d_X7565
JCudaTensor x3019;
JCudaTensor x3020, x3021;
x3020 = x2983;
x3021 = x104;
x3019 = x106.backward_data(x3020, x3021);

// V_cv13_W <~~ X4298 * d_Convolv(1,1)(X7561)/d_cv13_W
float x3023, x3024;
float x3025;
float x3026;
x3025 = 1;
x3026 = lrn_rate;
x3023 = x3025 * x3026;
x3024 = momentum;
JCudaTensor x3027, x3028;
x3027 = x2971;
x3028 = x87;
x99.backward_filter(x3027, x3028, x3022, x3023, x3024);

// val X4279 = X4278 * d_Convolv(1,0)(cv11_W)/d_X7557
JCudaTensor x3029;
JCudaTensor x3030, x3031;
x3030 = x2975;
x3031 = x70;
x3029 = x72.backward_data(x3030, x3031);

// V_cv11_W <~~ X4278 * d_Convolv(1,0)(X7557)/d_cv11_W
float x3033, x3034;
float x3035;
float x3036;
x3035 = 1;
x3036 = lrn_rate;
x3033 = x3035 * x3036;
x3034 = momentum;
JCudaTensor x3037, x3038;
x3037 = x2975;
x3038 = x53;
x72.backward_filter(x3037, x3038, x3032, x3033, x3034);

// Dealloc(X4278)
JCudaTensor x3039;
x3039 = x2975;
x3039.free();

// V_cv15_B <~~ X4322 * d_Convolv(1,2)()/d_cv15_B
float x3041, x3042;
float x3043;
float x3044;
x3043 = 2;
x3044 = lrn_rate;
x3041 = x3043 * x3044;
x3042 = momentum;
JCudaTensor x3045;
x3045 = x2983;
x106.backward_bias(x3045, x3040, x3041, x3042);

// Dealloc(X4322)
JCudaTensor x3046;
x3046 = x2983;
x3046.free();

// val X4299 = X4298 * d_Convolv(1,1)(cv13_W)/d_X7561
JCudaTensor x3047;
JCudaTensor x3048, x3049;
x3048 = x2971;
x3049 = x97;
x3047 = x99.backward_data(x3048, x3049);

// Dealloc(X4298)
JCudaTensor x3050;
x3050 = x2971;
x3050.free();

// val X4346 = X4345 * d_Convolv(1,0)(cv16_W)/d_X7568
JCudaTensor x3051;
JCudaTensor x3052, x3053;
x3052 = x2979;
x3053 = x84;
x3051 = x86.backward_data(x3052, x3053);

// Dealloc(X4345)
JCudaTensor x3054;
x3054 = x2979;
x3054.free();

// cv11_W <~~ V_cv11_W
float x3055, x3056;
x3055 = 1;
float x3057;
float x3058;
x3057 = 1;
float x3059;
float x3060;
float x3061;
float x3062;
x3061 = 1;
x3062 = decay;
x3059 = x3061 * x3062;
float x3063;
float x3064;
x3063 = 1;
x3064 = lrn_rate;
x3060 = x3063 * x3064;
x3058 = x3059 * x3060;
x3056 = x3057 + x3058;
JCudaTensor x3065;
x3065 = x3032;
x70.update(x3065, x3055, x3056);

// cv15_B <~~ V_cv15_B
float x3066, x3067;
x3066 = 1;
x3067 = 1;
JCudaTensor x3068;
x3068 = x3040;
x105.update(x3068, x3066, x3067);

// cv13_B <~~ V_cv13_B
float x3069, x3070;
x3069 = 1;
x3070 = 1;
JCudaTensor x3071;
x3071 = x2987;
x98.update(x3071, x3069, x3070);

// cv11_B <~~ V_cv11_B
float x3072, x3073;
x3072 = 1;
x3073 = 1;
JCudaTensor x3074;
x3074 = x2993;
x71.update(x3074, x3072, x3073);

// cv13_W <~~ V_cv13_W
float x3075, x3076;
x3075 = 1;
float x3077;
float x3078;
x3077 = 1;
float x3079;
float x3080;
float x3081;
float x3082;
x3081 = 1;
x3082 = decay;
x3079 = x3081 * x3082;
float x3083;
float x3084;
x3083 = 1;
x3084 = lrn_rate;
x3080 = x3083 * x3084;
x3078 = x3079 * x3080;
x3076 = x3077 + x3078;
JCudaTensor x3085;
x3085 = x3022;
x97.update(x3085, x3075, x3076);

// cv16_B <~~ V_cv16_B
float x3086, x3087;
x3086 = 1;
x3087 = 1;
JCudaTensor x3088;
x3088 = x3006;
x85.update(x3088, x3086, x3087);

// cv15_W <~~ V_cv15_W
float x3089, x3090;
x3089 = 1;
float x3091;
float x3092;
x3091 = 1;
float x3093;
float x3094;
float x3095;
float x3096;
x3095 = 1;
x3096 = decay;
x3093 = x3095 * x3096;
float x3097;
float x3098;
x3097 = 1;
x3098 = lrn_rate;
x3094 = x3097 * x3098;
x3092 = x3093 * x3094;
x3090 = x3091 + x3092;
JCudaTensor x3099;
x3099 = x3012;
x104.update(x3099, x3089, x3090);

// cv16_W <~~ V_cv16_W
float x3100, x3101;
x3100 = 1;
float x3102;
float x3103;
x3102 = 1;
float x3104;
float x3105;
float x3106;
float x3107;
x3106 = 1;
x3107 = decay;
x3104 = x3106 * x3107;
float x3108;
float x3109;
x3108 = 1;
x3109 = lrn_rate;
x3105 = x3108 * x3109;
x3103 = x3104 * x3105;
x3101 = x3102 + x3103;
JCudaTensor x3110;
x3110 = x2999;
x84.update(x3110, x3100, x3101);

// val X4325 = X4323 * d_ReLU()(X7565)/d_X7564
JCudaTensor x3111;
JCudaTensor x3112, x3113;
x3112 = x3019;
x3113 = x90;
x3111 = x92.backward(x3112, x3113);

// Dealloc(X7565)
JCudaTensor x3114;
x3114 = x90;
x3114.free();

// val X4301 = X4299 * d_ReLU()(X7561)/d_X7560
JCudaTensor x3115;
JCudaTensor x3116, x3117;
x3116 = x3047;
x3117 = x87;
x3115 = x89.backward(x3116, x3117);

// Dealloc(X7561)
JCudaTensor x3118;
x3118 = x87;
x3118.free();

// V_cv14_W <~~ X4325 * d_Convolv(1,0)(X7557)/d_cv14_W
float x3120, x3121;
float x3122;
float x3123;
x3122 = 1;
x3123 = lrn_rate;
x3120 = x3122 * x3123;
x3121 = momentum;
JCudaTensor x3124, x3125;
x3124 = x3111;
x3125 = x53;
x79.backward_filter(x3124, x3125, x3119, x3120, x3121);

// V_cv12_B <~~ X4301 * d_Convolv(1,0)()/d_cv12_B
float x3127, x3128;
float x3129;
float x3130;
x3129 = 2;
x3130 = lrn_rate;
x3127 = x3129 * x3130;
x3128 = momentum;
JCudaTensor x3131;
x3131 = x3115;
x62.backward_bias(x3131, x3126, x3127, x3128);

// val X4303 = (X4279 + X4301 * d_Convolv(1,0)(cv12_W)/d_X7557)
JCudaTensor x3132;
JCudaTensor x3133;
x3133 = x3029;
JCudaTensor x3134, x3135;
x3134 = x3115;
x3135 = x60;
x3132 = x62.backward_data(x3134,x3135, x3133);

// V_cv14_B <~~ X4325 * d_Convolv(1,0)()/d_cv14_B
float x3137, x3138;
float x3139;
float x3140;
x3139 = 2;
x3140 = lrn_rate;
x3137 = x3139 * x3140;
x3138 = momentum;
JCudaTensor x3141;
x3141 = x3111;
x79.backward_bias(x3141, x3136, x3137, x3138);

// V_cv12_W <~~ X4301 * d_Convolv(1,0)(X7557)/d_cv12_W
float x3143, x3144;
float x3145;
float x3146;
x3145 = 1;
x3146 = lrn_rate;
x3143 = x3145 * x3146;
x3144 = momentum;
JCudaTensor x3147, x3148;
x3147 = x3115;
x3148 = x53;
x62.backward_filter(x3147, x3148, x3142, x3143, x3144);

// Dealloc(X4301)
JCudaTensor x3149;
x3149 = x3115;
x3149.free();

// cv12_B <~~ V_cv12_B
float x3150, x3151;
x3150 = 1;
x3151 = 1;
JCudaTensor x3152;
x3152 = x3126;
x61.update(x3152, x3150, x3151);

// cv14_B <~~ V_cv14_B
float x3153, x3154;
x3153 = 1;
x3154 = 1;
JCudaTensor x3155;
x3155 = x3136;
x78.update(x3155, x3153, x3154);

// cv12_W <~~ V_cv12_W
float x3156, x3157;
x3156 = 1;
float x3158;
float x3159;
x3158 = 1;
float x3160;
float x3161;
float x3162;
float x3163;
x3162 = 1;
x3163 = decay;
x3160 = x3162 * x3163;
float x3164;
float x3165;
x3164 = 1;
x3165 = lrn_rate;
x3161 = x3164 * x3165;
x3159 = x3160 * x3161;
x3157 = x3158 + x3159;
JCudaTensor x3166;
x3166 = x3142;
x60.update(x3166, x3156, x3157);

// val X4327 = (X4303 + X4325 * d_Convolv(1,0)(cv14_W)/d_X7557)
JCudaTensor x3167;
JCudaTensor x3168;
x3168 = x3132;
JCudaTensor x3169, x3170;
x3169 = x3111;
x3170 = x77;
x3167 = x79.backward_data(x3169,x3170, x3168);

// Dealloc(X4325)
JCudaTensor x3171;
x3171 = x3111;
x3171.free();

// cv14_W <~~ V_cv14_W
float x3172, x3173;
x3172 = 1;
float x3174;
float x3175;
x3174 = 1;
float x3176;
float x3177;
float x3178;
float x3179;
x3178 = 1;
x3179 = decay;
x3176 = x3178 * x3179;
float x3180;
float x3181;
x3180 = 1;
x3181 = lrn_rate;
x3177 = x3180 * x3181;
x3175 = x3176 * x3177;
x3173 = x3174 + x3175;
JCudaTensor x3182;
x3182 = x3119;
x77.update(x3182, x3172, x3173);

// val X4349 = (X4327 + X4346 * d_Pooling(3,1,1,true)(X7568,X7557)/d_X7557)
JCudaTensor x3183;
JCudaTensor x3184;
x3184 = x3167;
JCudaTensor x3185, x3186, x3187;
x3185 = x3051;
x3186 = x63;
x3187 = x53;
x3183 = x65.backward(x3185,x3186,x3187, x3184);

// Dealloc(X4346)
JCudaTensor x3188;
x3188 = x3051;
x3188.free();

// Dealloc(X7568)
JCudaTensor x3189;
x3189 = x63;
x3189.free();

// val X4351 = X4349 * d_Pooling(3,2,1,true)(X7557,X7556)/d_X7556
JCudaTensor x3190;
JCudaTensor x3191, x3192, x3193;
x3191 = x3183;
x3192 = x53;
x3193 = x50;
x3190 = x55.backward(x3191, x3192, x3193);

// Dealloc(X4349)
JCudaTensor x3194;
x3194 = x3183;
x3194.free();

// Dealloc(X7557)
JCudaTensor x3195;
x3195 = x53;
x3195.free();

// val X4353 = X4351 * d_LRN(5,1.0E-4,0.75)(X7556,X7555)/d_X7555
JCudaTensor x3196;
JCudaTensor x3197, x3198, x3199;
x3197 = x3190;
x3198 = x50;
x3199 = x47;
x3196 = x52.backward(x3197, x3198, x3199);

// Dealloc(X7556)
JCudaTensor x3200;
x3200 = x50;
x3200.free();

// val X4355 = X4353 * d_ReLU()(X7555)/d_X7554
JCudaTensor x3201;
JCudaTensor x3202, x3203;
x3202 = x3196;
x3203 = x47;
x3201 = x49.backward(x3202, x3203);

// Dealloc(X7555)
JCudaTensor x3204;
x3204 = x47;
x3204.free();

// val X4356 = X4355 * d_Convolv(1,1)(cv3_W)/d_X7553
JCudaTensor x3205;
JCudaTensor x3206, x3207;
x3206 = x3201;
x3207 = x44;
x3205 = x46.backward_data(x3206, x3207);

// V_cv3_B <~~ X4355 * d_Convolv(1,1)()/d_cv3_B
float x3209, x3210;
float x3211;
float x3212;
x3211 = 2;
x3212 = lrn_rate;
x3209 = x3211 * x3212;
x3210 = momentum;
JCudaTensor x3213;
x3213 = x3201;
x46.backward_bias(x3213, x3208, x3209, x3210);

// V_cv3_W <~~ X4355 * d_Convolv(1,1)(X7553)/d_cv3_W
float x3215, x3216;
float x3217;
float x3218;
x3217 = 1;
x3218 = lrn_rate;
x3215 = x3217 * x3218;
x3216 = momentum;
JCudaTensor x3219, x3220;
x3219 = x3201;
x3220 = x37;
x46.backward_filter(x3219, x3220, x3214, x3215, x3216);

// Dealloc(X4355)
JCudaTensor x3221;
x3221 = x3201;
x3221.free();

// cv3_B <~~ V_cv3_B
float x3222, x3223;
x3222 = 1;
x3223 = 1;
JCudaTensor x3224;
x3224 = x3208;
x45.update(x3224, x3222, x3223);

// cv3_W <~~ V_cv3_W
float x3225, x3226;
x3225 = 1;
float x3227;
float x3228;
x3227 = 1;
float x3229;
float x3230;
float x3231;
float x3232;
x3231 = 1;
x3232 = decay;
x3229 = x3231 * x3232;
float x3233;
float x3234;
x3233 = 1;
x3234 = lrn_rate;
x3230 = x3233 * x3234;
x3228 = x3229 * x3230;
x3226 = x3227 + x3228;
JCudaTensor x3235;
x3235 = x3214;
x44.update(x3235, x3225, x3226);

// val X4358 = X4356 * d_ReLU()(X7553)/d_X7552
JCudaTensor x3236;
JCudaTensor x3237, x3238;
x3237 = x3205;
x3238 = x37;
x3236 = x39.backward(x3237, x3238);

// Dealloc(X7553)
JCudaTensor x3239;
x3239 = x37;
x3239.free();

// val X4359 = X4358 * d_Convolv(1,0)(cv2_W)/d_X7551
JCudaTensor x3240;
JCudaTensor x3241, x3242;
x3241 = x3236;
x3242 = x34;
x3240 = x36.backward_data(x3241, x3242);

// V_cv2_B <~~ X4358 * d_Convolv(1,0)()/d_cv2_B
float x3244, x3245;
float x3246;
float x3247;
x3246 = 2;
x3247 = lrn_rate;
x3244 = x3246 * x3247;
x3245 = momentum;
JCudaTensor x3248;
x3248 = x3236;
x36.backward_bias(x3248, x3243, x3244, x3245);

// V_cv2_W <~~ X4358 * d_Convolv(1,0)(X7551)/d_cv2_W
float x3250, x3251;
float x3252;
float x3253;
x3252 = 1;
x3253 = lrn_rate;
x3250 = x3252 * x3253;
x3251 = momentum;
JCudaTensor x3254, x3255;
x3254 = x3236;
x3255 = x27;
x36.backward_filter(x3254, x3255, x3249, x3250, x3251);

// Dealloc(X4358)
JCudaTensor x3256;
x3256 = x3236;
x3256.free();

// cv2_B <~~ V_cv2_B
float x3257, x3258;
x3257 = 1;
x3258 = 1;
JCudaTensor x3259;
x3259 = x3243;
x35.update(x3259, x3257, x3258);

// cv2_W <~~ V_cv2_W
float x3260, x3261;
x3260 = 1;
float x3262;
float x3263;
x3262 = 1;
float x3264;
float x3265;
float x3266;
float x3267;
x3266 = 1;
x3267 = decay;
x3264 = x3266 * x3267;
float x3268;
float x3269;
x3268 = 1;
x3269 = lrn_rate;
x3265 = x3268 * x3269;
x3263 = x3264 * x3265;
x3261 = x3262 + x3263;
JCudaTensor x3270;
x3270 = x3249;
x34.update(x3270, x3260, x3261);

// val X4361 = X4359 * d_LRN(5,1.0E-4,0.75)(X7551,X7550)/d_X7550
JCudaTensor x3271;
JCudaTensor x3272, x3273, x3274;
x3272 = x3240;
x3273 = x27;
x3274 = x24;
x3271 = x29.backward(x3272, x3273, x3274);

// Dealloc(X7551)
JCudaTensor x3275;
x3275 = x27;
x3275.free();

// val X4363 = X4361 * d_Pooling(3,2,1,true)(X7550,X7549)/d_X7549
JCudaTensor x3276;
JCudaTensor x3277, x3278, x3279;
x3277 = x3271;
x3278 = x24;
x3279 = x21;
x3276 = x26.backward(x3277, x3278, x3279);

// Dealloc(X4361)
JCudaTensor x3280;
x3280 = x3271;
x3280.free();

// Dealloc(X7550)
JCudaTensor x3281;
x3281 = x24;
x3281.free();

// val X4365 = X4363 * d_ReLU()(X7549)/d_X7548
JCudaTensor x3282;
JCudaTensor x3283, x3284;
x3283 = x3276;
x3284 = x21;
x3282 = x23.backward(x3283, x3284);

// Dealloc(X7549)
JCudaTensor x3285;
x3285 = x21;
x3285.free();

// V_cv1_W <~~ X4365 * d_Convolv(2,3)(X7547)/d_cv1_W
float x3287, x3288;
float x3289;
float x3290;
x3289 = 1;
x3290 = lrn_rate;
x3287 = x3289 * x3290;
x3288 = momentum;
JCudaTensor x3291, x3292;
x3291 = x3282;
x3292 = x7;
x17.backward_filter(x3291, x3292, x3286, x3287, x3288);

// Dealloc(X7547)
JCudaTensor x3293;
x3293 = x7;
x3293.free();

// V_cv1_B <~~ X4365 * d_Convolv(2,3)()/d_cv1_B
float x3295, x3296;
float x3297;
float x3298;
x3297 = 2;
x3298 = lrn_rate;
x3295 = x3297 * x3298;
x3296 = momentum;
JCudaTensor x3299;
x3299 = x3282;
x17.backward_bias(x3299, x3294, x3295, x3296);

// Dealloc(X4365)
JCudaTensor x3300;
x3300 = x3282;
x3300.free();

// cv1_W <~~ V_cv1_W
float x3301, x3302;
x3301 = 1;
float x3303;
float x3304;
x3303 = 1;
float x3305;
float x3306;
float x3307;
float x3308;
x3307 = 1;
x3308 = decay;
x3305 = x3307 * x3308;
float x3309;
float x3310;
x3309 = 1;
x3310 = lrn_rate;
x3306 = x3309 * x3310;
x3304 = x3305 * x3306;
x3302 = x3303 + x3304;
JCudaTensor x3311;
x3311 = x3286;
x15.update(x3311, x3301, x3302);

// cv1_B <~~ V_cv1_B
float x3312, x3313;
x3312 = 1;
x3313 = 1;
JCudaTensor x3314;
x3314 = x3294;
x16.update(x3314, x3312, x3313);

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X7719 = Cuda(X)
JCudaTensor x3315;
JTensorFloat x3316;
x3316 = x3;
x3315 = x3316.asJCudaTensor();

// val X7720 = Convolv(2,3)(X7719,cv1_W,cv1_B)
JCudaTensor x3317;
JCudaTensor x3318, x3319, x3320;
x3318 = x3315;
x3319 = x15;
x3320 = x16;
x3317 = x17.forward(x3318, x3319, x3320);

// Dealloc(X7719)
JCudaTensor x3321;
x3321 = x3315;
x3321.free();

// val X7721 = ReLU()(X7720)
JCudaTensor x3322;
JCudaTensor x3323;
x3323 = x3317;
x3322 = x23.forward(x3323);

// val X7722 = Pooling(3,2,1,true)(X7721)
JCudaTensor x3324;
JCudaTensor x3325;
x3325 = x3322;
x3324 = x26.forward(x3325);

// Dealloc(X7721)
JCudaTensor x3326;
x3326 = x3322;
x3326.free();

// val X7723 = LRN(5,1.0E-4,0.75)(X7722)
JCudaTensor x3327;
JCudaTensor x3328;
x3328 = x3324;
x3327 = x29.forward(x3328);

// Dealloc(X7722)
JCudaTensor x3329;
x3329 = x3324;
x3329.free();

// val X7724 = Convolv(1,0)(X7723,cv2_W,cv2_B)
JCudaTensor x3330;
JCudaTensor x3331, x3332, x3333;
x3331 = x3327;
x3332 = x34;
x3333 = x35;
x3330 = x36.forward(x3331, x3332, x3333);

// Dealloc(X7723)
JCudaTensor x3334;
x3334 = x3327;
x3334.free();

// val X7725 = ReLU()(X7724)
JCudaTensor x3335;
JCudaTensor x3336;
x3336 = x3330;
x3335 = x39.forward(x3336);

// val X7726 = Convolv(1,1)(X7725,cv3_W,cv3_B)
JCudaTensor x3337;
JCudaTensor x3338, x3339, x3340;
x3338 = x3335;
x3339 = x44;
x3340 = x45;
x3337 = x46.forward(x3338, x3339, x3340);

// Dealloc(X7725)
JCudaTensor x3341;
x3341 = x3335;
x3341.free();

// val X7727 = ReLU()(X7726)
JCudaTensor x3342;
JCudaTensor x3343;
x3343 = x3337;
x3342 = x49.forward(x3343);

// val X7728 = LRN(5,1.0E-4,0.75)(X7727)
JCudaTensor x3344;
JCudaTensor x3345;
x3345 = x3342;
x3344 = x52.forward(x3345);

// Dealloc(X7727)
JCudaTensor x3346;
x3346 = x3342;
x3346.free();

// val X7729 = Pooling(3,2,1,true)(X7728)
JCudaTensor x3347;
JCudaTensor x3348;
x3348 = x3344;
x3347 = x55.forward(x3348);

// Dealloc(X7728)
JCudaTensor x3349;
x3349 = x3344;
x3349.free();

// val X7730 = Convolv(1,0)(X7729,cv11_W,cv11_B)
JCudaTensor x3350;
JCudaTensor x3351, x3352, x3353;
x3351 = x3347;
x3352 = x70;
x3353 = x71;
x3350 = x72.forward(x3351, x3352, x3353);

// val X7732 = Convolv(1,0)(X7729,cv12_W,cv12_B)
JCudaTensor x3354;
JCudaTensor x3355, x3356, x3357;
x3355 = x3347;
x3356 = x60;
x3357 = x61;
x3354 = x62.forward(x3355, x3356, x3357);

// val X7736 = Convolv(1,0)(X7729,cv14_W,cv14_B)
JCudaTensor x3358;
JCudaTensor x3359, x3360, x3361;
x3359 = x3347;
x3360 = x77;
x3361 = x78;
x3358 = x79.forward(x3359, x3360, x3361);

// val X7740 = Pooling(3,1,1,true)(X7729)
JCudaTensor x3362;
JCudaTensor x3363;
x3363 = x3347;
x3362 = x65.forward(x3363);

// Dealloc(X7729)
JCudaTensor x3364;
x3364 = x3347;
x3364.free();

// val X7737 = ReLU()(X7736)
JCudaTensor x3365;
JCudaTensor x3366;
x3366 = x3358;
x3365 = x92.forward(x3366);

// val X7741 = Convolv(1,0)(X7740,cv16_W,cv16_B)
JCudaTensor x3367;
JCudaTensor x3368, x3369, x3370;
x3368 = x3362;
x3369 = x84;
x3370 = x85;
x3367 = x86.forward(x3368, x3369, x3370);

// Dealloc(X7740)
JCudaTensor x3371;
x3371 = x3362;
x3371.free();

// val X7733 = ReLU()(X7732)
JCudaTensor x3372;
JCudaTensor x3373;
x3373 = x3354;
x3372 = x89.forward(x3373);

// val X7738 = Convolv(1,2)(X7737,cv15_W,cv15_B)
JCudaTensor x3374;
JCudaTensor x3375, x3376, x3377;
x3375 = x3365;
x3376 = x104;
x3377 = x105;
x3374 = x106.forward(x3375, x3376, x3377);

// Dealloc(X7737)
JCudaTensor x3378;
x3378 = x3365;
x3378.free();

// val X7734 = Convolv(1,1)(X7733,cv13_W,cv13_B)
JCudaTensor x3379;
JCudaTensor x3380, x3381, x3382;
x3380 = x3372;
x3381 = x97;
x3382 = x98;
x3379 = x99.forward(x3380, x3381, x3382);

// Dealloc(X7733)
JCudaTensor x3383;
x3383 = x3372;
x3383.free();

// val X7731 = ReLU()(X7730)
JCudaTensor x3384;
JCudaTensor x3385;
x3385 = x3350;
x3384 = x109.forward(x3385);

// val X7735 = ReLU()(X7734)
JCudaTensor x3386;
JCudaTensor x3387;
x3387 = x3379;
x3386 = x112.forward(x3387);

// val X7739 = ReLU()(X7738)
JCudaTensor x3388;
JCudaTensor x3389;
x3389 = x3374;
x3388 = x115.forward(x3389);

// val X7742 = ReLU()(X7741)
JCudaTensor x3390;
JCudaTensor x3391;
x3391 = x3367;
x3390 = x115.forward(x3391);

// val X7743 = Concat(X7731,X7735,X7739,X7742)
JCudaTensor x3392;
JCudaTensor x3393, x3394, x3395, x3396;
x3393 = x3384;
x3394 = x3386;
x3395 = x3388;
x3396 = x3390;
x3392 = x119.forward(x3393,x3394,x3395,x3396);

// Dealloc(X7742)
JCudaTensor x3397;
x3397 = x3390;
x3397.free();

// Dealloc(X7739)
JCudaTensor x3398;
x3398 = x3388;
x3398.free();

// Dealloc(X7735)
JCudaTensor x3399;
x3399 = x3386;
x3399.free();

// Dealloc(X7731)
JCudaTensor x3400;
x3400 = x3384;
x3400.free();

// val X7754 = Pooling(3,1,1,true)(X7743)
JCudaTensor x3401;
JCudaTensor x3402;
x3402 = x3392;
x3401 = x126.forward(x3402);

// val X7746 = Convolv(1,0)(X7743,cv22_W,cv22_B)
JCudaTensor x3403;
JCudaTensor x3404, x3405, x3406;
x3404 = x3392;
x3405 = x145;
x3406 = x146;
x3403 = x147.forward(x3404, x3405, x3406);

// val X7744 = Convolv(1,0)(X7743,cv21_W,cv21_B)
JCudaTensor x3407;
JCudaTensor x3408, x3409, x3410;
x3408 = x3392;
x3409 = x138;
x3410 = x139;
x3407 = x140.forward(x3408, x3409, x3410);

// val X7750 = Convolv(1,0)(X7743,cv24_W,cv24_B)
JCudaTensor x3411;
JCudaTensor x3412, x3413, x3414;
x3412 = x3392;
x3413 = x131;
x3414 = x132;
x3411 = x133.forward(x3412, x3413, x3414);

// Dealloc(X7743)
JCudaTensor x3415;
x3415 = x3392;
x3415.free();

// val X7751 = ReLU()(X7750)
JCudaTensor x3416;
JCudaTensor x3417;
x3417 = x3411;
x3416 = x92.forward(x3417);

// val X7755 = Convolv(1,0)(X7754,cv26_W,cv26_B)
JCudaTensor x3418;
JCudaTensor x3419, x3420, x3421;
x3419 = x3401;
x3420 = x152;
x3421 = x153;
x3418 = x154.forward(x3419, x3420, x3421);

// Dealloc(X7754)
JCudaTensor x3422;
x3422 = x3401;
x3422.free();

// val X7747 = ReLU()(X7746)
JCudaTensor x3423;
JCudaTensor x3424;
x3424 = x3403;
x3423 = x89.forward(x3424);

// val X7748 = Convolv(1,1)(X7747,cv23_W,cv23_B)
JCudaTensor x3425;
JCudaTensor x3426, x3427, x3428;
x3426 = x3423;
x3427 = x169;
x3428 = x170;
x3425 = x99.forward(x3426, x3427, x3428);

// Dealloc(X7747)
JCudaTensor x3429;
x3429 = x3423;
x3429.free();

// val X7752 = Convolv(1,2)(X7751,cv25_W,cv25_B)
JCudaTensor x3430;
JCudaTensor x3431, x3432, x3433;
x3431 = x3416;
x3432 = x163;
x3433 = x164;
x3430 = x106.forward(x3431, x3432, x3433);

// Dealloc(X7751)
JCudaTensor x3434;
x3434 = x3416;
x3434.free();

// val X7745 = ReLU()(X7744)
JCudaTensor x3435;
JCudaTensor x3436;
x3436 = x3407;
x3435 = x109.forward(x3436);

// val X7749 = ReLU()(X7748)
JCudaTensor x3437;
JCudaTensor x3438;
x3438 = x3425;
x3437 = x112.forward(x3438);

// val X7753 = ReLU()(X7752)
JCudaTensor x3439;
JCudaTensor x3440;
x3440 = x3430;
x3439 = x115.forward(x3440);

// val X7756 = ReLU()(X7755)
JCudaTensor x3441;
JCudaTensor x3442;
x3442 = x3418;
x3441 = x115.forward(x3442);

// val X7757 = Concat(X7745,X7749,X7753,X7756)
JCudaTensor x3443;
JCudaTensor x3444, x3445, x3446, x3447;
x3444 = x3435;
x3445 = x3437;
x3446 = x3439;
x3447 = x3441;
x3443 = x119.forward(x3444,x3445,x3446,x3447);

// Dealloc(X7756)
JCudaTensor x3448;
x3448 = x3441;
x3448.free();

// Dealloc(X7753)
JCudaTensor x3449;
x3449 = x3439;
x3449.free();

// Dealloc(X7749)
JCudaTensor x3450;
x3450 = x3437;
x3450.free();

// Dealloc(X7745)
JCudaTensor x3451;
x3451 = x3435;
x3451.free();

// val X7758 = Pooling(3,2,1,true)(X7757)
JCudaTensor x3452;
JCudaTensor x3453;
x3453 = x3443;
x3452 = x186.forward(x3453);

// Dealloc(X7757)
JCudaTensor x3454;
x3454 = x3443;
x3454.free();

// val X7759 = Convolv(1,0)(X7758,cv31_W,cv31_B)
JCudaTensor x3455;
JCudaTensor x3456, x3457, x3458;
x3456 = x3452;
x3457 = x208;
x3458 = x209;
x3455 = x210.forward(x3456, x3457, x3458);

// val X7765 = Convolv(1,0)(X7758,cv34_W,cv34_B)
JCudaTensor x3459;
JCudaTensor x3460, x3461, x3462;
x3460 = x3452;
x3461 = x194;
x3462 = x195;
x3459 = x196.forward(x3460, x3461, x3462);

// val X7761 = Convolv(1,0)(X7758,cv32_W,cv32_B)
JCudaTensor x3463;
JCudaTensor x3464, x3465, x3466;
x3464 = x3452;
x3465 = x201;
x3466 = x202;
x3463 = x203.forward(x3464, x3465, x3466);

// val X7769 = Pooling(3,1,1,true)(X7758)
JCudaTensor x3467;
JCudaTensor x3468;
x3468 = x3452;
x3467 = x189.forward(x3468);

// Dealloc(X7758)
JCudaTensor x3469;
x3469 = x3452;
x3469.free();

// val X7766 = ReLU()(X7765)
JCudaTensor x3470;
JCudaTensor x3471;
x3471 = x3459;
x3470 = x213.forward(x3471);

// val X7770 = Convolv(1,0)(X7769,cv36_W,cv36_B)
JCudaTensor x3472;
JCudaTensor x3473, x3474, x3475;
x3473 = x3467;
x3474 = x221;
x3475 = x222;
x3472 = x223.forward(x3473, x3474, x3475);

// Dealloc(X7769)
JCudaTensor x3476;
x3476 = x3467;
x3476.free();

// val X7762 = ReLU()(X7761)
JCudaTensor x3477;
JCudaTensor x3478;
x3478 = x3463;
x3477 = x216.forward(x3478);

// val X7763 = Convolv(1,1)(X7762,cv33_W,cv33_B)
JCudaTensor x3479;
JCudaTensor x3480, x3481, x3482;
x3480 = x3477;
x3481 = x235;
x3482 = x236;
x3479 = x237.forward(x3480, x3481, x3482);

// Dealloc(X7762)
JCudaTensor x3483;
x3483 = x3477;
x3483.free();

// val X7767 = Convolv(1,2)(X7766,cv35_W,cv35_B)
JCudaTensor x3484;
JCudaTensor x3485, x3486, x3487;
x3485 = x3470;
x3486 = x228;
x3487 = x229;
x3484 = x230.forward(x3485, x3486, x3487);

// Dealloc(X7766)
JCudaTensor x3488;
x3488 = x3470;
x3488.free();

// val X7760 = ReLU()(X7759)
JCudaTensor x3489;
JCudaTensor x3490;
x3490 = x3455;
x3489 = x240.forward(x3490);

// val X7764 = ReLU()(X7763)
JCudaTensor x3491;
JCudaTensor x3492;
x3492 = x3479;
x3491 = x243.forward(x3492);

// val X7768 = ReLU()(X7767)
JCudaTensor x3493;
JCudaTensor x3494;
x3494 = x3484;
x3493 = x246.forward(x3494);

// val X7771 = ReLU()(X7770)
JCudaTensor x3495;
JCudaTensor x3496;
x3496 = x3472;
x3495 = x246.forward(x3496);

// val X7772 = Concat(X7760,X7764,X7768,X7771)
JCudaTensor x3497;
JCudaTensor x3498, x3499, x3500, x3501;
x3498 = x3489;
x3499 = x3491;
x3500 = x3493;
x3501 = x3495;
x3497 = x250.forward(x3498,x3499,x3500,x3501);

// Dealloc(X7771)
JCudaTensor x3502;
x3502 = x3495;
x3502.free();

// Dealloc(X7768)
JCudaTensor x3503;
x3503 = x3493;
x3503.free();

// Dealloc(X7764)
JCudaTensor x3504;
x3504 = x3491;
x3504.free();

// Dealloc(X7760)
JCudaTensor x3505;
x3505 = x3489;
x3505.free();

// val X7783 = Pooling(3,1,1,true)(X7772)
JCudaTensor x3506;
JCudaTensor x3507;
x3507 = x3497;
x3506 = x189.forward(x3507);

// val X7779 = Convolv(1,0)(X7772,cv44_W,cv44_B)
JCudaTensor x3508;
JCudaTensor x3509, x3510, x3511;
x3509 = x3497;
x3510 = x259;
x3511 = x260;
x3508 = x196.forward(x3509, x3510, x3511);

// val X7773 = Convolv(1,0)(X7772,cv41_W,cv41_B)
JCudaTensor x3512;
JCudaTensor x3513, x3514, x3515;
x3513 = x3497;
x3514 = x265;
x3515 = x266;
x3512 = x210.forward(x3513, x3514, x3515);

// val X7775 = Convolv(1,0)(X7772,cv42_W,cv42_B)
JCudaTensor x3516;
JCudaTensor x3517, x3518, x3519;
x3517 = x3497;
x3518 = x276;
x3519 = x277;
x3516 = x203.forward(x3517, x3518, x3519);

// Dealloc(X7772)
JCudaTensor x3520;
x3520 = x3497;
x3520.free();

// val X7780 = ReLU()(X7779)
JCudaTensor x3521;
JCudaTensor x3522;
x3522 = x3508;
x3521 = x213.forward(x3522);

// val X7784 = Convolv(1,0)(X7783,cv46_W,cv46_B)
JCudaTensor x3523;
JCudaTensor x3524, x3525, x3526;
x3524 = x3506;
x3525 = x284;
x3526 = x285;
x3523 = x223.forward(x3524, x3525, x3526);

// Dealloc(X7783)
JCudaTensor x3527;
x3527 = x3506;
x3527.free();

// val X7776 = ReLU()(X7775)
JCudaTensor x3528;
JCudaTensor x3529;
x3529 = x3516;
x3528 = x216.forward(x3529);

// val X7777 = Convolv(1,1)(X7776,cv43_W,cv43_B)
JCudaTensor x3530;
JCudaTensor x3531, x3532, x3533;
x3531 = x3528;
x3532 = x308;
x3533 = x309;
x3530 = x237.forward(x3531, x3532, x3533);

// Dealloc(X7776)
JCudaTensor x3534;
x3534 = x3528;
x3534.free();

// val X7781 = Convolv(1,2)(X7780,cv45_W,cv45_B)
JCudaTensor x3535;
JCudaTensor x3536, x3537, x3538;
x3536 = x3521;
x3537 = x299;
x3538 = x300;
x3535 = x230.forward(x3536, x3537, x3538);

// Dealloc(X7780)
JCudaTensor x3539;
x3539 = x3521;
x3539.free();

// val X7774 = ReLU()(X7773)
JCudaTensor x3540;
JCudaTensor x3541;
x3541 = x3512;
x3540 = x240.forward(x3541);

// val X7778 = ReLU()(X7777)
JCudaTensor x3542;
JCudaTensor x3543;
x3543 = x3530;
x3542 = x243.forward(x3543);

// val X7782 = ReLU()(X7781)
JCudaTensor x3544;
JCudaTensor x3545;
x3545 = x3535;
x3544 = x246.forward(x3545);

// val X7785 = ReLU()(X7784)
JCudaTensor x3546;
JCudaTensor x3547;
x3547 = x3523;
x3546 = x246.forward(x3547);

// val X7786 = Concat(X7774,X7778,X7782,X7785)
JCudaTensor x3548;
JCudaTensor x3549, x3550, x3551, x3552;
x3549 = x3540;
x3550 = x3542;
x3551 = x3544;
x3552 = x3546;
x3548 = x250.forward(x3549,x3550,x3551,x3552);

// Dealloc(X7785)
JCudaTensor x3553;
x3553 = x3546;
x3553.free();

// Dealloc(X7782)
JCudaTensor x3554;
x3554 = x3544;
x3554.free();

// Dealloc(X7778)
JCudaTensor x3555;
x3555 = x3542;
x3555.free();

// Dealloc(X7774)
JCudaTensor x3556;
x3556 = x3540;
x3556.free();

// val X7793 = Convolv(1,0)(X7786,cv54_W,cv54_B)
JCudaTensor x3557;
JCudaTensor x3558, x3559, x3560;
x3558 = x3548;
x3559 = x338;
x3560 = x339;
x3557 = x196.forward(x3558, x3559, x3560);

// val X7787 = Convolv(1,0)(X7786,cv51_W,cv51_B)
JCudaTensor x3561;
JCudaTensor x3562, x3563, x3564;
x3562 = x3548;
x3563 = x355;
x3564 = x356;
x3561 = x210.forward(x3562, x3563, x3564);

// val X7789 = Convolv(1,0)(X7786,cv52_W,cv52_B)
JCudaTensor x3565;
JCudaTensor x3566, x3567, x3568;
x3566 = x3548;
x3567 = x344;
x3568 = x345;
x3565 = x203.forward(x3566, x3567, x3568);

// val X7797 = Pooling(3,1,1,true)(X7786)
JCudaTensor x3569;
JCudaTensor x3570;
x3570 = x3548;
x3569 = x189.forward(x3570);

// Dealloc(X7786)
JCudaTensor x3571;
x3571 = x3548;
x3571.free();

// val X7794 = ReLU()(X7793)
JCudaTensor x3572;
JCudaTensor x3573;
x3573 = x3557;
x3572 = x213.forward(x3573);

// val X7798 = Convolv(1,0)(X7797,cv56_W,cv56_B)
JCudaTensor x3574;
JCudaTensor x3575, x3576, x3577;
x3575 = x3569;
x3576 = x364;
x3577 = x365;
x3574 = x223.forward(x3575, x3576, x3577);

// Dealloc(X7797)
JCudaTensor x3578;
x3578 = x3569;
x3578.free();

// val X7790 = ReLU()(X7789)
JCudaTensor x3579;
JCudaTensor x3580;
x3580 = x3565;
x3579 = x216.forward(x3580);

// val X7795 = Convolv(1,2)(X7794,cv55_W,cv55_B)
JCudaTensor x3581;
JCudaTensor x3582, x3583, x3584;
x3582 = x3572;
x3583 = x380;
x3584 = x381;
x3581 = x230.forward(x3582, x3583, x3584);

// Dealloc(X7794)
JCudaTensor x3585;
x3585 = x3572;
x3585.free();

// val X7791 = Convolv(1,1)(X7790,cv53_W,cv53_B)
JCudaTensor x3586;
JCudaTensor x3587, x3588, x3589;
x3587 = x3579;
x3588 = x374;
x3589 = x375;
x3586 = x237.forward(x3587, x3588, x3589);

// Dealloc(X7790)
JCudaTensor x3590;
x3590 = x3579;
x3590.free();

// val X7788 = ReLU()(X7787)
JCudaTensor x3591;
JCudaTensor x3592;
x3592 = x3561;
x3591 = x240.forward(x3592);

// val X7792 = ReLU()(X7791)
JCudaTensor x3593;
JCudaTensor x3594;
x3594 = x3586;
x3593 = x243.forward(x3594);

// val X7796 = ReLU()(X7795)
JCudaTensor x3595;
JCudaTensor x3596;
x3596 = x3581;
x3595 = x246.forward(x3596);

// val X7799 = ReLU()(X7798)
JCudaTensor x3597;
JCudaTensor x3598;
x3598 = x3574;
x3597 = x246.forward(x3598);

// val X7800 = Concat(X7788,X7792,X7796,X7799)
JCudaTensor x3599;
JCudaTensor x3600, x3601, x3602, x3603;
x3600 = x3591;
x3601 = x3593;
x3602 = x3595;
x3603 = x3597;
x3599 = x250.forward(x3600,x3601,x3602,x3603);

// Dealloc(X7799)
JCudaTensor x3604;
x3604 = x3597;
x3604.free();

// Dealloc(X7796)
JCudaTensor x3605;
x3605 = x3595;
x3605.free();

// Dealloc(X7792)
JCudaTensor x3606;
x3606 = x3593;
x3606.free();

// Dealloc(X7788)
JCudaTensor x3607;
x3607 = x3591;
x3607.free();

// val X7807 = Convolv(1,0)(X7800,cv64_W,cv64_B)
JCudaTensor x3608;
JCudaTensor x3609, x3610, x3611;
x3609 = x3599;
x3610 = x417;
x3611 = x418;
x3608 = x196.forward(x3609, x3610, x3611);

// val X7801 = Convolv(1,0)(X7800,cv61_W,cv61_B)
JCudaTensor x3612;
JCudaTensor x3613, x3614, x3615;
x3613 = x3599;
x3614 = x423;
x3615 = x424;
x3612 = x210.forward(x3613, x3614, x3615);

// val X7811 = Pooling(3,1,1,true)(X7800)
JCudaTensor x3616;
JCudaTensor x3617;
x3617 = x3599;
x3616 = x189.forward(x3617);

// val X7803 = Convolv(1,0)(X7800,cv62_W,cv62_B)
JCudaTensor x3618;
JCudaTensor x3619, x3620, x3621;
x3619 = x3599;
x3620 = x434;
x3621 = x435;
x3618 = x203.forward(x3619, x3620, x3621);

// Dealloc(X7800)
JCudaTensor x3622;
x3622 = x3599;
x3622.free();

// val X7808 = ReLU()(X7807)
JCudaTensor x3623;
JCudaTensor x3624;
x3624 = x3608;
x3623 = x213.forward(x3624);

// val X7804 = ReLU()(X7803)
JCudaTensor x3625;
JCudaTensor x3626;
x3626 = x3618;
x3625 = x216.forward(x3626);

// val X7812 = Convolv(1,0)(X7811,cv66_W,cv66_B)
JCudaTensor x3627;
JCudaTensor x3628, x3629, x3630;
x3628 = x3616;
x3629 = x450;
x3630 = x451;
x3627 = x223.forward(x3628, x3629, x3630);

// Dealloc(X7811)
JCudaTensor x3631;
x3631 = x3616;
x3631.free();

// val X7809 = Convolv(1,2)(X7808,cv65_W,cv65_B)
JCudaTensor x3632;
JCudaTensor x3633, x3634, x3635;
x3633 = x3623;
x3634 = x460;
x3635 = x461;
x3632 = x230.forward(x3633, x3634, x3635);

// Dealloc(X7808)
JCudaTensor x3636;
x3636 = x3623;
x3636.free();

// val X7805 = Convolv(1,1)(X7804,cv63_W,cv63_B)
JCudaTensor x3637;
JCudaTensor x3638, x3639, x3640;
x3638 = x3625;
x3639 = x469;
x3640 = x470;
x3637 = x237.forward(x3638, x3639, x3640);

// Dealloc(X7804)
JCudaTensor x3641;
x3641 = x3625;
x3641.free();

// val X7802 = ReLU()(X7801)
JCudaTensor x3642;
JCudaTensor x3643;
x3643 = x3612;
x3642 = x240.forward(x3643);

// val X7806 = ReLU()(X7805)
JCudaTensor x3644;
JCudaTensor x3645;
x3645 = x3637;
x3644 = x243.forward(x3645);

// val X7810 = ReLU()(X7809)
JCudaTensor x3646;
JCudaTensor x3647;
x3647 = x3632;
x3646 = x246.forward(x3647);

// val X7813 = ReLU()(X7812)
JCudaTensor x3648;
JCudaTensor x3649;
x3649 = x3627;
x3648 = x246.forward(x3649);

// val X7814 = Concat(X7802,X7806,X7810,X7813)
JCudaTensor x3650;
JCudaTensor x3651, x3652, x3653, x3654;
x3651 = x3642;
x3652 = x3644;
x3653 = x3646;
x3654 = x3648;
x3650 = x250.forward(x3651,x3652,x3653,x3654);

// Dealloc(X7813)
JCudaTensor x3655;
x3655 = x3648;
x3655.free();

// Dealloc(X7810)
JCudaTensor x3656;
x3656 = x3646;
x3656.free();

// Dealloc(X7806)
JCudaTensor x3657;
x3657 = x3644;
x3657.free();

// Dealloc(X7802)
JCudaTensor x3658;
x3658 = x3642;
x3658.free();

// val X7815 = Convolv(1,0)(X7814,cv71_W,cv71_B)
JCudaTensor x3659;
JCudaTensor x3660, x3661, x3662;
x3660 = x3650;
x3661 = x544;
x3662 = x545;
x3659 = x210.forward(x3660, x3661, x3662);

// val X7825 = Pooling(3,1,1,true)(X7814)
JCudaTensor x3663;
JCudaTensor x3664;
x3664 = x3650;
x3663 = x189.forward(x3664);

// val X7821 = Convolv(1,0)(X7814,cv74_W,cv74_B)
JCudaTensor x3665;
JCudaTensor x3666, x3667, x3668;
x3666 = x3650;
x3667 = x562;
x3668 = x563;
x3665 = x196.forward(x3666, x3667, x3668);

// val X7817 = Convolv(1,0)(X7814,cv72_W,cv72_B)
JCudaTensor x3669;
JCudaTensor x3670, x3671, x3672;
x3670 = x3650;
x3671 = x556;
x3672 = x557;
x3669 = x203.forward(x3670, x3671, x3672);

// Dealloc(X7814)
JCudaTensor x3673;
x3673 = x3650;
x3673.free();

// val X7822 = ReLU()(X7821)
JCudaTensor x3674;
JCudaTensor x3675;
x3675 = x3665;
x3674 = x213.forward(x3675);

// val X7826 = Convolv(1,0)(X7825,cv76_W,cv76_B)
JCudaTensor x3676;
JCudaTensor x3677, x3678, x3679;
x3677 = x3663;
x3678 = x592;
x3679 = x593;
x3676 = x223.forward(x3677, x3678, x3679);

// Dealloc(X7825)
JCudaTensor x3680;
x3680 = x3663;
x3680.free();

// val X7818 = ReLU()(X7817)
JCudaTensor x3681;
JCudaTensor x3682;
x3682 = x3669;
x3681 = x216.forward(x3682);

// val X7819 = Convolv(1,1)(X7818,cv73_W,cv73_B)
JCudaTensor x3683;
JCudaTensor x3684, x3685, x3686;
x3684 = x3681;
x3685 = x627;
x3686 = x628;
x3683 = x237.forward(x3684, x3685, x3686);

// Dealloc(X7818)
JCudaTensor x3687;
x3687 = x3681;
x3687.free();

// val X7823 = Convolv(1,2)(X7822,cv75_W,cv75_B)
JCudaTensor x3688;
JCudaTensor x3689, x3690, x3691;
x3689 = x3674;
x3690 = x621;
x3691 = x622;
x3688 = x230.forward(x3689, x3690, x3691);

// Dealloc(X7822)
JCudaTensor x3692;
x3692 = x3674;
x3692.free();

// val X7816 = ReLU()(X7815)
JCudaTensor x3693;
JCudaTensor x3694;
x3694 = x3659;
x3693 = x240.forward(x3694);

// val X7820 = ReLU()(X7819)
JCudaTensor x3695;
JCudaTensor x3696;
x3696 = x3683;
x3695 = x243.forward(x3696);

// val X7824 = ReLU()(X7823)
JCudaTensor x3697;
JCudaTensor x3698;
x3698 = x3688;
x3697 = x246.forward(x3698);

// val X7827 = ReLU()(X7826)
JCudaTensor x3699;
JCudaTensor x3700;
x3700 = x3676;
x3699 = x246.forward(x3700);

// val X7828 = Concat(X7816,X7820,X7824,X7827)
JCudaTensor x3701;
JCudaTensor x3702, x3703, x3704, x3705;
x3702 = x3693;
x3703 = x3695;
x3704 = x3697;
x3705 = x3699;
x3701 = x250.forward(x3702,x3703,x3704,x3705);

// Dealloc(X7827)
JCudaTensor x3706;
x3706 = x3699;
x3706.free();

// Dealloc(X7824)
JCudaTensor x3707;
x3707 = x3697;
x3707.free();

// Dealloc(X7820)
JCudaTensor x3708;
x3708 = x3695;
x3708.free();

// Dealloc(X7816)
JCudaTensor x3709;
x3709 = x3693;
x3709.free();

// val X7829 = Pooling(3,2,1,true)(X7828)
JCudaTensor x3710;
JCudaTensor x3711;
x3711 = x3701;
x3710 = x714.forward(x3711);

// Dealloc(X7828)
JCudaTensor x3712;
x3712 = x3701;
x3712.free();

// val X7840 = Pooling(3,1,1,true)(X7829)
JCudaTensor x3713;
JCudaTensor x3714;
x3714 = x3710;
x3713 = x719.forward(x3714);

// val X7832 = Convolv(1,0)(X7829,cv82_W,cv82_B)
JCudaTensor x3715;
JCudaTensor x3716, x3717, x3718;
x3716 = x3710;
x3717 = x738;
x3718 = x739;
x3715 = x740.forward(x3716, x3717, x3718);

// val X7830 = Convolv(1,0)(X7829,cv81_W,cv81_B)
JCudaTensor x3719;
JCudaTensor x3720, x3721, x3722;
x3720 = x3710;
x3721 = x731;
x3722 = x732;
x3719 = x733.forward(x3720, x3721, x3722);

// val X7836 = Convolv(1,0)(X7829,cv84_W,cv84_B)
JCudaTensor x3723;
JCudaTensor x3724, x3725, x3726;
x3724 = x3710;
x3725 = x724;
x3726 = x725;
x3723 = x726.forward(x3724, x3725, x3726);

// Dealloc(X7829)
JCudaTensor x3727;
x3727 = x3710;
x3727.free();

// val X7833 = ReLU()(X7832)
JCudaTensor x3728;
JCudaTensor x3729;
x3729 = x3715;
x3728 = x743.forward(x3729);

// val X7837 = ReLU()(X7836)
JCudaTensor x3730;
JCudaTensor x3731;
x3731 = x3723;
x3730 = x759.forward(x3731);

// val X7841 = Convolv(1,0)(X7840,cv86_W,cv86_B)
JCudaTensor x3732;
JCudaTensor x3733, x3734, x3735;
x3733 = x3713;
x3734 = x754;
x3735 = x755;
x3732 = x756.forward(x3733, x3734, x3735);

// Dealloc(X7840)
JCudaTensor x3736;
x3736 = x3713;
x3736.free();

// val X7838 = Convolv(1,2)(X7837,cv85_W,cv85_B)
JCudaTensor x3737;
JCudaTensor x3738, x3739, x3740;
x3738 = x3730;
x3739 = x771;
x3740 = x772;
x3737 = x773.forward(x3738, x3739, x3740);

// Dealloc(X7837)
JCudaTensor x3741;
x3741 = x3730;
x3741.free();

// val X7834 = Convolv(1,1)(X7833,cv83_W,cv83_B)
JCudaTensor x3742;
JCudaTensor x3743, x3744, x3745;
x3743 = x3728;
x3744 = x764;
x3745 = x765;
x3742 = x766.forward(x3743, x3744, x3745);

// Dealloc(X7833)
JCudaTensor x3746;
x3746 = x3728;
x3746.free();

// val X7831 = ReLU()(X7830)
JCudaTensor x3747;
JCudaTensor x3748;
x3748 = x3719;
x3747 = x788.forward(x3748);

// val X7835 = ReLU()(X7834)
JCudaTensor x3749;
JCudaTensor x3750;
x3750 = x3742;
x3749 = x780.forward(x3750);

// val X7839 = ReLU()(X7838)
JCudaTensor x3751;
JCudaTensor x3752;
x3752 = x3737;
x3751 = x783.forward(x3752);

// val X7842 = ReLU()(X7841)
JCudaTensor x3753;
JCudaTensor x3754;
x3754 = x3732;
x3753 = x783.forward(x3754);

// val X7843 = Concat(X7831,X7835,X7839,X7842)
JCudaTensor x3755;
JCudaTensor x3756, x3757, x3758, x3759;
x3756 = x3747;
x3757 = x3749;
x3758 = x3751;
x3759 = x3753;
x3755 = x793.forward(x3756,x3757,x3758,x3759);

// Dealloc(X7842)
JCudaTensor x3760;
x3760 = x3753;
x3760.free();

// Dealloc(X7839)
JCudaTensor x3761;
x3761 = x3751;
x3761.free();

// Dealloc(X7835)
JCudaTensor x3762;
x3762 = x3749;
x3762.free();

// Dealloc(X7831)
JCudaTensor x3763;
x3763 = x3747;
x3763.free();

// val X7846 = Convolv(1,0)(X7843,cv92_W,cv92_B)
JCudaTensor x3764;
JCudaTensor x3765, x3766, x3767;
x3765 = x3755;
x3766 = x827;
x3767 = x828;
x3764 = x740.forward(x3765, x3766, x3767);

// val X7854 = Pooling(3,1,1,true)(X7843)
JCudaTensor x3768;
JCudaTensor x3769;
x3769 = x3755;
x3768 = x719.forward(x3769);

// val X7850 = Convolv(1,0)(X7843,cv94_W,cv94_B)
JCudaTensor x3770;
JCudaTensor x3771, x3772, x3773;
x3771 = x3755;
x3772 = x817;
x3773 = x818;
x3770 = x726.forward(x3771, x3772, x3773);

// val X7844 = Convolv(1,0)(X7843,cv91_W,cv91_B)
JCudaTensor x3774;
JCudaTensor x3775, x3776, x3777;
x3775 = x3755;
x3776 = x807;
x3777 = x808;
x3774 = x733.forward(x3775, x3776, x3777);

// Dealloc(X7843)
JCudaTensor x3778;
x3778 = x3755;
x3778.free();

// val X7855 = Convolv(1,0)(X7854,cv96_W,cv96_B)
JCudaTensor x3779;
JCudaTensor x3780, x3781, x3782;
x3780 = x3768;
x3781 = x844;
x3782 = x845;
x3779 = x756.forward(x3780, x3781, x3782);

// Dealloc(X7854)
JCudaTensor x3783;
x3783 = x3768;
x3783.free();

// val X7847 = ReLU()(X7846)
JCudaTensor x3784;
JCudaTensor x3785;
x3785 = x3764;
x3784 = x743.forward(x3785);

// val X7851 = ReLU()(X7850)
JCudaTensor x3786;
JCudaTensor x3787;
x3787 = x3770;
x3786 = x759.forward(x3787);

// val X7852 = Convolv(1,2)(X7851,cv95_W,cv95_B)
JCudaTensor x3788;
JCudaTensor x3789, x3790, x3791;
x3789 = x3786;
x3790 = x870;
x3791 = x871;
x3788 = x773.forward(x3789, x3790, x3791);

// Dealloc(X7851)
JCudaTensor x3792;
x3792 = x3786;
x3792.free();

// val X7848 = Convolv(1,1)(X7847,cv93_W,cv93_B)
JCudaTensor x3793;
JCudaTensor x3794, x3795, x3796;
x3794 = x3784;
x3795 = x855;
x3796 = x856;
x3793 = x766.forward(x3794, x3795, x3796);

// Dealloc(X7847)
JCudaTensor x3797;
x3797 = x3784;
x3797.free();

// val X7845 = ReLU()(X7844)
JCudaTensor x3798;
JCudaTensor x3799;
x3799 = x3774;
x3798 = x788.forward(x3799);

// val X7849 = ReLU()(X7848)
JCudaTensor x3800;
JCudaTensor x3801;
x3801 = x3793;
x3800 = x780.forward(x3801);

// val X7853 = ReLU()(X7852)
JCudaTensor x3802;
JCudaTensor x3803;
x3803 = x3788;
x3802 = x783.forward(x3803);

// val X7856 = ReLU()(X7855)
JCudaTensor x3804;
JCudaTensor x3805;
x3805 = x3779;
x3804 = x783.forward(x3805);

// val X7857 = Concat(X7845,X7849,X7853,X7856)
JCudaTensor x3806;
JCudaTensor x3807, x3808, x3809, x3810;
x3807 = x3798;
x3808 = x3800;
x3809 = x3802;
x3810 = x3804;
x3806 = x793.forward(x3807,x3808,x3809,x3810);

// Dealloc(X7856)
JCudaTensor x3811;
x3811 = x3804;
x3811.free();

// Dealloc(X7853)
JCudaTensor x3812;
x3812 = x3802;
x3812.free();

// Dealloc(X7849)
JCudaTensor x3813;
x3813 = x3800;
x3813.free();

// Dealloc(X7845)
JCudaTensor x3814;
x3814 = x3798;
x3814.free();

// val X7858 = Pooling(7,1,0,false)(X7857)
JCudaTensor x3815;
JCudaTensor x3816;
x3816 = x3806;
x3815 = x938.forward(x3816);

// Dealloc(X7857)
JCudaTensor x3817;
x3817 = x3806;
x3817.free();

// val X7859 = Dropout(0.4)(X7858)
JCudaTensor x3818;
JCudaTensor x3819;
x3819 = x3815;
x3818 = x975.forward(x3819);

// Dealloc(X7858)
JCudaTensor x3820;
x3820 = x3815;
x3820.free();

// val X7860 = (X7859[1><3])(i12 | @) * (fc_W)(i13 | @)
JCudaTensor x3821;
JCudaMatrix x3822;
JCudaMatrix x3823;
JCudaTensor x3824;
JCudaTensor x3825;
x3825 = x3818;
x3824 = x3825.flatten(1, new int[]{256, 1, 1});
x3822 = x3824.asMatrix(1, true);
JCudaTensor x3826;
x3826 = x1010;
x3823 = x3826.asMatrix(1, true);
x3821 = x3822.times(x3823);

// Dealloc(X7859)
JCudaTensor x3827;
x3827 = x3818;
x3827.free();

// val X7862 = (X7860 + (i12) => fc_B)
JCudaTensor x3828;
JCudaTensor x3829, x3830;
x3829 = x3821;
x3830 = x1035;
x3828 = x3830.copy(128, x3829);

// Precision(Accuracy(X7862, Y, 1))
float x3832;
JCudaTensor x3833;
JTensorFloat x3834;
x3833 = x3828;
x3834 = x4;
x3832 = x3833.accuracy(x3834, 1);
System.out.println(x5 + " test precision "  + x3832);
x3831 += x3832;

// Dealloc(X7862)
JCudaTensor x3835;
x3835 = x3828;
x3835.free();

}
System.out.println();
System.out.println("average precision: " + x3831/test_itr);
System.out.println(); 
}

}