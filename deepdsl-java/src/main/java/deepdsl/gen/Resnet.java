package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Resnet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay
	static float decay = 5.0E-4f;
	// lrn_rate
	static float lrn_rate = -0.01f;
	// momentum
	static float momentum = 0.9f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/resnet";
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

	// (BatchNorm(1_bn),List(List(64, 64, 112, 112), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x27 = new JCudnnBatchNorm(network_dir + "/1_bn", new int[]{64,64,112,112});
	// (BatchNorm(2a1_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x61 = new JCudnnBatchNorm(network_dir + "/2a1_bn", new int[]{64,256,55,55});
	// (BatchNorm(2a2_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x54 = new JCudnnBatchNorm(network_dir + "/2a2_a_bn", new int[]{64,64,55,55});
	// (BatchNorm(2a2_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x78 = new JCudnnBatchNorm(network_dir + "/2a2_b_bn", new int[]{64,64,55,55});
	// (BatchNorm(2a2_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x93 = new JCudnnBatchNorm(network_dir + "/2a2_c_bn", new int[]{64,256,55,55});
	// (BatchNorm(2b_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x117 = new JCudnnBatchNorm(network_dir + "/2b_a_bn", new int[]{64,64,55,55});
	// (BatchNorm(2b_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x132 = new JCudnnBatchNorm(network_dir + "/2b_b_bn", new int[]{64,64,55,55});
	// (BatchNorm(2b_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x147 = new JCudnnBatchNorm(network_dir + "/2b_c_bn", new int[]{64,256,55,55});
	// (BatchNorm(2c_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x167 = new JCudnnBatchNorm(network_dir + "/2c_a_bn", new int[]{64,64,55,55});
	// (BatchNorm(2c_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
	static JCudnnBatchNorm x182 = new JCudnnBatchNorm(network_dir + "/2c_b_bn", new int[]{64,64,55,55});
	// (BatchNorm(2c_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x197 = new JCudnnBatchNorm(network_dir + "/2c_c_bn", new int[]{64,256,55,55});
	// (BatchNorm(3a1_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x225 = new JCudnnBatchNorm(network_dir + "/3a1_bn", new int[]{64,512,28,28});
	// (BatchNorm(3a2_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x232 = new JCudnnBatchNorm(network_dir + "/3a2_a_bn", new int[]{64,128,28,28});
	// (BatchNorm(3a2_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x249 = new JCudnnBatchNorm(network_dir + "/3a2_b_bn", new int[]{64,128,28,28});
	// (BatchNorm(3a2_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x265 = new JCudnnBatchNorm(network_dir + "/3a2_c_bn", new int[]{64,512,28,28});
	// (BatchNorm(3b_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x289 = new JCudnnBatchNorm(network_dir + "/3b_a_bn", new int[]{64,128,28,28});
	// (BatchNorm(3b_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x304 = new JCudnnBatchNorm(network_dir + "/3b_b_bn", new int[]{64,128,28,28});
	// (BatchNorm(3b_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x319 = new JCudnnBatchNorm(network_dir + "/3b_c_bn", new int[]{64,512,28,28});
	// (BatchNorm(3c_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x339 = new JCudnnBatchNorm(network_dir + "/3c_a_bn", new int[]{64,128,28,28});
	// (BatchNorm(3c_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x354 = new JCudnnBatchNorm(network_dir + "/3c_b_bn", new int[]{64,128,28,28});
	// (BatchNorm(3c_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x369 = new JCudnnBatchNorm(network_dir + "/3c_c_bn", new int[]{64,512,28,28});
	// (BatchNorm(3d_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x389 = new JCudnnBatchNorm(network_dir + "/3d_a_bn", new int[]{64,128,28,28});
	// (BatchNorm(3d_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
	static JCudnnBatchNorm x404 = new JCudnnBatchNorm(network_dir + "/3d_b_bn", new int[]{64,128,28,28});
	// (BatchNorm(3d_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x419 = new JCudnnBatchNorm(network_dir + "/3d_c_bn", new int[]{64,512,28,28});
	// (BatchNorm(4a1_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x447 = new JCudnnBatchNorm(network_dir + "/4a1_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4a2_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x454 = new JCudnnBatchNorm(network_dir + "/4a2_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4a2_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x471 = new JCudnnBatchNorm(network_dir + "/4a2_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4a2_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x487 = new JCudnnBatchNorm(network_dir + "/4a2_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4b_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x511 = new JCudnnBatchNorm(network_dir + "/4b_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4b_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x526 = new JCudnnBatchNorm(network_dir + "/4b_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4b_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x541 = new JCudnnBatchNorm(network_dir + "/4b_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4c_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x561 = new JCudnnBatchNorm(network_dir + "/4c_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4c_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x576 = new JCudnnBatchNorm(network_dir + "/4c_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4c_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x591 = new JCudnnBatchNorm(network_dir + "/4c_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4d_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x611 = new JCudnnBatchNorm(network_dir + "/4d_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4d_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x626 = new JCudnnBatchNorm(network_dir + "/4d_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4d_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x641 = new JCudnnBatchNorm(network_dir + "/4d_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4e_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x661 = new JCudnnBatchNorm(network_dir + "/4e_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4e_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x676 = new JCudnnBatchNorm(network_dir + "/4e_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4e_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x691 = new JCudnnBatchNorm(network_dir + "/4e_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(4f_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x711 = new JCudnnBatchNorm(network_dir + "/4f_a_bn", new int[]{64,256,14,14});
	// (BatchNorm(4f_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
	static JCudnnBatchNorm x726 = new JCudnnBatchNorm(network_dir + "/4f_b_bn", new int[]{64,256,14,14});
	// (BatchNorm(4f_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
	static JCudnnBatchNorm x741 = new JCudnnBatchNorm(network_dir + "/4f_c_bn", new int[]{64,1024,14,14});
	// (BatchNorm(5a1_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
	static JCudnnBatchNorm x769 = new JCudnnBatchNorm(network_dir + "/5a1_bn", new int[]{64,2048,7,7});
	// (BatchNorm(5a2_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x776 = new JCudnnBatchNorm(network_dir + "/5a2_a_bn", new int[]{64,512,7,7});
	// (BatchNorm(5a2_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x793 = new JCudnnBatchNorm(network_dir + "/5a2_b_bn", new int[]{64,512,7,7});
	// (BatchNorm(5a2_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
	static JCudnnBatchNorm x809 = new JCudnnBatchNorm(network_dir + "/5a2_c_bn", new int[]{64,2048,7,7});
	// (BatchNorm(5b_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x833 = new JCudnnBatchNorm(network_dir + "/5b_a_bn", new int[]{64,512,7,7});
	// (BatchNorm(5b_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x848 = new JCudnnBatchNorm(network_dir + "/5b_b_bn", new int[]{64,512,7,7});
	// (BatchNorm(5b_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
	static JCudnnBatchNorm x863 = new JCudnnBatchNorm(network_dir + "/5b_c_bn", new int[]{64,2048,7,7});
	// (BatchNorm(5c_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x883 = new JCudnnBatchNorm(network_dir + "/5c_a_bn", new int[]{64,512,7,7});
	// (BatchNorm(5c_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
	static JCudnnBatchNorm x898 = new JCudnnBatchNorm(network_dir + "/5c_b_bn", new int[]{64,512,7,7});
	// (BatchNorm(5c_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
	static JCudnnBatchNorm x913 = new JCudnnBatchNorm(network_dir + "/5c_c_bn", new int[]{64,2048,7,7});
	// (Convolv(1,0),List(List(64, 1024, 14, 14), List(256, 1024, 1, 1), List(256)))
	static JCudnnConvolution x504 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
	// (Convolv(1,0),List(List(64, 128, 28, 28), List(512, 128, 1, 1), List(512)))
	static JCudnnConvolution x258 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
	// (Convolv(1,0),List(List(64, 2048, 7, 7), List(512, 2048, 1, 1), List(512)))
	static JCudnnConvolution x826 = new JCudnnConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
	// (Convolv(1,0),List(List(64, 256, 14, 14), List(1024, 256, 1, 1), List(1024)))
	static JCudnnConvolution x480 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
	// (Convolv(1,0),List(List(64, 256, 55, 55), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x110 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(64, 512, 28, 28), List(128, 512, 1, 1), List(128)))
	static JCudnnConvolution x282 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
	// (Convolv(1,0),List(List(64, 512, 7, 7), List(2048, 512, 1, 1), List(2048)))
	static JCudnnConvolution x802 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
	// (Convolv(1,0),List(List(64, 64, 55, 55), List(256, 64, 1, 1), List(256)))
	static JCudnnConvolution x47 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
	// (Convolv(1,0),List(List(64, 64, 55, 55), List(64, 64, 1, 1), List(64)))
	static JCudnnConvolution x40 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,1),List(List(64, 128, 28, 28), List(128, 128, 3, 3), List(128)))
	static JCudnnConvolution x242 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(64, 256, 14, 14), List(256, 256, 3, 3), List(256)))
	static JCudnnConvolution x464 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
	// (Convolv(1,1),List(List(64, 512, 7, 7), List(512, 512, 3, 3), List(512)))
	static JCudnnConvolution x786 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
	// (Convolv(1,1),List(List(64, 64, 55, 55), List(64, 64, 3, 3), List(64)))
	static JCudnnConvolution x71 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
	// (Convolv(2,0),List(List(64, 1024, 14, 14), List(2048, 1024, 1, 1), List(2048)))
	static JCudnnConvolution x762 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{2048,1024,1,1},new int[]{2048}, 2, 0);
	// (Convolv(2,0),List(List(64, 1024, 14, 14), List(512, 1024, 1, 1), List(512)))
	static JCudnnConvolution x755 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{512,1024,1,1},new int[]{512}, 2, 0);
	// (Convolv(2,0),List(List(64, 256, 55, 55), List(128, 256, 1, 1), List(128)))
	static JCudnnConvolution x211 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{128,256,1,1},new int[]{128}, 2, 0);
	// (Convolv(2,0),List(List(64, 256, 55, 55), List(512, 256, 1, 1), List(512)))
	static JCudnnConvolution x218 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{512,256,1,1},new int[]{512}, 2, 0);
	// (Convolv(2,0),List(List(64, 512, 28, 28), List(1024, 512, 1, 1), List(1024)))
	static JCudnnConvolution x433 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{1024,512,1,1},new int[]{1024}, 2, 0);
	// (Convolv(2,0),List(List(64, 512, 28, 28), List(256, 512, 1, 1), List(256)))
	static JCudnnConvolution x440 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{256,512,1,1},new int[]{256}, 2, 0);
	// (Convolv(2,3),List(List(64, 3, 224, 224), List(64, 3, 7, 7), List(64)))
	static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
	// (Lmdb(1000000,10000,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, 1000, true);
	// (Lmdb(1000000,10000,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{64, 3, 224, 224}, 1000, false);
	// (LogSoftmax(),List(List(64, 1000)))
	static JCudnnSoftmax x937 = new JCudnnSoftmax(new int[]{64,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(3,2,0,true),List(List(64, 64, 112, 112)))
	static JCudnnPooling x33 = new JCudnnPooling(new int[]{64,64,112,112}, 3, 2, 0, PoolingType.MAX);
	// (Pooling(7,1,0,false),List(List(64, 2048, 7, 7)))
	static JCudnnPooling x923 = new JCudnnPooling(new int[]{64,2048,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (ReLU(),List(List(64, 1024, 14, 14)))
	static JCudnnActivation x490 = new JCudnnActivation(new int[]{64,1024,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 128, 28, 28)))
	static JCudnnActivation x235 = new JCudnnActivation(new int[]{64,128,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 2048, 7, 7)))
	static JCudnnActivation x812 = new JCudnnActivation(new int[]{64,2048,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 256, 14, 14)))
	static JCudnnActivation x457 = new JCudnnActivation(new int[]{64,256,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 256, 55, 55)))
	static JCudnnActivation x96 = new JCudnnActivation(new int[]{64,256,55,55}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 512, 28, 28)))
	static JCudnnActivation x268 = new JCudnnActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 512, 7, 7)))
	static JCudnnActivation x779 = new JCudnnActivation(new int[]{64,512,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 64, 112, 112)))
	static JCudnnActivation x30 = new JCudnnActivation(new int[]{64,64,112,112}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 64, 55, 55)))
	static JCudnnActivation x64 = new JCudnnActivation(new int[]{64,64,55,55}, ActivationMode.RELU);
	// 1_bn_bias
	static JCudaTensor x26 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_bias").asJCudaTensor();
	// 1_bn_scale
	static JCudaTensor x25 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_scale").asJCudaTensor();
	// 1_cv_B
	static JCudaTensor x16 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 1_cv_W
	static JCudaTensor x15 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/1_cv_W").asJCudaTensor();
	// 2a1_bn_bias
	static JCudaTensor x60 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_bias").asJCudaTensor();
	// 2a1_bn_scale
	static JCudaTensor x59 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_scale").asJCudaTensor();
	// 2a1_cv_B
	static JCudaTensor x46 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 2a1_cv_W
	static JCudaTensor x45 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a1_cv_W").asJCudaTensor();
	// 2a2_a_bn_bias
	static JCudaTensor x53 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_bias").asJCudaTensor();
	// 2a2_a_bn_scale
	static JCudaTensor x52 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_scale").asJCudaTensor();
	// 2a2_a_cv_B
	static JCudaTensor x39 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2a2_a_cv_W
	static JCudaTensor x38 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/2a2_a_cv_W").asJCudaTensor();
	// 2a2_b_bn_bias
	static JCudaTensor x77 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_bias").asJCudaTensor();
	// 2a2_b_bn_scale
	static JCudaTensor x76 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_scale").asJCudaTensor();
	// 2a2_b_cv_B
	static JCudaTensor x70 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2a2_b_cv_W
	static JCudaTensor x69 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2a2_b_cv_W").asJCudaTensor();
	// 2a2_c_bn_bias
	static JCudaTensor x92 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_bias").asJCudaTensor();
	// 2a2_c_bn_scale
	static JCudaTensor x91 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_scale").asJCudaTensor();
	// 2a2_c_cv_B
	static JCudaTensor x86 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 2a2_c_cv_W
	static JCudaTensor x85 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a2_c_cv_W").asJCudaTensor();
	// 2b_a_bn_bias
	static JCudaTensor x116 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_bias").asJCudaTensor();
	// 2b_a_bn_scale
	static JCudaTensor x115 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_scale").asJCudaTensor();
	// 2b_a_cv_B
	static JCudaTensor x109 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2b_a_cv_W
	static JCudaTensor x108 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2b_a_cv_W").asJCudaTensor();
	// 2b_b_bn_bias
	static JCudaTensor x131 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_bias").asJCudaTensor();
	// 2b_b_bn_scale
	static JCudaTensor x130 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_scale").asJCudaTensor();
	// 2b_b_cv_B
	static JCudaTensor x125 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2b_b_cv_W
	static JCudaTensor x124 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2b_b_cv_W").asJCudaTensor();
	// 2b_c_bn_bias
	static JCudaTensor x146 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_bias").asJCudaTensor();
	// 2b_c_bn_scale
	static JCudaTensor x145 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_scale").asJCudaTensor();
	// 2b_c_cv_B
	static JCudaTensor x140 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 2b_c_cv_W
	static JCudaTensor x139 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2b_c_cv_W").asJCudaTensor();
	// 2c_a_bn_bias
	static JCudaTensor x166 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_bias").asJCudaTensor();
	// 2c_a_bn_scale
	static JCudaTensor x165 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_scale").asJCudaTensor();
	// 2c_a_cv_B
	static JCudaTensor x160 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2c_a_cv_W
	static JCudaTensor x159 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2c_a_cv_W").asJCudaTensor();
	// 2c_b_bn_bias
	static JCudaTensor x181 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_bias").asJCudaTensor();
	// 2c_b_bn_scale
	static JCudaTensor x180 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_scale").asJCudaTensor();
	// 2c_b_cv_B
	static JCudaTensor x175 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// 2c_b_cv_W
	static JCudaTensor x174 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2c_b_cv_W").asJCudaTensor();
	// 2c_c_bn_bias
	static JCudaTensor x196 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_bias").asJCudaTensor();
	// 2c_c_bn_scale
	static JCudaTensor x195 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_scale").asJCudaTensor();
	// 2c_c_cv_B
	static JCudaTensor x190 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 2c_c_cv_W
	static JCudaTensor x189 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2c_c_cv_W").asJCudaTensor();
	// 3a1_bn_bias
	static JCudaTensor x224 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_bias").asJCudaTensor();
	// 3a1_bn_scale
	static JCudaTensor x223 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_scale").asJCudaTensor();
	// 3a1_cv_B
	static JCudaTensor x217 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 3a1_cv_W
	static JCudaTensor x216 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 512, 256, 1, 1).load(network_dir + "/3a1_cv_W").asJCudaTensor();
	// 3a2_a_bn_bias
	static JCudaTensor x231 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_bias").asJCudaTensor();
	// 3a2_a_bn_scale
	static JCudaTensor x230 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_scale").asJCudaTensor();
	// 3a2_a_cv_B
	static JCudaTensor x210 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3a2_a_cv_W
	static JCudaTensor x209 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/3a2_a_cv_W").asJCudaTensor();
	// 3a2_b_bn_bias
	static JCudaTensor x248 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_bias").asJCudaTensor();
	// 3a2_b_bn_scale
	static JCudaTensor x247 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_scale").asJCudaTensor();
	// 3a2_b_cv_B
	static JCudaTensor x241 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3a2_b_cv_W
	static JCudaTensor x240 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3a2_b_cv_W").asJCudaTensor();
	// 3a2_c_bn_bias
	static JCudaTensor x264 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_bias").asJCudaTensor();
	// 3a2_c_bn_scale
	static JCudaTensor x263 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_scale").asJCudaTensor();
	// 3a2_c_cv_B
	static JCudaTensor x257 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 3a2_c_cv_W
	static JCudaTensor x256 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3a2_c_cv_W").asJCudaTensor();
	// 3b_a_bn_bias
	static JCudaTensor x288 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_bias").asJCudaTensor();
	// 3b_a_bn_scale
	static JCudaTensor x287 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_scale").asJCudaTensor();
	// 3b_a_cv_B
	static JCudaTensor x281 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3b_a_cv_W
	static JCudaTensor x280 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3b_a_cv_W").asJCudaTensor();
	// 3b_b_bn_bias
	static JCudaTensor x303 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_bias").asJCudaTensor();
	// 3b_b_bn_scale
	static JCudaTensor x302 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_scale").asJCudaTensor();
	// 3b_b_cv_B
	static JCudaTensor x297 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3b_b_cv_W
	static JCudaTensor x296 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3b_b_cv_W").asJCudaTensor();
	// 3b_c_bn_bias
	static JCudaTensor x318 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_bias").asJCudaTensor();
	// 3b_c_bn_scale
	static JCudaTensor x317 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_scale").asJCudaTensor();
	// 3b_c_cv_B
	static JCudaTensor x312 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 3b_c_cv_W
	static JCudaTensor x311 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3b_c_cv_W").asJCudaTensor();
	// 3c_a_bn_bias
	static JCudaTensor x338 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_bias").asJCudaTensor();
	// 3c_a_bn_scale
	static JCudaTensor x337 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_scale").asJCudaTensor();
	// 3c_a_cv_B
	static JCudaTensor x332 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3c_a_cv_W
	static JCudaTensor x331 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3c_a_cv_W").asJCudaTensor();
	// 3c_b_bn_bias
	static JCudaTensor x353 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_bias").asJCudaTensor();
	// 3c_b_bn_scale
	static JCudaTensor x352 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_scale").asJCudaTensor();
	// 3c_b_cv_B
	static JCudaTensor x347 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3c_b_cv_W
	static JCudaTensor x346 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3c_b_cv_W").asJCudaTensor();
	// 3c_c_bn_bias
	static JCudaTensor x368 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_bias").asJCudaTensor();
	// 3c_c_bn_scale
	static JCudaTensor x367 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_scale").asJCudaTensor();
	// 3c_c_cv_B
	static JCudaTensor x362 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 3c_c_cv_W
	static JCudaTensor x361 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3c_c_cv_W").asJCudaTensor();
	// 3d_a_bn_bias
	static JCudaTensor x388 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_bias").asJCudaTensor();
	// 3d_a_bn_scale
	static JCudaTensor x387 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_scale").asJCudaTensor();
	// 3d_a_cv_B
	static JCudaTensor x382 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3d_a_cv_W
	static JCudaTensor x381 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3d_a_cv_W").asJCudaTensor();
	// 3d_b_bn_bias
	static JCudaTensor x403 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_bias").asJCudaTensor();
	// 3d_b_bn_scale
	static JCudaTensor x402 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_scale").asJCudaTensor();
	// 3d_b_cv_B
	static JCudaTensor x397 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// 3d_b_cv_W
	static JCudaTensor x396 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3d_b_cv_W").asJCudaTensor();
	// 3d_c_bn_bias
	static JCudaTensor x418 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_bias").asJCudaTensor();
	// 3d_c_bn_scale
	static JCudaTensor x417 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_scale").asJCudaTensor();
	// 3d_c_cv_B
	static JCudaTensor x412 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 3d_c_cv_W
	static JCudaTensor x411 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3d_c_cv_W").asJCudaTensor();
	// 4a1_bn_bias
	static JCudaTensor x446 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_bias").asJCudaTensor();
	// 4a1_bn_scale
	static JCudaTensor x445 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_scale").asJCudaTensor();
	// 4a1_cv_B
	static JCudaTensor x432 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4a1_cv_W
	static JCudaTensor x431 = JTensor.randomFloat(-0.0625f, 0.0625f, 1024, 512, 1, 1).load(network_dir + "/4a1_cv_W").asJCudaTensor();
	// 4a2_a_bn_bias
	static JCudaTensor x453 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_bias").asJCudaTensor();
	// 4a2_a_bn_scale
	static JCudaTensor x452 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_scale").asJCudaTensor();
	// 4a2_a_cv_B
	static JCudaTensor x439 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4a2_a_cv_W
	static JCudaTensor x438 = JTensor.randomFloat(-0.0625f, 0.0625f, 256, 512, 1, 1).load(network_dir + "/4a2_a_cv_W").asJCudaTensor();
	// 4a2_b_bn_bias
	static JCudaTensor x470 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_bias").asJCudaTensor();
	// 4a2_b_bn_scale
	static JCudaTensor x469 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_scale").asJCudaTensor();
	// 4a2_b_cv_B
	static JCudaTensor x463 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4a2_b_cv_W
	static JCudaTensor x462 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4a2_b_cv_W").asJCudaTensor();
	// 4a2_c_bn_bias
	static JCudaTensor x486 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_bias").asJCudaTensor();
	// 4a2_c_bn_scale
	static JCudaTensor x485 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_scale").asJCudaTensor();
	// 4a2_c_cv_B
	static JCudaTensor x479 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4a2_c_cv_W
	static JCudaTensor x478 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4a2_c_cv_W").asJCudaTensor();
	// 4b_a_bn_bias
	static JCudaTensor x510 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_bias").asJCudaTensor();
	// 4b_a_bn_scale
	static JCudaTensor x509 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_scale").asJCudaTensor();
	// 4b_a_cv_B
	static JCudaTensor x503 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4b_a_cv_W
	static JCudaTensor x502 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4b_a_cv_W").asJCudaTensor();
	// 4b_b_bn_bias
	static JCudaTensor x525 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_bias").asJCudaTensor();
	// 4b_b_bn_scale
	static JCudaTensor x524 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_scale").asJCudaTensor();
	// 4b_b_cv_B
	static JCudaTensor x519 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4b_b_cv_W
	static JCudaTensor x518 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4b_b_cv_W").asJCudaTensor();
	// 4b_c_bn_bias
	static JCudaTensor x540 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_bias").asJCudaTensor();
	// 4b_c_bn_scale
	static JCudaTensor x539 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_scale").asJCudaTensor();
	// 4b_c_cv_B
	static JCudaTensor x534 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4b_c_cv_W
	static JCudaTensor x533 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4b_c_cv_W").asJCudaTensor();
	// 4c_a_bn_bias
	static JCudaTensor x560 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_bias").asJCudaTensor();
	// 4c_a_bn_scale
	static JCudaTensor x559 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_scale").asJCudaTensor();
	// 4c_a_cv_B
	static JCudaTensor x554 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4c_a_cv_W
	static JCudaTensor x553 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4c_a_cv_W").asJCudaTensor();
	// 4c_b_bn_bias
	static JCudaTensor x575 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_bias").asJCudaTensor();
	// 4c_b_bn_scale
	static JCudaTensor x574 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_scale").asJCudaTensor();
	// 4c_b_cv_B
	static JCudaTensor x569 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4c_b_cv_W
	static JCudaTensor x568 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4c_b_cv_W").asJCudaTensor();
	// 4c_c_bn_bias
	static JCudaTensor x590 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_bias").asJCudaTensor();
	// 4c_c_bn_scale
	static JCudaTensor x589 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_scale").asJCudaTensor();
	// 4c_c_cv_B
	static JCudaTensor x584 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4c_c_cv_W
	static JCudaTensor x583 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4c_c_cv_W").asJCudaTensor();
	// 4d_a_bn_bias
	static JCudaTensor x610 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_bias").asJCudaTensor();
	// 4d_a_bn_scale
	static JCudaTensor x609 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_scale").asJCudaTensor();
	// 4d_a_cv_B
	static JCudaTensor x604 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4d_a_cv_W
	static JCudaTensor x603 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4d_a_cv_W").asJCudaTensor();
	// 4d_b_bn_bias
	static JCudaTensor x625 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_bias").asJCudaTensor();
	// 4d_b_bn_scale
	static JCudaTensor x624 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_scale").asJCudaTensor();
	// 4d_b_cv_B
	static JCudaTensor x619 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4d_b_cv_W
	static JCudaTensor x618 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4d_b_cv_W").asJCudaTensor();
	// 4d_c_bn_bias
	static JCudaTensor x640 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_bias").asJCudaTensor();
	// 4d_c_bn_scale
	static JCudaTensor x639 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_scale").asJCudaTensor();
	// 4d_c_cv_B
	static JCudaTensor x634 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4d_c_cv_W
	static JCudaTensor x633 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4d_c_cv_W").asJCudaTensor();
	// 4e_a_bn_bias
	static JCudaTensor x660 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_bias").asJCudaTensor();
	// 4e_a_bn_scale
	static JCudaTensor x659 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_scale").asJCudaTensor();
	// 4e_a_cv_B
	static JCudaTensor x654 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4e_a_cv_W
	static JCudaTensor x653 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4e_a_cv_W").asJCudaTensor();
	// 4e_b_bn_bias
	static JCudaTensor x675 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_bias").asJCudaTensor();
	// 4e_b_bn_scale
	static JCudaTensor x674 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_scale").asJCudaTensor();
	// 4e_b_cv_B
	static JCudaTensor x669 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4e_b_cv_W
	static JCudaTensor x668 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4e_b_cv_W").asJCudaTensor();
	// 4e_c_bn_bias
	static JCudaTensor x690 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_bias").asJCudaTensor();
	// 4e_c_bn_scale
	static JCudaTensor x689 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_scale").asJCudaTensor();
	// 4e_c_cv_B
	static JCudaTensor x684 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4e_c_cv_W
	static JCudaTensor x683 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4e_c_cv_W").asJCudaTensor();
	// 4f_a_bn_bias
	static JCudaTensor x710 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_bias").asJCudaTensor();
	// 4f_a_bn_scale
	static JCudaTensor x709 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_scale").asJCudaTensor();
	// 4f_a_cv_B
	static JCudaTensor x704 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4f_a_cv_W
	static JCudaTensor x703 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4f_a_cv_W").asJCudaTensor();
	// 4f_b_bn_bias
	static JCudaTensor x725 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_bias").asJCudaTensor();
	// 4f_b_bn_scale
	static JCudaTensor x724 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_scale").asJCudaTensor();
	// 4f_b_cv_B
	static JCudaTensor x719 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// 4f_b_cv_W
	static JCudaTensor x718 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4f_b_cv_W").asJCudaTensor();
	// 4f_c_bn_bias
	static JCudaTensor x740 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_bias").asJCudaTensor();
	// 4f_c_bn_scale
	static JCudaTensor x739 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_scale").asJCudaTensor();
	// 4f_c_cv_B
	static JCudaTensor x734 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// 4f_c_cv_W
	static JCudaTensor x733 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4f_c_cv_W").asJCudaTensor();
	// 5a1_bn_bias
	static JCudaTensor x768 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_bias").asJCudaTensor();
	// 5a1_bn_scale
	static JCudaTensor x767 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_scale").asJCudaTensor();
	// 5a1_cv_B
	static JCudaTensor x761 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
	// 5a1_cv_W
	static JCudaTensor x760 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 2048, 1024, 1, 1).load(network_dir + "/5a1_cv_W").asJCudaTensor();
	// 5a2_a_bn_bias
	static JCudaTensor x775 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_bias").asJCudaTensor();
	// 5a2_a_bn_scale
	static JCudaTensor x774 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_scale").asJCudaTensor();
	// 5a2_a_cv_B
	static JCudaTensor x754 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5a2_a_cv_W
	static JCudaTensor x753 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 512, 1024, 1, 1).load(network_dir + "/5a2_a_cv_W").asJCudaTensor();
	// 5a2_b_bn_bias
	static JCudaTensor x792 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_bias").asJCudaTensor();
	// 5a2_b_bn_scale
	static JCudaTensor x791 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_scale").asJCudaTensor();
	// 5a2_b_cv_B
	static JCudaTensor x785 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5a2_b_cv_W
	static JCudaTensor x784 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5a2_b_cv_W").asJCudaTensor();
	// 5a2_c_bn_bias
	static JCudaTensor x808 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_bias").asJCudaTensor();
	// 5a2_c_bn_scale
	static JCudaTensor x807 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_scale").asJCudaTensor();
	// 5a2_c_cv_B
	static JCudaTensor x801 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
	// 5a2_c_cv_W
	static JCudaTensor x800 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5a2_c_cv_W").asJCudaTensor();
	// 5b_a_bn_bias
	static JCudaTensor x832 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_bias").asJCudaTensor();
	// 5b_a_bn_scale
	static JCudaTensor x831 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_scale").asJCudaTensor();
	// 5b_a_cv_B
	static JCudaTensor x825 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5b_a_cv_W
	static JCudaTensor x824 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5b_a_cv_W").asJCudaTensor();
	// 5b_b_bn_bias
	static JCudaTensor x847 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_bias").asJCudaTensor();
	// 5b_b_bn_scale
	static JCudaTensor x846 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_scale").asJCudaTensor();
	// 5b_b_cv_B
	static JCudaTensor x841 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5b_b_cv_W
	static JCudaTensor x840 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5b_b_cv_W").asJCudaTensor();
	// 5b_c_bn_bias
	static JCudaTensor x862 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_bias").asJCudaTensor();
	// 5b_c_bn_scale
	static JCudaTensor x861 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_scale").asJCudaTensor();
	// 5b_c_cv_B
	static JCudaTensor x856 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
	// 5b_c_cv_W
	static JCudaTensor x855 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5b_c_cv_W").asJCudaTensor();
	// 5c_a_bn_bias
	static JCudaTensor x882 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_bias").asJCudaTensor();
	// 5c_a_bn_scale
	static JCudaTensor x881 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_scale").asJCudaTensor();
	// 5c_a_cv_B
	static JCudaTensor x876 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5c_a_cv_W
	static JCudaTensor x875 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5c_a_cv_W").asJCudaTensor();
	// 5c_b_bn_bias
	static JCudaTensor x897 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_bias").asJCudaTensor();
	// 5c_b_bn_scale
	static JCudaTensor x896 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_scale").asJCudaTensor();
	// 5c_b_cv_B
	static JCudaTensor x891 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// 5c_b_cv_W
	static JCudaTensor x890 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5c_b_cv_W").asJCudaTensor();
	// 5c_c_bn_bias
	static JCudaTensor x912 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_bias").asJCudaTensor();
	// 5c_c_bn_scale
	static JCudaTensor x911 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_scale").asJCudaTensor();
	// 5c_c_cv_B
	static JCudaTensor x906 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
	// 5c_c_cv_W
	static JCudaTensor x905 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5c_c_cv_W").asJCudaTensor();
	// Precision(Accuracy(X9009, Y, 1))
	static float x5895;
	// V_1_bn_bias
	static JCudaTensor x5112 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_1_bn_scale
	static JCudaTensor x5105 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_1_cv_W
	static JCudaTensor x5119 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
	// V_2a1_bn_bias
	static JCudaTensor x4822 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a1_bn_scale
	static JCudaTensor x4815 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a1_cv_W
	static JCudaTensor x4829 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_bn_bias
	static JCudaTensor x5029 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_bn_scale
	static JCudaTensor x5022 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_cv_W
	static JCudaTensor x5040 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_bn_bias
	static JCudaTensor x4964 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_bn_scale
	static JCudaTensor x4946 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_cv_W
	static JCudaTensor x4956 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2a2_c_bn_bias
	static JCudaTensor x4847 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a2_c_bn_scale
	static JCudaTensor x4840 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a2_c_cv_W
	static JCudaTensor x4854 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2b_a_bn_bias
	static JCudaTensor x4727 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_a_bn_scale
	static JCudaTensor x4720 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_a_cv_W
	static JCudaTensor x4712 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_2b_b_bn_bias
	static JCudaTensor x4640 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_b_bn_scale
	static JCudaTensor x4633 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_b_cv_W
	static JCudaTensor x4650 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2b_c_bn_bias
	static JCudaTensor x4575 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2b_c_bn_scale
	static JCudaTensor x4557 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2b_c_cv_W
	static JCudaTensor x4567 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2c_a_bn_bias
	static JCudaTensor x4484 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_a_bn_scale
	static JCudaTensor x4491 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_a_cv_W
	static JCudaTensor x4476 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_2c_b_bn_bias
	static JCudaTensor x4400 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_b_bn_scale
	static JCudaTensor x4415 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_b_cv_W
	static JCudaTensor x4407 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2c_c_bn_bias
	static JCudaTensor x4331 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2c_c_bn_scale
	static JCudaTensor x4324 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2c_c_cv_W
	static JCudaTensor x4338 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_3a1_bn_bias
	static JCudaTensor x4048 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a1_bn_scale
	static JCudaTensor x4073 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a1_cv_W
	static JCudaTensor x4033 = JTensor.constFloat(0.0f, 512, 256, 1, 1).asJCudaTensor();
	// V_3a2_a_bn_bias
	static JCudaTensor x4240 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_a_bn_scale
	static JCudaTensor x4259 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_a_cv_W
	static JCudaTensor x4251 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_3a2_b_bn_bias
	static JCudaTensor x4175 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_b_bn_scale
	static JCudaTensor x4182 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_b_cv_W
	static JCudaTensor x4167 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3a2_c_bn_bias
	static JCudaTensor x4059 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a2_c_bn_scale
	static JCudaTensor x4066 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a2_c_cv_W
	static JCudaTensor x4041 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3b_a_bn_bias
	static JCudaTensor x3927 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_a_bn_scale
	static JCudaTensor x3934 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_a_cv_W
	static JCudaTensor x3944 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3b_b_bn_bias
	static JCudaTensor x3851 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_b_bn_scale
	static JCudaTensor x3869 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_b_cv_W
	static JCudaTensor x3861 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3b_c_bn_bias
	static JCudaTensor x3786 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3b_c_bn_scale
	static JCudaTensor x3793 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3b_c_cv_W
	static JCudaTensor x3778 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3c_a_bn_bias
	static JCudaTensor x3701 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_a_bn_scale
	static JCudaTensor x3694 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_a_cv_W
	static JCudaTensor x3708 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3c_b_bn_bias
	static JCudaTensor x3625 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_b_bn_scale
	static JCudaTensor x3618 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_b_cv_W
	static JCudaTensor x3632 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3c_c_bn_bias
	static JCudaTensor x3549 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3c_c_bn_scale
	static JCudaTensor x3539 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3c_c_cv_W
	static JCudaTensor x3556 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3d_a_bn_bias
	static JCudaTensor x3455 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_a_bn_scale
	static JCudaTensor x3462 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_a_cv_W
	static JCudaTensor x3472 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3d_b_bn_bias
	static JCudaTensor x3379 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_b_bn_scale
	static JCudaTensor x3397 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_b_cv_W
	static JCudaTensor x3389 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3d_c_bn_bias
	static JCudaTensor x3303 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3d_c_bn_scale
	static JCudaTensor x3310 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3d_c_cv_W
	static JCudaTensor x3320 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_4a1_bn_bias
	static JCudaTensor x3026 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a1_bn_scale
	static JCudaTensor x3040 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a1_cv_W
	static JCudaTensor x3019 = JTensor.constFloat(0.0f, 1024, 512, 1, 1).asJCudaTensor();
	// V_4a2_a_bn_bias
	static JCudaTensor x3241 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_a_bn_scale
	static JCudaTensor x3222 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_a_cv_W
	static JCudaTensor x3233 = JTensor.constFloat(0.0f, 256, 512, 1, 1).asJCudaTensor();
	// V_4a2_b_bn_bias
	static JCudaTensor x3146 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_b_bn_scale
	static JCudaTensor x3153 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_b_cv_W
	static JCudaTensor x3163 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4a2_c_bn_bias
	static JCudaTensor x3055 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a2_c_bn_scale
	static JCudaTensor x3012 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a2_c_cv_W
	static JCudaTensor x3033 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4b_a_bn_bias
	static JCudaTensor x2909 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_a_bn_scale
	static JCudaTensor x2927 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_a_cv_W
	static JCudaTensor x2919 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4b_b_bn_bias
	static JCudaTensor x2851 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_b_bn_scale
	static JCudaTensor x2844 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_b_cv_W
	static JCudaTensor x2836 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4b_c_bn_bias
	static JCudaTensor x2768 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4b_c_bn_scale
	static JCudaTensor x2775 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4b_c_cv_W
	static JCudaTensor x2760 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4c_a_bn_bias
	static JCudaTensor x2684 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_a_bn_scale
	static JCudaTensor x2691 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_a_cv_W
	static JCudaTensor x2676 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4c_b_bn_bias
	static JCudaTensor x2615 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_b_bn_scale
	static JCudaTensor x2597 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_b_cv_W
	static JCudaTensor x2607 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4c_c_bn_bias
	static JCudaTensor x2539 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4c_c_bn_scale
	static JCudaTensor x2521 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4c_c_cv_W
	static JCudaTensor x2531 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4d_a_bn_bias
	static JCudaTensor x2437 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_a_bn_scale
	static JCudaTensor x2447 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_a_cv_W
	static JCudaTensor x2454 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4d_b_bn_bias
	static JCudaTensor x2361 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_b_bn_scale
	static JCudaTensor x2379 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_b_cv_W
	static JCudaTensor x2371 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4d_c_bn_bias
	static JCudaTensor x2303 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4d_c_bn_scale
	static JCudaTensor x2285 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4d_c_cv_W
	static JCudaTensor x2295 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4e_a_bn_bias
	static JCudaTensor x2204 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_a_bn_scale
	static JCudaTensor x2211 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_a_cv_W
	static JCudaTensor x2218 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4e_b_bn_bias
	static JCudaTensor x2128 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_b_bn_scale
	static JCudaTensor x2143 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_b_cv_W
	static JCudaTensor x2135 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4e_c_bn_bias
	static JCudaTensor x2049 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4e_c_bn_scale
	static JCudaTensor x2056 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4e_c_cv_W
	static JCudaTensor x2066 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4f_a_bn_bias
	static JCudaTensor x1976 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_a_bn_scale
	static JCudaTensor x1983 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_a_cv_W
	static JCudaTensor x1968 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4f_b_bn_bias
	static JCudaTensor x1889 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_b_bn_scale
	static JCudaTensor x1907 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_b_cv_W
	static JCudaTensor x1899 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4f_c_bn_bias
	static JCudaTensor x1813 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4f_c_bn_scale
	static JCudaTensor x1831 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4f_c_cv_W
	static JCudaTensor x1823 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_5a1_bn_bias
	static JCudaTensor x1539 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a1_bn_scale
	static JCudaTensor x1546 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a1_cv_W
	static JCudaTensor x1529 = JTensor.constFloat(0.0f, 2048, 1024, 1, 1).asJCudaTensor();
	// V_5a2_a_bn_bias
	static JCudaTensor x1751 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_a_bn_scale
	static JCudaTensor x1732 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_a_cv_W
	static JCudaTensor x1743 = JTensor.constFloat(0.0f, 512, 1024, 1, 1).asJCudaTensor();
	// V_5a2_b_bn_bias
	static JCudaTensor x1674 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_b_bn_scale
	static JCudaTensor x1656 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_b_cv_W
	static JCudaTensor x1666 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5a2_c_bn_bias
	static JCudaTensor x1557 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a2_c_bn_scale
	static JCudaTensor x1522 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a2_c_cv_W
	static JCudaTensor x1564 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_5b_a_bn_bias
	static JCudaTensor x1426 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_a_bn_scale
	static JCudaTensor x1419 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_a_cv_W
	static JCudaTensor x1436 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
	// V_5b_b_bn_bias
	static JCudaTensor x1353 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_b_bn_scale
	static JCudaTensor x1343 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_b_cv_W
	static JCudaTensor x1360 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5b_c_bn_bias
	static JCudaTensor x1267 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5b_c_bn_scale
	static JCudaTensor x1277 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5b_c_cv_W
	static JCudaTensor x1284 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_5c_a_bn_bias
	static JCudaTensor x1183 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_a_bn_scale
	static JCudaTensor x1193 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_a_cv_W
	static JCudaTensor x1200 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
	// V_5c_b_bn_bias
	static JCudaTensor x1114 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_b_bn_scale
	static JCudaTensor x1107 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_b_cv_W
	static JCudaTensor x1124 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5c_c_bn_bias
	static JCudaTensor x1034 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5c_c_bn_scale
	static JCudaTensor x1041 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5c_c_cv_W
	static JCudaTensor x1048 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_fc_B
	static JCudaTensor x966 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc_W
	static JCudaTensor x978 = JTensor.constFloat(0.0f, 1000, 2048).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// fc_B
	static JCudaTensor x934 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
	// fc_W
	static JCudaTensor x930 = JTensor.randomFloat(-0.03125f, 0.03125f, 1000, 2048).load(network_dir + "/fc_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x26.save(network_dir + "/1_bn_bias");
		x25.save(network_dir + "/1_bn_scale");
		x15.save(network_dir + "/1_cv_W");
		x60.save(network_dir + "/2a1_bn_bias");
		x59.save(network_dir + "/2a1_bn_scale");
		x45.save(network_dir + "/2a1_cv_W");
		x53.save(network_dir + "/2a2_a_bn_bias");
		x52.save(network_dir + "/2a2_a_bn_scale");
		x38.save(network_dir + "/2a2_a_cv_W");
		x77.save(network_dir + "/2a2_b_bn_bias");
		x76.save(network_dir + "/2a2_b_bn_scale");
		x69.save(network_dir + "/2a2_b_cv_W");
		x92.save(network_dir + "/2a2_c_bn_bias");
		x91.save(network_dir + "/2a2_c_bn_scale");
		x85.save(network_dir + "/2a2_c_cv_W");
		x116.save(network_dir + "/2b_a_bn_bias");
		x115.save(network_dir + "/2b_a_bn_scale");
		x108.save(network_dir + "/2b_a_cv_W");
		x131.save(network_dir + "/2b_b_bn_bias");
		x130.save(network_dir + "/2b_b_bn_scale");
		x124.save(network_dir + "/2b_b_cv_W");
		x146.save(network_dir + "/2b_c_bn_bias");
		x145.save(network_dir + "/2b_c_bn_scale");
		x139.save(network_dir + "/2b_c_cv_W");
		x166.save(network_dir + "/2c_a_bn_bias");
		x165.save(network_dir + "/2c_a_bn_scale");
		x159.save(network_dir + "/2c_a_cv_W");
		x181.save(network_dir + "/2c_b_bn_bias");
		x180.save(network_dir + "/2c_b_bn_scale");
		x174.save(network_dir + "/2c_b_cv_W");
		x196.save(network_dir + "/2c_c_bn_bias");
		x195.save(network_dir + "/2c_c_bn_scale");
		x189.save(network_dir + "/2c_c_cv_W");
		x224.save(network_dir + "/3a1_bn_bias");
		x223.save(network_dir + "/3a1_bn_scale");
		x216.save(network_dir + "/3a1_cv_W");
		x231.save(network_dir + "/3a2_a_bn_bias");
		x230.save(network_dir + "/3a2_a_bn_scale");
		x209.save(network_dir + "/3a2_a_cv_W");
		x248.save(network_dir + "/3a2_b_bn_bias");
		x247.save(network_dir + "/3a2_b_bn_scale");
		x240.save(network_dir + "/3a2_b_cv_W");
		x264.save(network_dir + "/3a2_c_bn_bias");
		x263.save(network_dir + "/3a2_c_bn_scale");
		x256.save(network_dir + "/3a2_c_cv_W");
		x288.save(network_dir + "/3b_a_bn_bias");
		x287.save(network_dir + "/3b_a_bn_scale");
		x280.save(network_dir + "/3b_a_cv_W");
		x303.save(network_dir + "/3b_b_bn_bias");
		x302.save(network_dir + "/3b_b_bn_scale");
		x296.save(network_dir + "/3b_b_cv_W");
		x318.save(network_dir + "/3b_c_bn_bias");
		x317.save(network_dir + "/3b_c_bn_scale");
		x311.save(network_dir + "/3b_c_cv_W");
		x338.save(network_dir + "/3c_a_bn_bias");
		x337.save(network_dir + "/3c_a_bn_scale");
		x331.save(network_dir + "/3c_a_cv_W");
		x353.save(network_dir + "/3c_b_bn_bias");
		x352.save(network_dir + "/3c_b_bn_scale");
		x346.save(network_dir + "/3c_b_cv_W");
		x368.save(network_dir + "/3c_c_bn_bias");
		x367.save(network_dir + "/3c_c_bn_scale");
		x361.save(network_dir + "/3c_c_cv_W");
		x388.save(network_dir + "/3d_a_bn_bias");
		x387.save(network_dir + "/3d_a_bn_scale");
		x381.save(network_dir + "/3d_a_cv_W");
		x403.save(network_dir + "/3d_b_bn_bias");
		x402.save(network_dir + "/3d_b_bn_scale");
		x396.save(network_dir + "/3d_b_cv_W");
		x418.save(network_dir + "/3d_c_bn_bias");
		x417.save(network_dir + "/3d_c_bn_scale");
		x411.save(network_dir + "/3d_c_cv_W");
		x446.save(network_dir + "/4a1_bn_bias");
		x445.save(network_dir + "/4a1_bn_scale");
		x431.save(network_dir + "/4a1_cv_W");
		x453.save(network_dir + "/4a2_a_bn_bias");
		x452.save(network_dir + "/4a2_a_bn_scale");
		x438.save(network_dir + "/4a2_a_cv_W");
		x470.save(network_dir + "/4a2_b_bn_bias");
		x469.save(network_dir + "/4a2_b_bn_scale");
		x462.save(network_dir + "/4a2_b_cv_W");
		x486.save(network_dir + "/4a2_c_bn_bias");
		x485.save(network_dir + "/4a2_c_bn_scale");
		x478.save(network_dir + "/4a2_c_cv_W");
		x510.save(network_dir + "/4b_a_bn_bias");
		x509.save(network_dir + "/4b_a_bn_scale");
		x502.save(network_dir + "/4b_a_cv_W");
		x525.save(network_dir + "/4b_b_bn_bias");
		x524.save(network_dir + "/4b_b_bn_scale");
		x518.save(network_dir + "/4b_b_cv_W");
		x540.save(network_dir + "/4b_c_bn_bias");
		x539.save(network_dir + "/4b_c_bn_scale");
		x533.save(network_dir + "/4b_c_cv_W");
		x560.save(network_dir + "/4c_a_bn_bias");
		x559.save(network_dir + "/4c_a_bn_scale");
		x553.save(network_dir + "/4c_a_cv_W");
		x575.save(network_dir + "/4c_b_bn_bias");
		x574.save(network_dir + "/4c_b_bn_scale");
		x568.save(network_dir + "/4c_b_cv_W");
		x590.save(network_dir + "/4c_c_bn_bias");
		x589.save(network_dir + "/4c_c_bn_scale");
		x583.save(network_dir + "/4c_c_cv_W");
		x610.save(network_dir + "/4d_a_bn_bias");
		x609.save(network_dir + "/4d_a_bn_scale");
		x603.save(network_dir + "/4d_a_cv_W");
		x625.save(network_dir + "/4d_b_bn_bias");
		x624.save(network_dir + "/4d_b_bn_scale");
		x618.save(network_dir + "/4d_b_cv_W");
		x640.save(network_dir + "/4d_c_bn_bias");
		x639.save(network_dir + "/4d_c_bn_scale");
		x633.save(network_dir + "/4d_c_cv_W");
		x660.save(network_dir + "/4e_a_bn_bias");
		x659.save(network_dir + "/4e_a_bn_scale");
		x653.save(network_dir + "/4e_a_cv_W");
		x675.save(network_dir + "/4e_b_bn_bias");
		x674.save(network_dir + "/4e_b_bn_scale");
		x668.save(network_dir + "/4e_b_cv_W");
		x690.save(network_dir + "/4e_c_bn_bias");
		x689.save(network_dir + "/4e_c_bn_scale");
		x683.save(network_dir + "/4e_c_cv_W");
		x710.save(network_dir + "/4f_a_bn_bias");
		x709.save(network_dir + "/4f_a_bn_scale");
		x703.save(network_dir + "/4f_a_cv_W");
		x725.save(network_dir + "/4f_b_bn_bias");
		x724.save(network_dir + "/4f_b_bn_scale");
		x718.save(network_dir + "/4f_b_cv_W");
		x740.save(network_dir + "/4f_c_bn_bias");
		x739.save(network_dir + "/4f_c_bn_scale");
		x733.save(network_dir + "/4f_c_cv_W");
		x768.save(network_dir + "/5a1_bn_bias");
		x767.save(network_dir + "/5a1_bn_scale");
		x760.save(network_dir + "/5a1_cv_W");
		x775.save(network_dir + "/5a2_a_bn_bias");
		x774.save(network_dir + "/5a2_a_bn_scale");
		x753.save(network_dir + "/5a2_a_cv_W");
		x792.save(network_dir + "/5a2_b_bn_bias");
		x791.save(network_dir + "/5a2_b_bn_scale");
		x784.save(network_dir + "/5a2_b_cv_W");
		x808.save(network_dir + "/5a2_c_bn_bias");
		x807.save(network_dir + "/5a2_c_bn_scale");
		x800.save(network_dir + "/5a2_c_cv_W");
		x832.save(network_dir + "/5b_a_bn_bias");
		x831.save(network_dir + "/5b_a_bn_scale");
		x824.save(network_dir + "/5b_a_cv_W");
		x847.save(network_dir + "/5b_b_bn_bias");
		x846.save(network_dir + "/5b_b_bn_scale");
		x840.save(network_dir + "/5b_b_cv_W");
		x862.save(network_dir + "/5b_c_bn_bias");
		x861.save(network_dir + "/5b_c_bn_scale");
		x855.save(network_dir + "/5b_c_cv_W");
		x882.save(network_dir + "/5c_a_bn_bias");
		x881.save(network_dir + "/5c_a_bn_scale");
		x875.save(network_dir + "/5c_a_cv_W");
		x897.save(network_dir + "/5c_b_bn_bias");
		x896.save(network_dir + "/5c_b_bn_scale");
		x890.save(network_dir + "/5c_b_cv_W");
		x912.save(network_dir + "/5c_c_bn_bias");
		x911.save(network_dir + "/5c_c_bn_scale");
		x905.save(network_dir + "/5c_c_cv_W");
		x934.save(network_dir + "/fc_B");
		x930.save(network_dir + "/fc_W");
		x381.free();
		x1284.free();
		x934.free();
		x5119.free();
		x753.free();
		x4041.free();
		x469.free();
		x53.free();
		x733.free();
		x131.free();
		x15.free();
		x317.free();
		x4847.free();
		x1899.free();
		x116.free();
		x683.free();
		x1193.free();
		x767.free();
		x825.free();
		x855.free();
		x4324.free();
		x774.free();
		x3241.free();
		x337.free();
		x453.free();
		x140.free();
		x906.free();
		x1360.free();
		x1564.free();
		x1557.free();
		x3033.free();
		x890.free();
		x241.free();
		x4633.free();
		x703.free();
		x4033.free();
		x4484.free();
		x847.free();
		x831.free();
		x534.free();
		x180.free();
		x856.free();
		x709.free();
		x1539.free();
		x2285.free();
		x39.free();
		x5105.free();
		x2454.free();
		x841.free();
		x2379.free();
		x3462.free();
		x2676.free();
		x784.free();
		x3625.free();
		x2909.free();
		x397.free();
		x3549.free();
		x247.free();
		x224.free();
		x2361.free();
		x718.free();
		x331.free();
		x775.free();
		x2927.free();
		x175.free();
		x2836.free();
		x124.free();
		x396.free();
		x740.free();
		x1183.free();
		x891.free();
		x2143.free();
		x2539.free();
		x5040.free();
		x674.free();
		x2135.free();
		x2128.free();
		x734.free();
		x3786.free();
		x739.free();
		x125.free();
		x4259.free();
		x3618.free();
		x689.free();
		x130.free();
		x3869.free();
		x159.free();
		x280.free();
		x59.free();
		x1034.free();
		x896.free();
		x1732.free();
		x710.free();
		x2760.free();
		x875.free();
		x486.free();
		x352.free();
		x368.free();
		x231.free();
		x230.free();
		x554.free();
		x181.free();
		x4854.free();
		x518.free();
		x296.free();
		x4822.free();
		x584.free();
		x3163.free();
		x332.free();
		x3310.free();
		x1968.free();
		x353.free();
		x311.free();
		x624.free();
		x704.free();
		x668.free();
		x634.free();
		x445.free();
		x3055.free();
		x297.free();
		x1823.free();
		x145.free();
		x1522.free();
		x618.free();
		x3012.free();
		x3379.free();
		x801.free();
		x5029.free();
		x824.free();
		x4066.free();
		x609.free();
		x4557.free();
		x619.free();
		x590.free();
		x4476.free();
		x2531.free();
		x4400.free();
		x70.free();
		x403.free();
		x675.free();
		x77.free();
		x3861.free();
		x263.free();
		x2066.free();
		x248.free();
		x633.free();
		x25.free();
		x4491.free();
		x196.free();
		x966.free();
		x3233.free();
		x911.free();
		x574.free();
		x1529.free();
		x195.free();
		x4240.free();
		x3701.free();
		x719.free();
		x4407.free();
		x690.free();
		x432.free();
		x519.free();
		x165.free();
		x1666.free();
		x26.free();
		x2295.free();
		x256.free();
		x510.free();
		x660.free();
		x257.free();
		x761.free();
		x338.free();
		x4575.free();
		x4073.free();
		x5112.free();
		x3222.free();
		x470.free();
		x318.free();
		x85.free();
		x724.free();
		x1983.free();
		x553.free();
		x367.free();
		x881.free();
		x388.free();
		x223.free();
		x4251.free();
		x166.free();
		x45.free();
		x288.free();
		x478.free();
		x1353.free();
		x60.free();
		x832.free();
		x768.free();
		x791.free();
		x479.free();
		x2775.free();
		x2691.free();
		x3455.free();
		x2615.free();
		x86.free();
		x684.free();
		x808.free();
		x978.free();
		x792.free();
		x3389.free();
		x3026.free();
		x1436.free();
		x1114.free();
		x217.free();
		x4175.free();
		x625.free();
		x603.free();
		x1419.free();
		x930.free();
		x92.free();
		x639.free();
		x4059.free();
		x108.free();
		x3632.free();
		x91.free();
		x362.free();
		x402.free();
		x52.free();
		x1674.free();
		x382.free();
		x640.free();
		x2844.free();
		x2607.free();
		x1976.free();
		x2371.free();
		x303.free();
		x4712.free();
		x4815.free();
		x1426.free();
		x46.free();
		x725.free();
		x3793.free();
		x4964.free();
		x2303.free();
		x4650.free();
		x3303.free();
		x846.free();
		x139.free();
		x439.free();
		x4727.free();
		x1200.free();
		x1267.free();
		x861.free();
		x1743.free();
		x3934.free();
		x3146.free();
		x4415.free();
		x2056.free();
		x38.free();
		x264.free();
		x446.free();
		x4338.free();
		x2049.free();
		x312.free();
		x69.free();
		x3040.free();
		x2218.free();
		x115.free();
		x575.free();
		x3927.free();
		x569.free();
		x3778.free();
		x4946.free();
		x1907.free();
		x503.free();
		x438.free();
		x1751.free();
		x160.free();
		x146.free();
		x3153.free();
		x5022.free();
		x800.free();
		x109.free();
		x568.free();
		x346.free();
		x540.free();
		x2851.free();
		x418.free();
		x862.free();
		x287.free();
		x3944.free();
		x4840.free();
		x281.free();
		x840.free();
		x3694.free();
		x4048.free();
		x16.free();
		x524.free();
		x4567.free();
		x3556.free();
		x897.free();
		x502.free();
		x589.free();
		x785.free();
		x653.free();
		x905.free();
		x533.free();
		x216.free();
		x1831.free();
		x209.free();
		x1041.free();
		x604.free();
		x417.free();
		x882.free();
		x1546.free();
		x659.free();
		x463.free();
		x411.free();
		x559.free();
		x2597.free();
		x1107.free();
		x4331.free();
		x2447.free();
		x2211.free();
		x2204.free();
		x4956.free();
		x3708.free();
		x347.free();
		x1048.free();
		x1124.free();
		x412.free();
		x2768.free();
		x583.free();
		x2437.free();
		x76.free();
		x912.free();
		x190.free();
		x876.free();
		x485.free();
		x3539.free();
		x525.free();
		x560.free();
		x760.free();
		x654.free();
		x1813.free();
		x2919.free();
		x807.free();
		x509.free();
		x174.free();
		x4720.free();
		x3851.free();
		x189.free();
		x452.free();
		x4640.free();
		x302.free();
		x4182.free();
		x4167.free();
		x2684.free();
		x210.free();
		x3019.free();
		x2521.free();
		x754.free();
		x539.free();
		x3397.free();
		x3320.free();
		x1343.free();
		x4829.free();
		x669.free();
		x240.free();
		x610.free();
		x431.free();
		x462.free();
		x1656.free();
		x3472.free();
		x1277.free();
		x1889.free();
		x387.free();
		x361.free();
		x61.free();
		x937.free();
		x167.free();
		x812.free();
		x404.free();
		x242.free();
		x776.free();
		x232.free();
		x741.free();
		x526.free();
		x235.free();
		x769.free();
		x561.free();
		x898.free();
		x369.free();
		x64.free();
		x661.free();
		x147.free();
		x96.free();
		x319.free();
		x40.free();
		x487.free();
		x71.free();
		x591.free();
		x27.free();
		x339.free();
		x913.free();
		x110.free();
		x54.free();
		x576.free();
		x833.free();
		x447.free();
		x78.free();
		x354.free();
		x225.free();
		x923.free();
		x93.free();
		x490.free();
		x268.free();
		x17.free();
		x480.free();
		x457.free();
		x848.free();
		x863.free();
		x433.free();
		x883.free();
		x691.free();
		x30.free();
		x762.free();
		x182.free();
		x793.free();
		x419.free();
		x511.free();
		x626.free();
		x454.free();
		x258.free();
		x541.free();
		x676.free();
		x211.free();
		x809.free();
		x464.free();
		x779.free();
		x826.free();
		x33.free();
		x132.free();
		x755.free();
		x289.free();
		x282.free();
		x786.free();
		x265.free();
		x711.free();
		x641.free();
		x471.free();
		x504.free();
		x726.free();
		x389.free();
		x218.free();
		x47.free();
		x197.free();
		x440.free();
		x802.free();
		x117.free();
		x611.free();
		x304.free();
		x249.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X8614 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X8812 = Cuda(Indicator(Y, 1000))
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x4.asIndicator(1000);
			x9 = x10.asJCudaTensor();

			// val X8615 = Convolv(2,3)(X8614,1_cv_W,1_cv_B)
			JCudaTensor x11;
			JCudaTensor x12, x13, x14;
			x12 = x7;
			x13 = x15;
			x14 = x16;
			x11 = x17.forward(x12, x13, x14);

			// val X3209 = - X8812.copy
			JCudaTensor x18;
			JCudaTensor x19;
			float x20;
			x19 = x9;
			x19 = x19.clone();
			x20 = -1;
			x18 = x19.times_i(x20);

			// val X8616 = BatchNorm(1_bn)(X8615,1_bn_scale,1_bn_bias)
			JCudaTensor x21;
			JCudaTensor x22, x23, x24;
			x22 = x11;
			x23 = x25;
			x24 = x26;
			x21 = x27.forward(x22, x23, x24);

			// val X8617 = ReLU()(X8616)
			JCudaTensor x28;
			JCudaTensor x29;
			x29 = x21;
			x28 = x30.forward(x29);

			// val X8618 = Pooling(3,2,0,true)(X8617)
			JCudaTensor x31;
			JCudaTensor x32;
			x32 = x28;
			x31 = x33.forward(x32);

			// val X8622 = Convolv(1,0)(X8618,2a2_a_cv_W,2a2_a_cv_B)
			JCudaTensor x34;
			JCudaTensor x35, x36, x37;
			x35 = x31;
			x36 = x38;
			x37 = x39;
			x34 = x40.forward(x35, x36, x37);

			// val X8619 = Convolv(1,0)(X8618,2a1_cv_W,2a1_cv_B)
			JCudaTensor x41;
			JCudaTensor x42, x43, x44;
			x42 = x31;
			x43 = x45;
			x44 = x46;
			x41 = x47.forward(x42, x43, x44);

			// val X8623 = BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale,2a2_a_bn_bias)
			JCudaTensor x48;
			JCudaTensor x49, x50, x51;
			x49 = x34;
			x50 = x52;
			x51 = x53;
			x48 = x54.forward(x49, x50, x51);

			// val X8620 = BatchNorm(2a1_bn)(X8619,2a1_bn_scale,2a1_bn_bias)
			JCudaTensor x55;
			JCudaTensor x56, x57, x58;
			x56 = x41;
			x57 = x59;
			x58 = x60;
			x55 = x61.forward(x56, x57, x58);

			// val X8624 = ReLU()(X8623)
			JCudaTensor x62;
			JCudaTensor x63;
			x63 = x48;
			x62 = x64.forward(x63);

			// val X8625 = Convolv(1,1)(X8624,2a2_b_cv_W,2a2_b_cv_B)
			JCudaTensor x65;
			JCudaTensor x66, x67, x68;
			x66 = x62;
			x67 = x69;
			x68 = x70;
			x65 = x71.forward(x66, x67, x68);

			// val X8626 = BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale,2a2_b_bn_bias)
			JCudaTensor x72;
			JCudaTensor x73, x74, x75;
			x73 = x65;
			x74 = x76;
			x75 = x77;
			x72 = x78.forward(x73, x74, x75);

			// val X8627 = ReLU()(X8626)
			JCudaTensor x79;
			JCudaTensor x80;
			x80 = x72;
			x79 = x64.forward(x80);

			// val X8628 = Convolv(1,0)(X8627,2a2_c_cv_W,2a2_c_cv_B)
			JCudaTensor x81;
			JCudaTensor x82, x83, x84;
			x82 = x79;
			x83 = x85;
			x84 = x86;
			x81 = x47.forward(x82, x83, x84);

			// val X8629 = BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale,2a2_c_bn_bias)
			JCudaTensor x87;
			JCudaTensor x88, x89, x90;
			x88 = x81;
			x89 = x91;
			x90 = x92;
			x87 = x93.forward(x88, x89, x90);

			// val X8621 = ReLU()(X8620)
			JCudaTensor x94;
			JCudaTensor x95;
			x95 = x55;
			x94 = x96.forward(x95);

			// val X8630 = ReLU()(X8629)
			JCudaTensor x97;
			JCudaTensor x98;
			x98 = x87;
			x97 = x96.forward(x98);

			// val X8631 = (X8621.copy + X8630)
			JCudaTensor x99;
			JCudaTensor x100, x101;
			x100 = x94;
			x100 = x100.clone();
			x101 = x97;
			x99 = x100.plus_i(x101);

			// val X8632 = ReLU()(X8631)
			JCudaTensor x102;
			JCudaTensor x103;
			x103 = x99;
			x102 = x96.forward(x103);

			// val X8633 = Convolv(1,0)(X8632,2b_a_cv_W,2b_a_cv_B)
			JCudaTensor x104;
			JCudaTensor x105, x106, x107;
			x105 = x102;
			x106 = x108;
			x107 = x109;
			x104 = x110.forward(x105, x106, x107);

			// val X8634 = BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale,2b_a_bn_bias)
			JCudaTensor x111;
			JCudaTensor x112, x113, x114;
			x112 = x104;
			x113 = x115;
			x114 = x116;
			x111 = x117.forward(x112, x113, x114);

			// val X8635 = ReLU()(X8634)
			JCudaTensor x118;
			JCudaTensor x119;
			x119 = x111;
			x118 = x64.forward(x119);

			// val X8636 = Convolv(1,1)(X8635,2b_b_cv_W,2b_b_cv_B)
			JCudaTensor x120;
			JCudaTensor x121, x122, x123;
			x121 = x118;
			x122 = x124;
			x123 = x125;
			x120 = x71.forward(x121, x122, x123);

			// val X8637 = BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale,2b_b_bn_bias)
			JCudaTensor x126;
			JCudaTensor x127, x128, x129;
			x127 = x120;
			x128 = x130;
			x129 = x131;
			x126 = x132.forward(x127, x128, x129);

			// val X8638 = ReLU()(X8637)
			JCudaTensor x133;
			JCudaTensor x134;
			x134 = x126;
			x133 = x64.forward(x134);

			// val X8639 = Convolv(1,0)(X8638,2b_c_cv_W,2b_c_cv_B)
			JCudaTensor x135;
			JCudaTensor x136, x137, x138;
			x136 = x133;
			x137 = x139;
			x138 = x140;
			x135 = x47.forward(x136, x137, x138);

			// val X8640 = BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale,2b_c_bn_bias)
			JCudaTensor x141;
			JCudaTensor x142, x143, x144;
			x142 = x135;
			x143 = x145;
			x144 = x146;
			x141 = x147.forward(x142, x143, x144);

			// val X8641 = ReLU()(X8640)
			JCudaTensor x148;
			JCudaTensor x149;
			x149 = x141;
			x148 = x96.forward(x149);

			// val X8642 = (X8641.copy + X8632)
			JCudaTensor x150;
			JCudaTensor x151, x152;
			x151 = x148;
			x151 = x151.clone();
			x152 = x102;
			x150 = x151.plus_i(x152);

			// val X8643 = ReLU()(X8642)
			JCudaTensor x153;
			JCudaTensor x154;
			x154 = x150;
			x153 = x96.forward(x154);

			// val X8644 = Convolv(1,0)(X8643,2c_a_cv_W,2c_a_cv_B)
			JCudaTensor x155;
			JCudaTensor x156, x157, x158;
			x156 = x153;
			x157 = x159;
			x158 = x160;
			x155 = x110.forward(x156, x157, x158);

			// val X8645 = BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale,2c_a_bn_bias)
			JCudaTensor x161;
			JCudaTensor x162, x163, x164;
			x162 = x155;
			x163 = x165;
			x164 = x166;
			x161 = x167.forward(x162, x163, x164);

			// val X8646 = ReLU()(X8645)
			JCudaTensor x168;
			JCudaTensor x169;
			x169 = x161;
			x168 = x64.forward(x169);

			// val X8647 = Convolv(1,1)(X8646,2c_b_cv_W,2c_b_cv_B)
			JCudaTensor x170;
			JCudaTensor x171, x172, x173;
			x171 = x168;
			x172 = x174;
			x173 = x175;
			x170 = x71.forward(x171, x172, x173);

			// val X8648 = BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale,2c_b_bn_bias)
			JCudaTensor x176;
			JCudaTensor x177, x178, x179;
			x177 = x170;
			x178 = x180;
			x179 = x181;
			x176 = x182.forward(x177, x178, x179);

			// val X8649 = ReLU()(X8648)
			JCudaTensor x183;
			JCudaTensor x184;
			x184 = x176;
			x183 = x64.forward(x184);

			// val X8650 = Convolv(1,0)(X8649,2c_c_cv_W,2c_c_cv_B)
			JCudaTensor x185;
			JCudaTensor x186, x187, x188;
			x186 = x183;
			x187 = x189;
			x188 = x190;
			x185 = x47.forward(x186, x187, x188);

			// val X8651 = BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale,2c_c_bn_bias)
			JCudaTensor x191;
			JCudaTensor x192, x193, x194;
			x192 = x185;
			x193 = x195;
			x194 = x196;
			x191 = x197.forward(x192, x193, x194);

			// val X8652 = ReLU()(X8651)
			JCudaTensor x198;
			JCudaTensor x199;
			x199 = x191;
			x198 = x96.forward(x199);

			// val X8653 = (X8652.copy + X8643)
			JCudaTensor x200;
			JCudaTensor x201, x202;
			x201 = x198;
			x201 = x201.clone();
			x202 = x153;
			x200 = x201.plus_i(x202);

			// val X8654 = ReLU()(X8653)
			JCudaTensor x203;
			JCudaTensor x204;
			x204 = x200;
			x203 = x96.forward(x204);

			// val X8658 = Convolv(2,0)(X8654,3a2_a_cv_W,3a2_a_cv_B)
			JCudaTensor x205;
			JCudaTensor x206, x207, x208;
			x206 = x203;
			x207 = x209;
			x208 = x210;
			x205 = x211.forward(x206, x207, x208);

			// val X8655 = Convolv(2,0)(X8654,3a1_cv_W,3a1_cv_B)
			JCudaTensor x212;
			JCudaTensor x213, x214, x215;
			x213 = x203;
			x214 = x216;
			x215 = x217;
			x212 = x218.forward(x213, x214, x215);

			// val X8656 = BatchNorm(3a1_bn)(X8655,3a1_bn_scale,3a1_bn_bias)
			JCudaTensor x219;
			JCudaTensor x220, x221, x222;
			x220 = x212;
			x221 = x223;
			x222 = x224;
			x219 = x225.forward(x220, x221, x222);

			// val X8659 = BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale,3a2_a_bn_bias)
			JCudaTensor x226;
			JCudaTensor x227, x228, x229;
			x227 = x205;
			x228 = x230;
			x229 = x231;
			x226 = x232.forward(x227, x228, x229);

			// val X8660 = ReLU()(X8659)
			JCudaTensor x233;
			JCudaTensor x234;
			x234 = x226;
			x233 = x235.forward(x234);

			// val X8661 = Convolv(1,1)(X8660,3a2_b_cv_W,3a2_b_cv_B)
			JCudaTensor x236;
			JCudaTensor x237, x238, x239;
			x237 = x233;
			x238 = x240;
			x239 = x241;
			x236 = x242.forward(x237, x238, x239);

			// val X8662 = BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale,3a2_b_bn_bias)
			JCudaTensor x243;
			JCudaTensor x244, x245, x246;
			x244 = x236;
			x245 = x247;
			x246 = x248;
			x243 = x249.forward(x244, x245, x246);

			// val X8663 = ReLU()(X8662)
			JCudaTensor x250;
			JCudaTensor x251;
			x251 = x243;
			x250 = x235.forward(x251);

			// val X8664 = Convolv(1,0)(X8663,3a2_c_cv_W,3a2_c_cv_B)
			JCudaTensor x252;
			JCudaTensor x253, x254, x255;
			x253 = x250;
			x254 = x256;
			x255 = x257;
			x252 = x258.forward(x253, x254, x255);

			// val X8665 = BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale,3a2_c_bn_bias)
			JCudaTensor x259;
			JCudaTensor x260, x261, x262;
			x260 = x252;
			x261 = x263;
			x262 = x264;
			x259 = x265.forward(x260, x261, x262);

			// val X8657 = ReLU()(X8656)
			JCudaTensor x266;
			JCudaTensor x267;
			x267 = x219;
			x266 = x268.forward(x267);

			// val X8666 = ReLU()(X8665)
			JCudaTensor x269;
			JCudaTensor x270;
			x270 = x259;
			x269 = x268.forward(x270);

			// val X8667 = (X8657.copy + X8666)
			JCudaTensor x271;
			JCudaTensor x272, x273;
			x272 = x266;
			x272 = x272.clone();
			x273 = x269;
			x271 = x272.plus_i(x273);

			// val X8668 = ReLU()(X8667)
			JCudaTensor x274;
			JCudaTensor x275;
			x275 = x271;
			x274 = x268.forward(x275);

			// val X8669 = Convolv(1,0)(X8668,3b_a_cv_W,3b_a_cv_B)
			JCudaTensor x276;
			JCudaTensor x277, x278, x279;
			x277 = x274;
			x278 = x280;
			x279 = x281;
			x276 = x282.forward(x277, x278, x279);

			// val X8670 = BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale,3b_a_bn_bias)
			JCudaTensor x283;
			JCudaTensor x284, x285, x286;
			x284 = x276;
			x285 = x287;
			x286 = x288;
			x283 = x289.forward(x284, x285, x286);

			// val X8671 = ReLU()(X8670)
			JCudaTensor x290;
			JCudaTensor x291;
			x291 = x283;
			x290 = x235.forward(x291);

			// val X8672 = Convolv(1,1)(X8671,3b_b_cv_W,3b_b_cv_B)
			JCudaTensor x292;
			JCudaTensor x293, x294, x295;
			x293 = x290;
			x294 = x296;
			x295 = x297;
			x292 = x242.forward(x293, x294, x295);

			// val X8673 = BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale,3b_b_bn_bias)
			JCudaTensor x298;
			JCudaTensor x299, x300, x301;
			x299 = x292;
			x300 = x302;
			x301 = x303;
			x298 = x304.forward(x299, x300, x301);

			// val X8674 = ReLU()(X8673)
			JCudaTensor x305;
			JCudaTensor x306;
			x306 = x298;
			x305 = x235.forward(x306);

			// val X8675 = Convolv(1,0)(X8674,3b_c_cv_W,3b_c_cv_B)
			JCudaTensor x307;
			JCudaTensor x308, x309, x310;
			x308 = x305;
			x309 = x311;
			x310 = x312;
			x307 = x258.forward(x308, x309, x310);

			// val X8676 = BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale,3b_c_bn_bias)
			JCudaTensor x313;
			JCudaTensor x314, x315, x316;
			x314 = x307;
			x315 = x317;
			x316 = x318;
			x313 = x319.forward(x314, x315, x316);

			// val X8677 = ReLU()(X8676)
			JCudaTensor x320;
			JCudaTensor x321;
			x321 = x313;
			x320 = x268.forward(x321);

			// val X8678 = (X8677.copy + X8668)
			JCudaTensor x322;
			JCudaTensor x323, x324;
			x323 = x320;
			x323 = x323.clone();
			x324 = x274;
			x322 = x323.plus_i(x324);

			// val X8679 = ReLU()(X8678)
			JCudaTensor x325;
			JCudaTensor x326;
			x326 = x322;
			x325 = x268.forward(x326);

			// val X8680 = Convolv(1,0)(X8679,3c_a_cv_W,3c_a_cv_B)
			JCudaTensor x327;
			JCudaTensor x328, x329, x330;
			x328 = x325;
			x329 = x331;
			x330 = x332;
			x327 = x282.forward(x328, x329, x330);

			// val X8681 = BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale,3c_a_bn_bias)
			JCudaTensor x333;
			JCudaTensor x334, x335, x336;
			x334 = x327;
			x335 = x337;
			x336 = x338;
			x333 = x339.forward(x334, x335, x336);

			// val X8682 = ReLU()(X8681)
			JCudaTensor x340;
			JCudaTensor x341;
			x341 = x333;
			x340 = x235.forward(x341);

			// val X8683 = Convolv(1,1)(X8682,3c_b_cv_W,3c_b_cv_B)
			JCudaTensor x342;
			JCudaTensor x343, x344, x345;
			x343 = x340;
			x344 = x346;
			x345 = x347;
			x342 = x242.forward(x343, x344, x345);

			// val X8684 = BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale,3c_b_bn_bias)
			JCudaTensor x348;
			JCudaTensor x349, x350, x351;
			x349 = x342;
			x350 = x352;
			x351 = x353;
			x348 = x354.forward(x349, x350, x351);

			// val X8685 = ReLU()(X8684)
			JCudaTensor x355;
			JCudaTensor x356;
			x356 = x348;
			x355 = x235.forward(x356);

			// val X8686 = Convolv(1,0)(X8685,3c_c_cv_W,3c_c_cv_B)
			JCudaTensor x357;
			JCudaTensor x358, x359, x360;
			x358 = x355;
			x359 = x361;
			x360 = x362;
			x357 = x258.forward(x358, x359, x360);

			// val X8687 = BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale,3c_c_bn_bias)
			JCudaTensor x363;
			JCudaTensor x364, x365, x366;
			x364 = x357;
			x365 = x367;
			x366 = x368;
			x363 = x369.forward(x364, x365, x366);

			// val X8688 = ReLU()(X8687)
			JCudaTensor x370;
			JCudaTensor x371;
			x371 = x363;
			x370 = x268.forward(x371);

			// val X8689 = (X8688.copy + X8679)
			JCudaTensor x372;
			JCudaTensor x373, x374;
			x373 = x370;
			x373 = x373.clone();
			x374 = x325;
			x372 = x373.plus_i(x374);

			// val X8690 = ReLU()(X8689)
			JCudaTensor x375;
			JCudaTensor x376;
			x376 = x372;
			x375 = x268.forward(x376);

			// val X8691 = Convolv(1,0)(X8690,3d_a_cv_W,3d_a_cv_B)
			JCudaTensor x377;
			JCudaTensor x378, x379, x380;
			x378 = x375;
			x379 = x381;
			x380 = x382;
			x377 = x282.forward(x378, x379, x380);

			// val X8692 = BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale,3d_a_bn_bias)
			JCudaTensor x383;
			JCudaTensor x384, x385, x386;
			x384 = x377;
			x385 = x387;
			x386 = x388;
			x383 = x389.forward(x384, x385, x386);

			// val X8693 = ReLU()(X8692)
			JCudaTensor x390;
			JCudaTensor x391;
			x391 = x383;
			x390 = x235.forward(x391);

			// val X8694 = Convolv(1,1)(X8693,3d_b_cv_W,3d_b_cv_B)
			JCudaTensor x392;
			JCudaTensor x393, x394, x395;
			x393 = x390;
			x394 = x396;
			x395 = x397;
			x392 = x242.forward(x393, x394, x395);

			// val X8695 = BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale,3d_b_bn_bias)
			JCudaTensor x398;
			JCudaTensor x399, x400, x401;
			x399 = x392;
			x400 = x402;
			x401 = x403;
			x398 = x404.forward(x399, x400, x401);

			// val X8696 = ReLU()(X8695)
			JCudaTensor x405;
			JCudaTensor x406;
			x406 = x398;
			x405 = x235.forward(x406);

			// val X8697 = Convolv(1,0)(X8696,3d_c_cv_W,3d_c_cv_B)
			JCudaTensor x407;
			JCudaTensor x408, x409, x410;
			x408 = x405;
			x409 = x411;
			x410 = x412;
			x407 = x258.forward(x408, x409, x410);

			// val X8698 = BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale,3d_c_bn_bias)
			JCudaTensor x413;
			JCudaTensor x414, x415, x416;
			x414 = x407;
			x415 = x417;
			x416 = x418;
			x413 = x419.forward(x414, x415, x416);

			// val X8699 = ReLU()(X8698)
			JCudaTensor x420;
			JCudaTensor x421;
			x421 = x413;
			x420 = x268.forward(x421);

			// val X8700 = (X8699.copy + X8690)
			JCudaTensor x422;
			JCudaTensor x423, x424;
			x423 = x420;
			x423 = x423.clone();
			x424 = x375;
			x422 = x423.plus_i(x424);

			// val X8701 = ReLU()(X8700)
			JCudaTensor x425;
			JCudaTensor x426;
			x426 = x422;
			x425 = x268.forward(x426);

			// val X8702 = Convolv(2,0)(X8701,4a1_cv_W,4a1_cv_B)
			JCudaTensor x427;
			JCudaTensor x428, x429, x430;
			x428 = x425;
			x429 = x431;
			x430 = x432;
			x427 = x433.forward(x428, x429, x430);

			// val X8705 = Convolv(2,0)(X8701,4a2_a_cv_W,4a2_a_cv_B)
			JCudaTensor x434;
			JCudaTensor x435, x436, x437;
			x435 = x425;
			x436 = x438;
			x437 = x439;
			x434 = x440.forward(x435, x436, x437);

			// val X8703 = BatchNorm(4a1_bn)(X8702,4a1_bn_scale,4a1_bn_bias)
			JCudaTensor x441;
			JCudaTensor x442, x443, x444;
			x442 = x427;
			x443 = x445;
			x444 = x446;
			x441 = x447.forward(x442, x443, x444);

			// val X8706 = BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale,4a2_a_bn_bias)
			JCudaTensor x448;
			JCudaTensor x449, x450, x451;
			x449 = x434;
			x450 = x452;
			x451 = x453;
			x448 = x454.forward(x449, x450, x451);

			// val X8707 = ReLU()(X8706)
			JCudaTensor x455;
			JCudaTensor x456;
			x456 = x448;
			x455 = x457.forward(x456);

			// val X8708 = Convolv(1,1)(X8707,4a2_b_cv_W,4a2_b_cv_B)
			JCudaTensor x458;
			JCudaTensor x459, x460, x461;
			x459 = x455;
			x460 = x462;
			x461 = x463;
			x458 = x464.forward(x459, x460, x461);

			// val X8709 = BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale,4a2_b_bn_bias)
			JCudaTensor x465;
			JCudaTensor x466, x467, x468;
			x466 = x458;
			x467 = x469;
			x468 = x470;
			x465 = x471.forward(x466, x467, x468);

			// val X8710 = ReLU()(X8709)
			JCudaTensor x472;
			JCudaTensor x473;
			x473 = x465;
			x472 = x457.forward(x473);

			// val X8711 = Convolv(1,0)(X8710,4a2_c_cv_W,4a2_c_cv_B)
			JCudaTensor x474;
			JCudaTensor x475, x476, x477;
			x475 = x472;
			x476 = x478;
			x477 = x479;
			x474 = x480.forward(x475, x476, x477);

			// val X8712 = BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale,4a2_c_bn_bias)
			JCudaTensor x481;
			JCudaTensor x482, x483, x484;
			x482 = x474;
			x483 = x485;
			x484 = x486;
			x481 = x487.forward(x482, x483, x484);

			// val X8704 = ReLU()(X8703)
			JCudaTensor x488;
			JCudaTensor x489;
			x489 = x441;
			x488 = x490.forward(x489);

			// val X8713 = ReLU()(X8712)
			JCudaTensor x491;
			JCudaTensor x492;
			x492 = x481;
			x491 = x490.forward(x492);

			// val X8714 = (X8704.copy + X8713)
			JCudaTensor x493;
			JCudaTensor x494, x495;
			x494 = x488;
			x494 = x494.clone();
			x495 = x491;
			x493 = x494.plus_i(x495);

			// val X8715 = ReLU()(X8714)
			JCudaTensor x496;
			JCudaTensor x497;
			x497 = x493;
			x496 = x490.forward(x497);

			// val X8716 = Convolv(1,0)(X8715,4b_a_cv_W,4b_a_cv_B)
			JCudaTensor x498;
			JCudaTensor x499, x500, x501;
			x499 = x496;
			x500 = x502;
			x501 = x503;
			x498 = x504.forward(x499, x500, x501);

			// val X8717 = BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale,4b_a_bn_bias)
			JCudaTensor x505;
			JCudaTensor x506, x507, x508;
			x506 = x498;
			x507 = x509;
			x508 = x510;
			x505 = x511.forward(x506, x507, x508);

			// val X8718 = ReLU()(X8717)
			JCudaTensor x512;
			JCudaTensor x513;
			x513 = x505;
			x512 = x457.forward(x513);

			// val X8719 = Convolv(1,1)(X8718,4b_b_cv_W,4b_b_cv_B)
			JCudaTensor x514;
			JCudaTensor x515, x516, x517;
			x515 = x512;
			x516 = x518;
			x517 = x519;
			x514 = x464.forward(x515, x516, x517);

			// val X8720 = BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale,4b_b_bn_bias)
			JCudaTensor x520;
			JCudaTensor x521, x522, x523;
			x521 = x514;
			x522 = x524;
			x523 = x525;
			x520 = x526.forward(x521, x522, x523);

			// val X8721 = ReLU()(X8720)
			JCudaTensor x527;
			JCudaTensor x528;
			x528 = x520;
			x527 = x457.forward(x528);

			// val X8722 = Convolv(1,0)(X8721,4b_c_cv_W,4b_c_cv_B)
			JCudaTensor x529;
			JCudaTensor x530, x531, x532;
			x530 = x527;
			x531 = x533;
			x532 = x534;
			x529 = x480.forward(x530, x531, x532);

			// val X8723 = BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale,4b_c_bn_bias)
			JCudaTensor x535;
			JCudaTensor x536, x537, x538;
			x536 = x529;
			x537 = x539;
			x538 = x540;
			x535 = x541.forward(x536, x537, x538);

			// val X8724 = ReLU()(X8723)
			JCudaTensor x542;
			JCudaTensor x543;
			x543 = x535;
			x542 = x490.forward(x543);

			// val X8725 = (X8724.copy + X8715)
			JCudaTensor x544;
			JCudaTensor x545, x546;
			x545 = x542;
			x545 = x545.clone();
			x546 = x496;
			x544 = x545.plus_i(x546);

			// val X8726 = ReLU()(X8725)
			JCudaTensor x547;
			JCudaTensor x548;
			x548 = x544;
			x547 = x490.forward(x548);

			// val X8727 = Convolv(1,0)(X8726,4c_a_cv_W,4c_a_cv_B)
			JCudaTensor x549;
			JCudaTensor x550, x551, x552;
			x550 = x547;
			x551 = x553;
			x552 = x554;
			x549 = x504.forward(x550, x551, x552);

			// val X8728 = BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale,4c_a_bn_bias)
			JCudaTensor x555;
			JCudaTensor x556, x557, x558;
			x556 = x549;
			x557 = x559;
			x558 = x560;
			x555 = x561.forward(x556, x557, x558);

			// val X8729 = ReLU()(X8728)
			JCudaTensor x562;
			JCudaTensor x563;
			x563 = x555;
			x562 = x457.forward(x563);

			// val X8730 = Convolv(1,1)(X8729,4c_b_cv_W,4c_b_cv_B)
			JCudaTensor x564;
			JCudaTensor x565, x566, x567;
			x565 = x562;
			x566 = x568;
			x567 = x569;
			x564 = x464.forward(x565, x566, x567);

			// val X8731 = BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale,4c_b_bn_bias)
			JCudaTensor x570;
			JCudaTensor x571, x572, x573;
			x571 = x564;
			x572 = x574;
			x573 = x575;
			x570 = x576.forward(x571, x572, x573);

			// val X8732 = ReLU()(X8731)
			JCudaTensor x577;
			JCudaTensor x578;
			x578 = x570;
			x577 = x457.forward(x578);

			// val X8733 = Convolv(1,0)(X8732,4c_c_cv_W,4c_c_cv_B)
			JCudaTensor x579;
			JCudaTensor x580, x581, x582;
			x580 = x577;
			x581 = x583;
			x582 = x584;
			x579 = x480.forward(x580, x581, x582);

			// val X8734 = BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale,4c_c_bn_bias)
			JCudaTensor x585;
			JCudaTensor x586, x587, x588;
			x586 = x579;
			x587 = x589;
			x588 = x590;
			x585 = x591.forward(x586, x587, x588);

			// val X8735 = ReLU()(X8734)
			JCudaTensor x592;
			JCudaTensor x593;
			x593 = x585;
			x592 = x490.forward(x593);

			// val X8736 = (X8735.copy + X8726)
			JCudaTensor x594;
			JCudaTensor x595, x596;
			x595 = x592;
			x595 = x595.clone();
			x596 = x547;
			x594 = x595.plus_i(x596);

			// val X8737 = ReLU()(X8736)
			JCudaTensor x597;
			JCudaTensor x598;
			x598 = x594;
			x597 = x490.forward(x598);

			// val X8738 = Convolv(1,0)(X8737,4d_a_cv_W,4d_a_cv_B)
			JCudaTensor x599;
			JCudaTensor x600, x601, x602;
			x600 = x597;
			x601 = x603;
			x602 = x604;
			x599 = x504.forward(x600, x601, x602);

			// val X8739 = BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale,4d_a_bn_bias)
			JCudaTensor x605;
			JCudaTensor x606, x607, x608;
			x606 = x599;
			x607 = x609;
			x608 = x610;
			x605 = x611.forward(x606, x607, x608);

			// val X8740 = ReLU()(X8739)
			JCudaTensor x612;
			JCudaTensor x613;
			x613 = x605;
			x612 = x457.forward(x613);

			// val X8741 = Convolv(1,1)(X8740,4d_b_cv_W,4d_b_cv_B)
			JCudaTensor x614;
			JCudaTensor x615, x616, x617;
			x615 = x612;
			x616 = x618;
			x617 = x619;
			x614 = x464.forward(x615, x616, x617);

			// val X8742 = BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale,4d_b_bn_bias)
			JCudaTensor x620;
			JCudaTensor x621, x622, x623;
			x621 = x614;
			x622 = x624;
			x623 = x625;
			x620 = x626.forward(x621, x622, x623);

			// val X8743 = ReLU()(X8742)
			JCudaTensor x627;
			JCudaTensor x628;
			x628 = x620;
			x627 = x457.forward(x628);

			// val X8744 = Convolv(1,0)(X8743,4d_c_cv_W,4d_c_cv_B)
			JCudaTensor x629;
			JCudaTensor x630, x631, x632;
			x630 = x627;
			x631 = x633;
			x632 = x634;
			x629 = x480.forward(x630, x631, x632);

			// val X8745 = BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale,4d_c_bn_bias)
			JCudaTensor x635;
			JCudaTensor x636, x637, x638;
			x636 = x629;
			x637 = x639;
			x638 = x640;
			x635 = x641.forward(x636, x637, x638);

			// val X8746 = ReLU()(X8745)
			JCudaTensor x642;
			JCudaTensor x643;
			x643 = x635;
			x642 = x490.forward(x643);

			// val X8747 = (X8746.copy + X8737)
			JCudaTensor x644;
			JCudaTensor x645, x646;
			x645 = x642;
			x645 = x645.clone();
			x646 = x597;
			x644 = x645.plus_i(x646);

			// val X8748 = ReLU()(X8747)
			JCudaTensor x647;
			JCudaTensor x648;
			x648 = x644;
			x647 = x490.forward(x648);

			// val X8749 = Convolv(1,0)(X8748,4e_a_cv_W,4e_a_cv_B)
			JCudaTensor x649;
			JCudaTensor x650, x651, x652;
			x650 = x647;
			x651 = x653;
			x652 = x654;
			x649 = x504.forward(x650, x651, x652);

			// val X8750 = BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale,4e_a_bn_bias)
			JCudaTensor x655;
			JCudaTensor x656, x657, x658;
			x656 = x649;
			x657 = x659;
			x658 = x660;
			x655 = x661.forward(x656, x657, x658);

			// val X8751 = ReLU()(X8750)
			JCudaTensor x662;
			JCudaTensor x663;
			x663 = x655;
			x662 = x457.forward(x663);

			// val X8752 = Convolv(1,1)(X8751,4e_b_cv_W,4e_b_cv_B)
			JCudaTensor x664;
			JCudaTensor x665, x666, x667;
			x665 = x662;
			x666 = x668;
			x667 = x669;
			x664 = x464.forward(x665, x666, x667);

			// val X8753 = BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale,4e_b_bn_bias)
			JCudaTensor x670;
			JCudaTensor x671, x672, x673;
			x671 = x664;
			x672 = x674;
			x673 = x675;
			x670 = x676.forward(x671, x672, x673);

			// val X8754 = ReLU()(X8753)
			JCudaTensor x677;
			JCudaTensor x678;
			x678 = x670;
			x677 = x457.forward(x678);

			// val X8755 = Convolv(1,0)(X8754,4e_c_cv_W,4e_c_cv_B)
			JCudaTensor x679;
			JCudaTensor x680, x681, x682;
			x680 = x677;
			x681 = x683;
			x682 = x684;
			x679 = x480.forward(x680, x681, x682);

			// val X8756 = BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale,4e_c_bn_bias)
			JCudaTensor x685;
			JCudaTensor x686, x687, x688;
			x686 = x679;
			x687 = x689;
			x688 = x690;
			x685 = x691.forward(x686, x687, x688);

			// val X8757 = ReLU()(X8756)
			JCudaTensor x692;
			JCudaTensor x693;
			x693 = x685;
			x692 = x490.forward(x693);

			// val X8758 = (X8757.copy + X8748)
			JCudaTensor x694;
			JCudaTensor x695, x696;
			x695 = x692;
			x695 = x695.clone();
			x696 = x647;
			x694 = x695.plus_i(x696);

			// val X8759 = ReLU()(X8758)
			JCudaTensor x697;
			JCudaTensor x698;
			x698 = x694;
			x697 = x490.forward(x698);

			// val X8760 = Convolv(1,0)(X8759,4f_a_cv_W,4f_a_cv_B)
			JCudaTensor x699;
			JCudaTensor x700, x701, x702;
			x700 = x697;
			x701 = x703;
			x702 = x704;
			x699 = x504.forward(x700, x701, x702);

			// val X8761 = BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale,4f_a_bn_bias)
			JCudaTensor x705;
			JCudaTensor x706, x707, x708;
			x706 = x699;
			x707 = x709;
			x708 = x710;
			x705 = x711.forward(x706, x707, x708);

			// val X8762 = ReLU()(X8761)
			JCudaTensor x712;
			JCudaTensor x713;
			x713 = x705;
			x712 = x457.forward(x713);

			// val X8763 = Convolv(1,1)(X8762,4f_b_cv_W,4f_b_cv_B)
			JCudaTensor x714;
			JCudaTensor x715, x716, x717;
			x715 = x712;
			x716 = x718;
			x717 = x719;
			x714 = x464.forward(x715, x716, x717);

			// val X8764 = BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale,4f_b_bn_bias)
			JCudaTensor x720;
			JCudaTensor x721, x722, x723;
			x721 = x714;
			x722 = x724;
			x723 = x725;
			x720 = x726.forward(x721, x722, x723);

			// val X8765 = ReLU()(X8764)
			JCudaTensor x727;
			JCudaTensor x728;
			x728 = x720;
			x727 = x457.forward(x728);

			// val X8766 = Convolv(1,0)(X8765,4f_c_cv_W,4f_c_cv_B)
			JCudaTensor x729;
			JCudaTensor x730, x731, x732;
			x730 = x727;
			x731 = x733;
			x732 = x734;
			x729 = x480.forward(x730, x731, x732);

			// val X8767 = BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale,4f_c_bn_bias)
			JCudaTensor x735;
			JCudaTensor x736, x737, x738;
			x736 = x729;
			x737 = x739;
			x738 = x740;
			x735 = x741.forward(x736, x737, x738);

			// val X8768 = ReLU()(X8767)
			JCudaTensor x742;
			JCudaTensor x743;
			x743 = x735;
			x742 = x490.forward(x743);

			// val X8769 = (X8768.copy + X8759)
			JCudaTensor x744;
			JCudaTensor x745, x746;
			x745 = x742;
			x745 = x745.clone();
			x746 = x697;
			x744 = x745.plus_i(x746);

			// val X8770 = ReLU()(X8769)
			JCudaTensor x747;
			JCudaTensor x748;
			x748 = x744;
			x747 = x490.forward(x748);

			// val X8774 = Convolv(2,0)(X8770,5a2_a_cv_W,5a2_a_cv_B)
			JCudaTensor x749;
			JCudaTensor x750, x751, x752;
			x750 = x747;
			x751 = x753;
			x752 = x754;
			x749 = x755.forward(x750, x751, x752);

			// val X8771 = Convolv(2,0)(X8770,5a1_cv_W,5a1_cv_B)
			JCudaTensor x756;
			JCudaTensor x757, x758, x759;
			x757 = x747;
			x758 = x760;
			x759 = x761;
			x756 = x762.forward(x757, x758, x759);

			// val X8772 = BatchNorm(5a1_bn)(X8771,5a1_bn_scale,5a1_bn_bias)
			JCudaTensor x763;
			JCudaTensor x764, x765, x766;
			x764 = x756;
			x765 = x767;
			x766 = x768;
			x763 = x769.forward(x764, x765, x766);

			// val X8775 = BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale,5a2_a_bn_bias)
			JCudaTensor x770;
			JCudaTensor x771, x772, x773;
			x771 = x749;
			x772 = x774;
			x773 = x775;
			x770 = x776.forward(x771, x772, x773);

			// val X8776 = ReLU()(X8775)
			JCudaTensor x777;
			JCudaTensor x778;
			x778 = x770;
			x777 = x779.forward(x778);

			// val X8777 = Convolv(1,1)(X8776,5a2_b_cv_W,5a2_b_cv_B)
			JCudaTensor x780;
			JCudaTensor x781, x782, x783;
			x781 = x777;
			x782 = x784;
			x783 = x785;
			x780 = x786.forward(x781, x782, x783);

			// val X8778 = BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale,5a2_b_bn_bias)
			JCudaTensor x787;
			JCudaTensor x788, x789, x790;
			x788 = x780;
			x789 = x791;
			x790 = x792;
			x787 = x793.forward(x788, x789, x790);

			// val X8779 = ReLU()(X8778)
			JCudaTensor x794;
			JCudaTensor x795;
			x795 = x787;
			x794 = x779.forward(x795);

			// val X8780 = Convolv(1,0)(X8779,5a2_c_cv_W,5a2_c_cv_B)
			JCudaTensor x796;
			JCudaTensor x797, x798, x799;
			x797 = x794;
			x798 = x800;
			x799 = x801;
			x796 = x802.forward(x797, x798, x799);

			// val X8781 = BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale,5a2_c_bn_bias)
			JCudaTensor x803;
			JCudaTensor x804, x805, x806;
			x804 = x796;
			x805 = x807;
			x806 = x808;
			x803 = x809.forward(x804, x805, x806);

			// val X8773 = ReLU()(X8772)
			JCudaTensor x810;
			JCudaTensor x811;
			x811 = x763;
			x810 = x812.forward(x811);

			// val X8782 = ReLU()(X8781)
			JCudaTensor x813;
			JCudaTensor x814;
			x814 = x803;
			x813 = x812.forward(x814);

			// val X8783 = (X8773.copy + X8782)
			JCudaTensor x815;
			JCudaTensor x816, x817;
			x816 = x810;
			x816 = x816.clone();
			x817 = x813;
			x815 = x816.plus_i(x817);

			// val X8784 = ReLU()(X8783)
			JCudaTensor x818;
			JCudaTensor x819;
			x819 = x815;
			x818 = x812.forward(x819);

			// val X8785 = Convolv(1,0)(X8784,5b_a_cv_W,5b_a_cv_B)
			JCudaTensor x820;
			JCudaTensor x821, x822, x823;
			x821 = x818;
			x822 = x824;
			x823 = x825;
			x820 = x826.forward(x821, x822, x823);

			// val X8786 = BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale,5b_a_bn_bias)
			JCudaTensor x827;
			JCudaTensor x828, x829, x830;
			x828 = x820;
			x829 = x831;
			x830 = x832;
			x827 = x833.forward(x828, x829, x830);

			// val X8787 = ReLU()(X8786)
			JCudaTensor x834;
			JCudaTensor x835;
			x835 = x827;
			x834 = x779.forward(x835);

			// val X8788 = Convolv(1,1)(X8787,5b_b_cv_W,5b_b_cv_B)
			JCudaTensor x836;
			JCudaTensor x837, x838, x839;
			x837 = x834;
			x838 = x840;
			x839 = x841;
			x836 = x786.forward(x837, x838, x839);

			// val X8789 = BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale,5b_b_bn_bias)
			JCudaTensor x842;
			JCudaTensor x843, x844, x845;
			x843 = x836;
			x844 = x846;
			x845 = x847;
			x842 = x848.forward(x843, x844, x845);

			// val X8790 = ReLU()(X8789)
			JCudaTensor x849;
			JCudaTensor x850;
			x850 = x842;
			x849 = x779.forward(x850);

			// val X8791 = Convolv(1,0)(X8790,5b_c_cv_W,5b_c_cv_B)
			JCudaTensor x851;
			JCudaTensor x852, x853, x854;
			x852 = x849;
			x853 = x855;
			x854 = x856;
			x851 = x802.forward(x852, x853, x854);

			// val X8792 = BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale,5b_c_bn_bias)
			JCudaTensor x857;
			JCudaTensor x858, x859, x860;
			x858 = x851;
			x859 = x861;
			x860 = x862;
			x857 = x863.forward(x858, x859, x860);

			// val X8793 = ReLU()(X8792)
			JCudaTensor x864;
			JCudaTensor x865;
			x865 = x857;
			x864 = x812.forward(x865);

			// val X8794 = (X8793.copy + X8784)
			JCudaTensor x866;
			JCudaTensor x867, x868;
			x867 = x864;
			x867 = x867.clone();
			x868 = x818;
			x866 = x867.plus_i(x868);

			// val X8795 = ReLU()(X8794)
			JCudaTensor x869;
			JCudaTensor x870;
			x870 = x866;
			x869 = x812.forward(x870);

			// val X8796 = Convolv(1,0)(X8795,5c_a_cv_W,5c_a_cv_B)
			JCudaTensor x871;
			JCudaTensor x872, x873, x874;
			x872 = x869;
			x873 = x875;
			x874 = x876;
			x871 = x826.forward(x872, x873, x874);

			// val X8797 = BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale,5c_a_bn_bias)
			JCudaTensor x877;
			JCudaTensor x878, x879, x880;
			x878 = x871;
			x879 = x881;
			x880 = x882;
			x877 = x883.forward(x878, x879, x880);

			// val X8798 = ReLU()(X8797)
			JCudaTensor x884;
			JCudaTensor x885;
			x885 = x877;
			x884 = x779.forward(x885);

			// val X8799 = Convolv(1,1)(X8798,5c_b_cv_W,5c_b_cv_B)
			JCudaTensor x886;
			JCudaTensor x887, x888, x889;
			x887 = x884;
			x888 = x890;
			x889 = x891;
			x886 = x786.forward(x887, x888, x889);

			// val X8800 = BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale,5c_b_bn_bias)
			JCudaTensor x892;
			JCudaTensor x893, x894, x895;
			x893 = x886;
			x894 = x896;
			x895 = x897;
			x892 = x898.forward(x893, x894, x895);

			// val X8801 = ReLU()(X8800)
			JCudaTensor x899;
			JCudaTensor x900;
			x900 = x892;
			x899 = x779.forward(x900);

			// val X8802 = Convolv(1,0)(X8801,5c_c_cv_W,5c_c_cv_B)
			JCudaTensor x901;
			JCudaTensor x902, x903, x904;
			x902 = x899;
			x903 = x905;
			x904 = x906;
			x901 = x802.forward(x902, x903, x904);

			// val X8803 = BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale,5c_c_bn_bias)
			JCudaTensor x907;
			JCudaTensor x908, x909, x910;
			x908 = x901;
			x909 = x911;
			x910 = x912;
			x907 = x913.forward(x908, x909, x910);

			// val X8804 = ReLU()(X8803)
			JCudaTensor x914;
			JCudaTensor x915;
			x915 = x907;
			x914 = x812.forward(x915);

			// val X8805 = (X8804.copy + X8795)
			JCudaTensor x916;
			JCudaTensor x917, x918;
			x917 = x914;
			x917 = x917.clone();
			x918 = x869;
			x916 = x917.plus_i(x918);

			// val X8806 = ReLU()(X8805)
			JCudaTensor x919;
			JCudaTensor x920;
			x920 = x916;
			x919 = x812.forward(x920);

			// val X8807 = Pooling(7,1,0,false)(X8806)
			JCudaTensor x921;
			JCudaTensor x922;
			x922 = x919;
			x921 = x923.forward(x922);

			// val X8808 = (X8807[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x924;
			JCudaMatrix x925;
			JCudaMatrix x926;
			JCudaTensor x927;
			JCudaTensor x928;
			x928 = x921;
			x927 = x928.flatten(1, new int[]{2048, 1, 1});
			x925 = x927.asMatrix(1, true);
			JCudaTensor x929;
			x929 = x930;
			x926 = x929.asMatrix(1, true);
			x924 = x925.times(x926);

			// val X8810 = (X8808 + (i) => fc_B)
			JCudaTensor x931;
			JCudaTensor x932, x933;
			x932 = x924;
			x933 = x934;
			x931 = x933.copy(64, x932);

			// val X8811 = LogSoftmax()(X8810)
			JCudaTensor x935;
			JCudaTensor x936;
			x936 = x931;
			x935 = x937.forward(x936);

			// Dealloc(X8810)
			JCudaTensor x938;
			x938 = x931;
			x938.free();

			// val X3210 = (X3209 / |64|)
			JCudaTensor x939;
			JCudaTensor x940;
			float x941;
			x940 = x18;
			float x942;
			x942 = 64;
			x941 = 1 / x942;
			x939 = x940.times_i(x941);

			// Cost(((0 - (X8812 . X8811)) / |64|))
			float x943;
			float x944;
			float x945;
			float x946;
			JCudaTensor x947, x948;
			x947 = x9;
			x948 = x935;
			x946 = x947.dot(x948);
			x944 = - x946;
			x945 = 64;
			x943 = x944 / x945;
			System.out.println(x5 + " " + x943);
			if (Float.isNaN(x943)) { System.exit(-1); }

			// Dealloc(X8812)
			JCudaTensor x949;
			x949 = x9;
			x949.free();

			// val X3409 = X3210 * d_LogSoftmax()(X8811)/d_X8810
			JCudaTensor x950;
			JCudaTensor x951, x952;
			x951 = x939;
			x952 = x935;
			x950 = x937.backward(x951, x952);

			// Dealloc(X3210)
			JCudaTensor x953;
			x953 = x939;
			x953.free();

			// Dealloc(X8811)
			JCudaTensor x954;
			x954 = x935;
			x954.free();

			// val m1 = (i7069) => fc_W[@, i7069]
			JCudaMatrix x955;
			JCudaTensor x956;
			x956 = x930;
			x955 = x956.asMatrix(1, false);

			// val m6 = (i15) => X3409[@, i15]
			JCudaMatrix x957;
			JCudaTensor x958;
			x958 = x950;
			x957 = x958.asMatrix(1, false);

			// val m8 = (i19) => X8807[1><3][@, i19]
			JCudaMatrix x959;
			JCudaTensor x960;
			JCudaTensor x961;
			x961 = x921;
			x960 = x961.flatten(1, new int[]{2048, 1, 1});
			x959 = x960.asMatrix(1, false);

			// val X3603 = (X3409)(i7068 | @) * m1
			JCudaTensor x962;
			JCudaMatrix x963;
			JCudaMatrix x964;
			JCudaTensor x965;
			x965 = x950;
			x963 = x965.asMatrix(1, true);
			x964 = x955;
			x962 = x963.times(x964);

			// V_fc_B <~~ Sum(m6)
			float x967, x968;
			float x969;
			float x970;
			x969 = 1;
			x970 = lrn_rate;
			x967 = x969 * x970;
			x968 = momentum;
			JCudaMatrix x971;
			x971 = x957;
			x971.sum(x966, x967, x968);

			// val X3605 = X3603[1<>3] * d_Pooling(7,1,0,false)(X8807,X8806)/d_X8806
			JCudaTensor x972;
			JCudaTensor x973, x974, x975;
			JCudaTensor x976;
			x976 = x962;
			x973 = x976.unflatten(1, new int[]{2048, 1, 1});
			x974 = x921;
			x975 = x919;
			x972 = x923.backward(x973, x974, x975);

			// Dealloc(X3603)
			JCudaTensor x977;
			x977 = x962;
			x977.free();

			// V_fc_W <~~ m6 * m8
			float x979, x980;
			float x981;
			float x982;
			x981 = 1;
			x982 = lrn_rate;
			x979 = x981 * x982;
			x980 = momentum;
			JCudaMatrix x983;
			JCudaMatrix x984;
			x983 = x957;
			x984 = x959;
			x983.times(x984, x978, x979, x980);

			// Dealloc(X3409)
			JCudaTensor x985;
			x985 = x950;
			x985.free();

			// Dealloc(X8807)
			JCudaTensor x986;
			x986 = x921;
			x986.free();

			// fc_B <~~ V_fc_B
			float x987, x988;
			x987 = 1;
			float x989;
			float x990;
			x989 = 1;
			float x991;
			float x992;
			float x993;
			float x994;
			x993 = 1;
			x994 = decay;
			x991 = x993 * x994;
			float x995;
			float x996;
			x995 = 1;
			x996 = lrn_rate;
			x992 = x995 * x996;
			x990 = x991 * x992;
			x988 = x989 + x990;
			JCudaTensor x997;
			x997 = x966;
			x934.update(x997, x987, x988);

			// fc_W <~~ V_fc_W
			float x998, x999;
			x998 = 1;
			float x1000;
			float x1001;
			x1000 = 1;
			float x1002;
			float x1003;
			float x1004;
			float x1005;
			x1004 = 1;
			x1005 = decay;
			x1002 = x1004 * x1005;
			float x1006;
			float x1007;
			x1006 = 1;
			x1007 = lrn_rate;
			x1003 = x1006 * x1007;
			x1001 = x1002 * x1003;
			x999 = x1000 + x1001;
			JCudaTensor x1008;
			x1008 = x978;
			x930.update(x1008, x998, x999);

			// val X3642 = X3605 * d_ReLU()(X8806)/d_X8805
			JCudaTensor x1009;
			JCudaTensor x1010, x1011;
			x1010 = x972;
			x1011 = x919;
			x1009 = x812.backward(x1010, x1011);

			// Dealloc(X8806)
			JCudaTensor x1012;
			x1012 = x919;
			x1012.free();

			// val X3652 = X3642.copy * d_ReLU()(X8804)/d_X8803
			JCudaTensor x1013;
			JCudaTensor x1014, x1015;
			x1014 = x1009;
			x1014 = x1014.clone();
			x1015 = x914;
			x1013 = x812.backward(x1014, x1015);

			// Dealloc(X8804)
			JCudaTensor x1016;
			x1016 = x914;
			x1016.free();

			// val X8609 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_5c_c_bn_bias
			JCudaTensor x1017;
			JCudaTensor x1018, x1019, x1020;
			x1018 = x1013;
			x1019 = x901;
			x1020 = x911;
			JCudaTensor[] x1021 = x913.backward(x1018,x1019,x1020);
			x1017 = x1021[2];

			// val X3653 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_X8802
			JCudaTensor x1022;
			x1022 = x1021[0];

			// val X8610 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_5c_c_bn_scale
			JCudaTensor x1026;
			x1026 = x1021[1];

			// Dealloc(X8802)
			JCudaTensor x1030;
			x1030 = x901;
			x1030.free();

			// val X3654 = X3653 * d_Convolv(1,0)(5c_c_cv_W)/d_X8801
			JCudaTensor x1031;
			JCudaTensor x1032, x1033;
			x1032 = x1022;
			x1033 = x905;
			x1031 = x802.backward_data(x1032, x1033);

			// V_5c_c_bn_bias <~~ X8609
			float x1035, x1036;
			float x1037;
			float x1038;
			x1037 = 1;
			x1038 = lrn_rate;
			x1035 = x1037 * x1038;
			x1036 = momentum;
			JCudaTensor x1039;
			x1039 = x1017;
			x1034.update(x1039, x1035, x1036);

			// Dealloc(X8609)
			JCudaTensor x1040;
			x1040 = x1017;
			x1040.free();

			// V_5c_c_bn_scale <~~ X8610
			float x1042, x1043;
			float x1044;
			float x1045;
			x1044 = 1;
			x1045 = lrn_rate;
			x1042 = x1044 * x1045;
			x1043 = momentum;
			JCudaTensor x1046;
			x1046 = x1026;
			x1041.update(x1046, x1042, x1043);

			// Dealloc(X8610)
			JCudaTensor x1047;
			x1047 = x1026;
			x1047.free();

			// V_5c_c_cv_W <~~ X3653 * d_Convolv(1,0)(X8801)/d_5c_c_cv_W
			float x1049, x1050;
			float x1051;
			float x1052;
			x1051 = 1;
			x1052 = lrn_rate;
			x1049 = x1051 * x1052;
			x1050 = momentum;
			JCudaTensor x1053, x1054;
			x1053 = x1022;
			x1054 = x899;
			x802.backward_filter(x1053, x1054, x1048, x1049, x1050);

			// Dealloc(X3653)
			JCudaTensor x1055;
			x1055 = x1022;
			x1055.free();

			// 5c_c_bn_bias <~~ V_5c_c_bn_bias
			float x1056, x1057;
			x1056 = 1;
			float x1058;
			float x1059;
			x1058 = 1;
			float x1060;
			float x1061;
			float x1062;
			float x1063;
			x1062 = 1;
			x1063 = decay;
			x1060 = x1062 * x1063;
			float x1064;
			float x1065;
			x1064 = 1;
			x1065 = lrn_rate;
			x1061 = x1064 * x1065;
			x1059 = x1060 * x1061;
			x1057 = x1058 + x1059;
			JCudaTensor x1066;
			x1066 = x1034;
			x912.update(x1066, x1056, x1057);

			// 5c_c_bn_scale <~~ V_5c_c_bn_scale
			float x1067, x1068;
			x1067 = 1;
			float x1069;
			float x1070;
			x1069 = 1;
			float x1071;
			float x1072;
			float x1073;
			float x1074;
			x1073 = 1;
			x1074 = decay;
			x1071 = x1073 * x1074;
			float x1075;
			float x1076;
			x1075 = 1;
			x1076 = lrn_rate;
			x1072 = x1075 * x1076;
			x1070 = x1071 * x1072;
			x1068 = x1069 + x1070;
			JCudaTensor x1077;
			x1077 = x1041;
			x911.update(x1077, x1067, x1068);

			// 5c_c_cv_W <~~ V_5c_c_cv_W
			float x1078, x1079;
			x1078 = 1;
			float x1080;
			float x1081;
			x1080 = 1;
			float x1082;
			float x1083;
			float x1084;
			float x1085;
			x1084 = 1;
			x1085 = decay;
			x1082 = x1084 * x1085;
			float x1086;
			float x1087;
			x1086 = 1;
			x1087 = lrn_rate;
			x1083 = x1086 * x1087;
			x1081 = x1082 * x1083;
			x1079 = x1080 + x1081;
			JCudaTensor x1088;
			x1088 = x1048;
			x905.update(x1088, x1078, x1079);

			// val X3658 = X3654 * d_ReLU()(X8801)/d_X8800
			JCudaTensor x1089;
			JCudaTensor x1090, x1091;
			x1090 = x1031;
			x1091 = x899;
			x1089 = x779.backward(x1090, x1091);

			// Dealloc(X8801)
			JCudaTensor x1092;
			x1092 = x899;
			x1092.free();

			// val X3659 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_X8799
			JCudaTensor x1093;
			JCudaTensor x1094, x1095, x1096;
			x1094 = x1089;
			x1095 = x886;
			x1096 = x896;
			JCudaTensor[] x1097 = x898.backward(x1094,x1095,x1096);
			x1093 = x1097[0];

			// val X8607 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_5c_b_bn_scale
			JCudaTensor x1098;
			x1098 = x1097[1];

			// val X8606 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_5c_b_bn_bias
			JCudaTensor x1102;
			x1102 = x1097[2];

			// Dealloc(X8799)
			JCudaTensor x1106;
			x1106 = x886;
			x1106.free();

			// V_5c_b_bn_scale <~~ X8607
			float x1108, x1109;
			float x1110;
			float x1111;
			x1110 = 1;
			x1111 = lrn_rate;
			x1108 = x1110 * x1111;
			x1109 = momentum;
			JCudaTensor x1112;
			x1112 = x1098;
			x1107.update(x1112, x1108, x1109);

			// Dealloc(X8607)
			JCudaTensor x1113;
			x1113 = x1098;
			x1113.free();

			// V_5c_b_bn_bias <~~ X8606
			float x1115, x1116;
			float x1117;
			float x1118;
			x1117 = 1;
			x1118 = lrn_rate;
			x1115 = x1117 * x1118;
			x1116 = momentum;
			JCudaTensor x1119;
			x1119 = x1102;
			x1114.update(x1119, x1115, x1116);

			// Dealloc(X8606)
			JCudaTensor x1120;
			x1120 = x1102;
			x1120.free();

			// val X3660 = X3659 * d_Convolv(1,1)(5c_b_cv_W)/d_X8798
			JCudaTensor x1121;
			JCudaTensor x1122, x1123;
			x1122 = x1093;
			x1123 = x890;
			x1121 = x786.backward_data(x1122, x1123);

			// V_5c_b_cv_W <~~ X3659 * d_Convolv(1,1)(X8798)/d_5c_b_cv_W
			float x1125, x1126;
			float x1127;
			float x1128;
			x1127 = 1;
			x1128 = lrn_rate;
			x1125 = x1127 * x1128;
			x1126 = momentum;
			JCudaTensor x1129, x1130;
			x1129 = x1093;
			x1130 = x884;
			x786.backward_filter(x1129, x1130, x1124, x1125, x1126);

			// Dealloc(X3659)
			JCudaTensor x1131;
			x1131 = x1093;
			x1131.free();

			// 5c_b_bn_scale <~~ V_5c_b_bn_scale
			float x1132, x1133;
			x1132 = 1;
			float x1134;
			float x1135;
			x1134 = 1;
			float x1136;
			float x1137;
			float x1138;
			float x1139;
			x1138 = 1;
			x1139 = decay;
			x1136 = x1138 * x1139;
			float x1140;
			float x1141;
			x1140 = 1;
			x1141 = lrn_rate;
			x1137 = x1140 * x1141;
			x1135 = x1136 * x1137;
			x1133 = x1134 + x1135;
			JCudaTensor x1142;
			x1142 = x1107;
			x896.update(x1142, x1132, x1133);

			// 5c_b_bn_bias <~~ V_5c_b_bn_bias
			float x1143, x1144;
			x1143 = 1;
			float x1145;
			float x1146;
			x1145 = 1;
			float x1147;
			float x1148;
			float x1149;
			float x1150;
			x1149 = 1;
			x1150 = decay;
			x1147 = x1149 * x1150;
			float x1151;
			float x1152;
			x1151 = 1;
			x1152 = lrn_rate;
			x1148 = x1151 * x1152;
			x1146 = x1147 * x1148;
			x1144 = x1145 + x1146;
			JCudaTensor x1153;
			x1153 = x1114;
			x897.update(x1153, x1143, x1144);

			// 5c_b_cv_W <~~ V_5c_b_cv_W
			float x1154, x1155;
			x1154 = 1;
			float x1156;
			float x1157;
			x1156 = 1;
			float x1158;
			float x1159;
			float x1160;
			float x1161;
			x1160 = 1;
			x1161 = decay;
			x1158 = x1160 * x1161;
			float x1162;
			float x1163;
			x1162 = 1;
			x1163 = lrn_rate;
			x1159 = x1162 * x1163;
			x1157 = x1158 * x1159;
			x1155 = x1156 + x1157;
			JCudaTensor x1164;
			x1164 = x1124;
			x890.update(x1164, x1154, x1155);

			// val X3664 = X3660 * d_ReLU()(X8798)/d_X8797
			JCudaTensor x1165;
			JCudaTensor x1166, x1167;
			x1166 = x1121;
			x1167 = x884;
			x1165 = x779.backward(x1166, x1167);

			// Dealloc(X8798)
			JCudaTensor x1168;
			x1168 = x884;
			x1168.free();

			// val X3665 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_X8796
			JCudaTensor x1169;
			JCudaTensor x1170, x1171, x1172;
			x1170 = x1165;
			x1171 = x871;
			x1172 = x881;
			JCudaTensor[] x1173 = x883.backward(x1170,x1171,x1172);
			x1169 = x1173[0];

			// val X8603 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_5c_a_bn_bias
			JCudaTensor x1174;
			x1174 = x1173[2];

			// val X8604 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_5c_a_bn_scale
			JCudaTensor x1178;
			x1178 = x1173[1];

			// Dealloc(X8796)
			JCudaTensor x1182;
			x1182 = x871;
			x1182.free();

			// V_5c_a_bn_bias <~~ X8603
			float x1184, x1185;
			float x1186;
			float x1187;
			x1186 = 1;
			x1187 = lrn_rate;
			x1184 = x1186 * x1187;
			x1185 = momentum;
			JCudaTensor x1188;
			x1188 = x1174;
			x1183.update(x1188, x1184, x1185);

			// Dealloc(X8603)
			JCudaTensor x1189;
			x1189 = x1174;
			x1189.free();

			// val X3666 = X3665 * d_Convolv(1,0)(5c_a_cv_W)/d_X8795
			JCudaTensor x1190;
			JCudaTensor x1191, x1192;
			x1191 = x1169;
			x1192 = x875;
			x1190 = x826.backward_data(x1191, x1192);

			// V_5c_a_bn_scale <~~ X8604
			float x1194, x1195;
			float x1196;
			float x1197;
			x1196 = 1;
			x1197 = lrn_rate;
			x1194 = x1196 * x1197;
			x1195 = momentum;
			JCudaTensor x1198;
			x1198 = x1178;
			x1193.update(x1198, x1194, x1195);

			// Dealloc(X8604)
			JCudaTensor x1199;
			x1199 = x1178;
			x1199.free();

			// V_5c_a_cv_W <~~ X3665 * d_Convolv(1,0)(X8795)/d_5c_a_cv_W
			float x1201, x1202;
			float x1203;
			float x1204;
			x1203 = 1;
			x1204 = lrn_rate;
			x1201 = x1203 * x1204;
			x1202 = momentum;
			JCudaTensor x1205, x1206;
			x1205 = x1169;
			x1206 = x869;
			x826.backward_filter(x1205, x1206, x1200, x1201, x1202);

			// Dealloc(X3665)
			JCudaTensor x1207;
			x1207 = x1169;
			x1207.free();

			// 5c_a_bn_bias <~~ V_5c_a_bn_bias
			float x1208, x1209;
			x1208 = 1;
			float x1210;
			float x1211;
			x1210 = 1;
			float x1212;
			float x1213;
			float x1214;
			float x1215;
			x1214 = 1;
			x1215 = decay;
			x1212 = x1214 * x1215;
			float x1216;
			float x1217;
			x1216 = 1;
			x1217 = lrn_rate;
			x1213 = x1216 * x1217;
			x1211 = x1212 * x1213;
			x1209 = x1210 + x1211;
			JCudaTensor x1218;
			x1218 = x1183;
			x882.update(x1218, x1208, x1209);

			// 5c_a_bn_scale <~~ V_5c_a_bn_scale
			float x1219, x1220;
			x1219 = 1;
			float x1221;
			float x1222;
			x1221 = 1;
			float x1223;
			float x1224;
			float x1225;
			float x1226;
			x1225 = 1;
			x1226 = decay;
			x1223 = x1225 * x1226;
			float x1227;
			float x1228;
			x1227 = 1;
			x1228 = lrn_rate;
			x1224 = x1227 * x1228;
			x1222 = x1223 * x1224;
			x1220 = x1221 + x1222;
			JCudaTensor x1229;
			x1229 = x1193;
			x881.update(x1229, x1219, x1220);

			// 5c_a_cv_W <~~ V_5c_a_cv_W
			float x1230, x1231;
			x1230 = 1;
			float x1232;
			float x1233;
			x1232 = 1;
			float x1234;
			float x1235;
			float x1236;
			float x1237;
			x1236 = 1;
			x1237 = decay;
			x1234 = x1236 * x1237;
			float x1238;
			float x1239;
			x1238 = 1;
			x1239 = lrn_rate;
			x1235 = x1238 * x1239;
			x1233 = x1234 * x1235;
			x1231 = x1232 + x1233;
			JCudaTensor x1240;
			x1240 = x1200;
			x875.update(x1240, x1230, x1231);

			// val X3667 = (X3666 + X3642)
			JCudaTensor x1241;
			JCudaTensor x1242, x1243;
			x1242 = x1190;
			x1243 = x1009;
			x1241 = x1242.plus_i(x1243);

			// Dealloc(X3642)
			JCudaTensor x1244;
			x1244 = x1009;
			x1244.free();

			// val X3679 = X3667 * d_ReLU()(X8795)/d_X8794
			JCudaTensor x1245;
			JCudaTensor x1246, x1247;
			x1246 = x1241;
			x1247 = x869;
			x1245 = x812.backward(x1246, x1247);

			// Dealloc(X8795)
			JCudaTensor x1248;
			x1248 = x869;
			x1248.free();

			// val X3689 = X3679.copy * d_ReLU()(X8793)/d_X8792
			JCudaTensor x1249;
			JCudaTensor x1250, x1251;
			x1250 = x1245;
			x1250 = x1250.clone();
			x1251 = x864;
			x1249 = x812.backward(x1250, x1251);

			// Dealloc(X8793)
			JCudaTensor x1252;
			x1252 = x864;
			x1252.free();

			// val X8601 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_5b_c_bn_scale
			JCudaTensor x1253;
			JCudaTensor x1254, x1255, x1256;
			x1254 = x1249;
			x1255 = x851;
			x1256 = x861;
			JCudaTensor[] x1257 = x863.backward(x1254,x1255,x1256);
			x1253 = x1257[1];

			// val X8600 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_5b_c_bn_bias
			JCudaTensor x1258;
			x1258 = x1257[2];

			// val X3690 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_X8791
			JCudaTensor x1262;
			x1262 = x1257[0];

			// Dealloc(X8791)
			JCudaTensor x1266;
			x1266 = x851;
			x1266.free();

			// V_5b_c_bn_bias <~~ X8600
			float x1268, x1269;
			float x1270;
			float x1271;
			x1270 = 1;
			x1271 = lrn_rate;
			x1268 = x1270 * x1271;
			x1269 = momentum;
			JCudaTensor x1272;
			x1272 = x1258;
			x1267.update(x1272, x1268, x1269);

			// Dealloc(X8600)
			JCudaTensor x1273;
			x1273 = x1258;
			x1273.free();

			// val X3691 = X3690 * d_Convolv(1,0)(5b_c_cv_W)/d_X8790
			JCudaTensor x1274;
			JCudaTensor x1275, x1276;
			x1275 = x1262;
			x1276 = x855;
			x1274 = x802.backward_data(x1275, x1276);

			// V_5b_c_bn_scale <~~ X8601
			float x1278, x1279;
			float x1280;
			float x1281;
			x1280 = 1;
			x1281 = lrn_rate;
			x1278 = x1280 * x1281;
			x1279 = momentum;
			JCudaTensor x1282;
			x1282 = x1253;
			x1277.update(x1282, x1278, x1279);

			// Dealloc(X8601)
			JCudaTensor x1283;
			x1283 = x1253;
			x1283.free();

			// V_5b_c_cv_W <~~ X3690 * d_Convolv(1,0)(X8790)/d_5b_c_cv_W
			float x1285, x1286;
			float x1287;
			float x1288;
			x1287 = 1;
			x1288 = lrn_rate;
			x1285 = x1287 * x1288;
			x1286 = momentum;
			JCudaTensor x1289, x1290;
			x1289 = x1262;
			x1290 = x849;
			x802.backward_filter(x1289, x1290, x1284, x1285, x1286);

			// Dealloc(X3690)
			JCudaTensor x1291;
			x1291 = x1262;
			x1291.free();

			// 5b_c_bn_bias <~~ V_5b_c_bn_bias
			float x1292, x1293;
			x1292 = 1;
			float x1294;
			float x1295;
			x1294 = 1;
			float x1296;
			float x1297;
			float x1298;
			float x1299;
			x1298 = 1;
			x1299 = decay;
			x1296 = x1298 * x1299;
			float x1300;
			float x1301;
			x1300 = 1;
			x1301 = lrn_rate;
			x1297 = x1300 * x1301;
			x1295 = x1296 * x1297;
			x1293 = x1294 + x1295;
			JCudaTensor x1302;
			x1302 = x1267;
			x862.update(x1302, x1292, x1293);

			// 5b_c_bn_scale <~~ V_5b_c_bn_scale
			float x1303, x1304;
			x1303 = 1;
			float x1305;
			float x1306;
			x1305 = 1;
			float x1307;
			float x1308;
			float x1309;
			float x1310;
			x1309 = 1;
			x1310 = decay;
			x1307 = x1309 * x1310;
			float x1311;
			float x1312;
			x1311 = 1;
			x1312 = lrn_rate;
			x1308 = x1311 * x1312;
			x1306 = x1307 * x1308;
			x1304 = x1305 + x1306;
			JCudaTensor x1313;
			x1313 = x1277;
			x861.update(x1313, x1303, x1304);

			// 5b_c_cv_W <~~ V_5b_c_cv_W
			float x1314, x1315;
			x1314 = 1;
			float x1316;
			float x1317;
			x1316 = 1;
			float x1318;
			float x1319;
			float x1320;
			float x1321;
			x1320 = 1;
			x1321 = decay;
			x1318 = x1320 * x1321;
			float x1322;
			float x1323;
			x1322 = 1;
			x1323 = lrn_rate;
			x1319 = x1322 * x1323;
			x1317 = x1318 * x1319;
			x1315 = x1316 + x1317;
			JCudaTensor x1324;
			x1324 = x1284;
			x855.update(x1324, x1314, x1315);

			// val X3695 = X3691 * d_ReLU()(X8790)/d_X8789
			JCudaTensor x1325;
			JCudaTensor x1326, x1327;
			x1326 = x1274;
			x1327 = x849;
			x1325 = x779.backward(x1326, x1327);

			// Dealloc(X8790)
			JCudaTensor x1328;
			x1328 = x849;
			x1328.free();

			// val X3696 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_X8788
			JCudaTensor x1329;
			JCudaTensor x1330, x1331, x1332;
			x1330 = x1325;
			x1331 = x836;
			x1332 = x846;
			JCudaTensor[] x1333 = x848.backward(x1330,x1331,x1332);
			x1329 = x1333[0];

			// val X8598 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_5b_b_bn_scale
			JCudaTensor x1334;
			x1334 = x1333[1];

			// val X8597 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_5b_b_bn_bias
			JCudaTensor x1338;
			x1338 = x1333[2];

			// Dealloc(X8788)
			JCudaTensor x1342;
			x1342 = x836;
			x1342.free();

			// V_5b_b_bn_scale <~~ X8598
			float x1344, x1345;
			float x1346;
			float x1347;
			x1346 = 1;
			x1347 = lrn_rate;
			x1344 = x1346 * x1347;
			x1345 = momentum;
			JCudaTensor x1348;
			x1348 = x1334;
			x1343.update(x1348, x1344, x1345);

			// Dealloc(X8598)
			JCudaTensor x1349;
			x1349 = x1334;
			x1349.free();

			// val X3697 = X3696 * d_Convolv(1,1)(5b_b_cv_W)/d_X8787
			JCudaTensor x1350;
			JCudaTensor x1351, x1352;
			x1351 = x1329;
			x1352 = x840;
			x1350 = x786.backward_data(x1351, x1352);

			// V_5b_b_bn_bias <~~ X8597
			float x1354, x1355;
			float x1356;
			float x1357;
			x1356 = 1;
			x1357 = lrn_rate;
			x1354 = x1356 * x1357;
			x1355 = momentum;
			JCudaTensor x1358;
			x1358 = x1338;
			x1353.update(x1358, x1354, x1355);

			// Dealloc(X8597)
			JCudaTensor x1359;
			x1359 = x1338;
			x1359.free();

			// V_5b_b_cv_W <~~ X3696 * d_Convolv(1,1)(X8787)/d_5b_b_cv_W
			float x1361, x1362;
			float x1363;
			float x1364;
			x1363 = 1;
			x1364 = lrn_rate;
			x1361 = x1363 * x1364;
			x1362 = momentum;
			JCudaTensor x1365, x1366;
			x1365 = x1329;
			x1366 = x834;
			x786.backward_filter(x1365, x1366, x1360, x1361, x1362);

			// Dealloc(X3696)
			JCudaTensor x1367;
			x1367 = x1329;
			x1367.free();

			// 5b_b_bn_scale <~~ V_5b_b_bn_scale
			float x1368, x1369;
			x1368 = 1;
			float x1370;
			float x1371;
			x1370 = 1;
			float x1372;
			float x1373;
			float x1374;
			float x1375;
			x1374 = 1;
			x1375 = decay;
			x1372 = x1374 * x1375;
			float x1376;
			float x1377;
			x1376 = 1;
			x1377 = lrn_rate;
			x1373 = x1376 * x1377;
			x1371 = x1372 * x1373;
			x1369 = x1370 + x1371;
			JCudaTensor x1378;
			x1378 = x1343;
			x846.update(x1378, x1368, x1369);

			// 5b_b_bn_bias <~~ V_5b_b_bn_bias
			float x1379, x1380;
			x1379 = 1;
			float x1381;
			float x1382;
			x1381 = 1;
			float x1383;
			float x1384;
			float x1385;
			float x1386;
			x1385 = 1;
			x1386 = decay;
			x1383 = x1385 * x1386;
			float x1387;
			float x1388;
			x1387 = 1;
			x1388 = lrn_rate;
			x1384 = x1387 * x1388;
			x1382 = x1383 * x1384;
			x1380 = x1381 + x1382;
			JCudaTensor x1389;
			x1389 = x1353;
			x847.update(x1389, x1379, x1380);

			// 5b_b_cv_W <~~ V_5b_b_cv_W
			float x1390, x1391;
			x1390 = 1;
			float x1392;
			float x1393;
			x1392 = 1;
			float x1394;
			float x1395;
			float x1396;
			float x1397;
			x1396 = 1;
			x1397 = decay;
			x1394 = x1396 * x1397;
			float x1398;
			float x1399;
			x1398 = 1;
			x1399 = lrn_rate;
			x1395 = x1398 * x1399;
			x1393 = x1394 * x1395;
			x1391 = x1392 + x1393;
			JCudaTensor x1400;
			x1400 = x1360;
			x840.update(x1400, x1390, x1391);

			// val X3701 = X3697 * d_ReLU()(X8787)/d_X8786
			JCudaTensor x1401;
			JCudaTensor x1402, x1403;
			x1402 = x1350;
			x1403 = x834;
			x1401 = x779.backward(x1402, x1403);

			// Dealloc(X8787)
			JCudaTensor x1404;
			x1404 = x834;
			x1404.free();

			// val X3702 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_X8785
			JCudaTensor x1405;
			JCudaTensor x1406, x1407, x1408;
			x1406 = x1401;
			x1407 = x820;
			x1408 = x831;
			JCudaTensor[] x1409 = x833.backward(x1406,x1407,x1408);
			x1405 = x1409[0];

			// val X8594 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_5b_a_bn_bias
			JCudaTensor x1410;
			x1410 = x1409[2];

			// val X8595 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_5b_a_bn_scale
			JCudaTensor x1414;
			x1414 = x1409[1];

			// Dealloc(X8785)
			JCudaTensor x1418;
			x1418 = x820;
			x1418.free();

			// V_5b_a_bn_scale <~~ X8595
			float x1420, x1421;
			float x1422;
			float x1423;
			x1422 = 1;
			x1423 = lrn_rate;
			x1420 = x1422 * x1423;
			x1421 = momentum;
			JCudaTensor x1424;
			x1424 = x1414;
			x1419.update(x1424, x1420, x1421);

			// Dealloc(X8595)
			JCudaTensor x1425;
			x1425 = x1414;
			x1425.free();

			// V_5b_a_bn_bias <~~ X8594
			float x1427, x1428;
			float x1429;
			float x1430;
			x1429 = 1;
			x1430 = lrn_rate;
			x1427 = x1429 * x1430;
			x1428 = momentum;
			JCudaTensor x1431;
			x1431 = x1410;
			x1426.update(x1431, x1427, x1428);

			// Dealloc(X8594)
			JCudaTensor x1432;
			x1432 = x1410;
			x1432.free();

			// val X3703 = X3702 * d_Convolv(1,0)(5b_a_cv_W)/d_X8784
			JCudaTensor x1433;
			JCudaTensor x1434, x1435;
			x1434 = x1405;
			x1435 = x824;
			x1433 = x826.backward_data(x1434, x1435);

			// V_5b_a_cv_W <~~ X3702 * d_Convolv(1,0)(X8784)/d_5b_a_cv_W
			float x1437, x1438;
			float x1439;
			float x1440;
			x1439 = 1;
			x1440 = lrn_rate;
			x1437 = x1439 * x1440;
			x1438 = momentum;
			JCudaTensor x1441, x1442;
			x1441 = x1405;
			x1442 = x818;
			x826.backward_filter(x1441, x1442, x1436, x1437, x1438);

			// Dealloc(X3702)
			JCudaTensor x1443;
			x1443 = x1405;
			x1443.free();

			// 5b_a_bn_scale <~~ V_5b_a_bn_scale
			float x1444, x1445;
			x1444 = 1;
			float x1446;
			float x1447;
			x1446 = 1;
			float x1448;
			float x1449;
			float x1450;
			float x1451;
			x1450 = 1;
			x1451 = decay;
			x1448 = x1450 * x1451;
			float x1452;
			float x1453;
			x1452 = 1;
			x1453 = lrn_rate;
			x1449 = x1452 * x1453;
			x1447 = x1448 * x1449;
			x1445 = x1446 + x1447;
			JCudaTensor x1454;
			x1454 = x1419;
			x831.update(x1454, x1444, x1445);

			// 5b_a_bn_bias <~~ V_5b_a_bn_bias
			float x1455, x1456;
			x1455 = 1;
			float x1457;
			float x1458;
			x1457 = 1;
			float x1459;
			float x1460;
			float x1461;
			float x1462;
			x1461 = 1;
			x1462 = decay;
			x1459 = x1461 * x1462;
			float x1463;
			float x1464;
			x1463 = 1;
			x1464 = lrn_rate;
			x1460 = x1463 * x1464;
			x1458 = x1459 * x1460;
			x1456 = x1457 + x1458;
			JCudaTensor x1465;
			x1465 = x1426;
			x832.update(x1465, x1455, x1456);

			// 5b_a_cv_W <~~ V_5b_a_cv_W
			float x1466, x1467;
			x1466 = 1;
			float x1468;
			float x1469;
			x1468 = 1;
			float x1470;
			float x1471;
			float x1472;
			float x1473;
			x1472 = 1;
			x1473 = decay;
			x1470 = x1472 * x1473;
			float x1474;
			float x1475;
			x1474 = 1;
			x1475 = lrn_rate;
			x1471 = x1474 * x1475;
			x1469 = x1470 * x1471;
			x1467 = x1468 + x1469;
			JCudaTensor x1476;
			x1476 = x1436;
			x824.update(x1476, x1466, x1467);

			// val X3704 = (X3703 + X3679)
			JCudaTensor x1477;
			JCudaTensor x1478, x1479;
			x1478 = x1433;
			x1479 = x1245;
			x1477 = x1478.plus_i(x1479);

			// Dealloc(X3679)
			JCudaTensor x1480;
			x1480 = x1245;
			x1480.free();

			// val X3719 = X3704 * d_ReLU()(X8784)/d_X8783
			JCudaTensor x1481;
			JCudaTensor x1482, x1483;
			x1482 = x1477;
			x1483 = x818;
			x1481 = x812.backward(x1482, x1483);

			// Dealloc(X8784)
			JCudaTensor x1484;
			x1484 = x818;
			x1484.free();

			// val X3723 = X3719.copy * d_ReLU()(X8773)/d_X8772
			JCudaTensor x1485;
			JCudaTensor x1486, x1487;
			x1486 = x1481;
			x1486 = x1486.clone();
			x1487 = x810;
			x1485 = x812.backward(x1486, x1487);

			// Dealloc(X8773)
			JCudaTensor x1488;
			x1488 = x810;
			x1488.free();

			// val X3735 = X3719.copy * d_ReLU()(X8782)/d_X8781
			JCudaTensor x1489;
			JCudaTensor x1490, x1491;
			x1490 = x1481;
			x1490 = x1490.clone();
			x1491 = x813;
			x1489 = x812.backward(x1490, x1491);

			// Dealloc(X3719)
			JCudaTensor x1492;
			x1492 = x1481;
			x1492.free();

			// Dealloc(X8782)
			JCudaTensor x1493;
			x1493 = x813;
			x1493.free();

			// val X8583 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_5a1_bn_scale
			JCudaTensor x1494;
			JCudaTensor x1495, x1496, x1497;
			x1495 = x1485;
			x1496 = x756;
			x1497 = x767;
			JCudaTensor[] x1498 = x769.backward(x1495,x1496,x1497);
			x1494 = x1498[1];

			// val X3724 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_X8771
			JCudaTensor x1499;
			x1499 = x1498[0];

			// val X8592 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_5a2_c_bn_scale
			JCudaTensor x1503;
			JCudaTensor x1504, x1505, x1506;
			x1504 = x1489;
			x1505 = x796;
			x1506 = x807;
			JCudaTensor[] x1507 = x809.backward(x1504,x1505,x1506);
			x1503 = x1507[1];

			// val X8582 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_5a1_bn_bias
			JCudaTensor x1508;
			x1508 = x1498[2];

			// Dealloc(X8771)
			JCudaTensor x1512;
			x1512 = x756;
			x1512.free();

			// val X3736 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_X8780
			JCudaTensor x1513;
			x1513 = x1507[0];

			// val X8591 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_5a2_c_bn_bias
			JCudaTensor x1517;
			x1517 = x1507[2];

			// Dealloc(X8780)
			JCudaTensor x1521;
			x1521 = x796;
			x1521.free();

			// V_5a2_c_bn_scale <~~ X8592
			float x1523, x1524;
			float x1525;
			float x1526;
			x1525 = 1;
			x1526 = lrn_rate;
			x1523 = x1525 * x1526;
			x1524 = momentum;
			JCudaTensor x1527;
			x1527 = x1503;
			x1522.update(x1527, x1523, x1524);

			// Dealloc(X8592)
			JCudaTensor x1528;
			x1528 = x1503;
			x1528.free();

			// V_5a1_cv_W <~~ X3724 * d_Convolv(2,0)(X8770)/d_5a1_cv_W
			float x1530, x1531;
			float x1532;
			float x1533;
			x1532 = 1;
			x1533 = lrn_rate;
			x1530 = x1532 * x1533;
			x1531 = momentum;
			JCudaTensor x1534, x1535;
			x1534 = x1499;
			x1535 = x747;
			x762.backward_filter(x1534, x1535, x1529, x1530, x1531);

			// val X3737 = X3736 * d_Convolv(1,0)(5a2_c_cv_W)/d_X8779
			JCudaTensor x1536;
			JCudaTensor x1537, x1538;
			x1537 = x1513;
			x1538 = x800;
			x1536 = x802.backward_data(x1537, x1538);

			// V_5a1_bn_bias <~~ X8582
			float x1540, x1541;
			float x1542;
			float x1543;
			x1542 = 1;
			x1543 = lrn_rate;
			x1540 = x1542 * x1543;
			x1541 = momentum;
			JCudaTensor x1544;
			x1544 = x1508;
			x1539.update(x1544, x1540, x1541);

			// Dealloc(X8582)
			JCudaTensor x1545;
			x1545 = x1508;
			x1545.free();

			// V_5a1_bn_scale <~~ X8583
			float x1547, x1548;
			float x1549;
			float x1550;
			x1549 = 1;
			x1550 = lrn_rate;
			x1547 = x1549 * x1550;
			x1548 = momentum;
			JCudaTensor x1551;
			x1551 = x1494;
			x1546.update(x1551, x1547, x1548);

			// Dealloc(X8583)
			JCudaTensor x1552;
			x1552 = x1494;
			x1552.free();

			// val X3725 = X3724 * d_Convolv(2,0)(5a1_cv_W)/d_X8770
			JCudaTensor x1553;
			JCudaTensor x1554, x1555;
			x1554 = x1499;
			x1555 = x760;
			x1553 = x762.backward_data(x1554, x1555);

			// Dealloc(X3724)
			JCudaTensor x1556;
			x1556 = x1499;
			x1556.free();

			// V_5a2_c_bn_bias <~~ X8591
			float x1558, x1559;
			float x1560;
			float x1561;
			x1560 = 1;
			x1561 = lrn_rate;
			x1558 = x1560 * x1561;
			x1559 = momentum;
			JCudaTensor x1562;
			x1562 = x1517;
			x1557.update(x1562, x1558, x1559);

			// Dealloc(X8591)
			JCudaTensor x1563;
			x1563 = x1517;
			x1563.free();

			// V_5a2_c_cv_W <~~ X3736 * d_Convolv(1,0)(X8779)/d_5a2_c_cv_W
			float x1565, x1566;
			float x1567;
			float x1568;
			x1567 = 1;
			x1568 = lrn_rate;
			x1565 = x1567 * x1568;
			x1566 = momentum;
			JCudaTensor x1569, x1570;
			x1569 = x1513;
			x1570 = x794;
			x802.backward_filter(x1569, x1570, x1564, x1565, x1566);

			// Dealloc(X3736)
			JCudaTensor x1571;
			x1571 = x1513;
			x1571.free();

			// 5a2_c_bn_scale <~~ V_5a2_c_bn_scale
			float x1572, x1573;
			x1572 = 1;
			float x1574;
			float x1575;
			x1574 = 1;
			float x1576;
			float x1577;
			float x1578;
			float x1579;
			x1578 = 1;
			x1579 = decay;
			x1576 = x1578 * x1579;
			float x1580;
			float x1581;
			x1580 = 1;
			x1581 = lrn_rate;
			x1577 = x1580 * x1581;
			x1575 = x1576 * x1577;
			x1573 = x1574 + x1575;
			JCudaTensor x1582;
			x1582 = x1522;
			x807.update(x1582, x1572, x1573);

			// 5a1_cv_W <~~ V_5a1_cv_W
			float x1583, x1584;
			x1583 = 1;
			float x1585;
			float x1586;
			x1585 = 1;
			float x1587;
			float x1588;
			float x1589;
			float x1590;
			x1589 = 1;
			x1590 = decay;
			x1587 = x1589 * x1590;
			float x1591;
			float x1592;
			x1591 = 1;
			x1592 = lrn_rate;
			x1588 = x1591 * x1592;
			x1586 = x1587 * x1588;
			x1584 = x1585 + x1586;
			JCudaTensor x1593;
			x1593 = x1529;
			x760.update(x1593, x1583, x1584);

			// 5a1_bn_bias <~~ V_5a1_bn_bias
			float x1594, x1595;
			x1594 = 1;
			float x1596;
			float x1597;
			x1596 = 1;
			float x1598;
			float x1599;
			float x1600;
			float x1601;
			x1600 = 1;
			x1601 = decay;
			x1598 = x1600 * x1601;
			float x1602;
			float x1603;
			x1602 = 1;
			x1603 = lrn_rate;
			x1599 = x1602 * x1603;
			x1597 = x1598 * x1599;
			x1595 = x1596 + x1597;
			JCudaTensor x1604;
			x1604 = x1539;
			x768.update(x1604, x1594, x1595);

			// 5a1_bn_scale <~~ V_5a1_bn_scale
			float x1605, x1606;
			x1605 = 1;
			float x1607;
			float x1608;
			x1607 = 1;
			float x1609;
			float x1610;
			float x1611;
			float x1612;
			x1611 = 1;
			x1612 = decay;
			x1609 = x1611 * x1612;
			float x1613;
			float x1614;
			x1613 = 1;
			x1614 = lrn_rate;
			x1610 = x1613 * x1614;
			x1608 = x1609 * x1610;
			x1606 = x1607 + x1608;
			JCudaTensor x1615;
			x1615 = x1546;
			x767.update(x1615, x1605, x1606);

			// 5a2_c_cv_W <~~ V_5a2_c_cv_W
			float x1616, x1617;
			x1616 = 1;
			float x1618;
			float x1619;
			x1618 = 1;
			float x1620;
			float x1621;
			float x1622;
			float x1623;
			x1622 = 1;
			x1623 = decay;
			x1620 = x1622 * x1623;
			float x1624;
			float x1625;
			x1624 = 1;
			x1625 = lrn_rate;
			x1621 = x1624 * x1625;
			x1619 = x1620 * x1621;
			x1617 = x1618 + x1619;
			JCudaTensor x1626;
			x1626 = x1564;
			x800.update(x1626, x1616, x1617);

			// 5a2_c_bn_bias <~~ V_5a2_c_bn_bias
			float x1627, x1628;
			x1627 = 1;
			float x1629;
			float x1630;
			x1629 = 1;
			float x1631;
			float x1632;
			float x1633;
			float x1634;
			x1633 = 1;
			x1634 = decay;
			x1631 = x1633 * x1634;
			float x1635;
			float x1636;
			x1635 = 1;
			x1636 = lrn_rate;
			x1632 = x1635 * x1636;
			x1630 = x1631 * x1632;
			x1628 = x1629 + x1630;
			JCudaTensor x1637;
			x1637 = x1557;
			x808.update(x1637, x1627, x1628);

			// val X3741 = X3737 * d_ReLU()(X8779)/d_X8778
			JCudaTensor x1638;
			JCudaTensor x1639, x1640;
			x1639 = x1536;
			x1640 = x794;
			x1638 = x779.backward(x1639, x1640);

			// Dealloc(X8779)
			JCudaTensor x1641;
			x1641 = x794;
			x1641.free();

			// val X3742 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_X8777
			JCudaTensor x1642;
			JCudaTensor x1643, x1644, x1645;
			x1643 = x1638;
			x1644 = x780;
			x1645 = x791;
			JCudaTensor[] x1646 = x793.backward(x1643,x1644,x1645);
			x1642 = x1646[0];

			// val X8589 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_5a2_b_bn_scale
			JCudaTensor x1647;
			x1647 = x1646[1];

			// val X8588 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_5a2_b_bn_bias
			JCudaTensor x1651;
			x1651 = x1646[2];

			// Dealloc(X8777)
			JCudaTensor x1655;
			x1655 = x780;
			x1655.free();

			// V_5a2_b_bn_scale <~~ X8589
			float x1657, x1658;
			float x1659;
			float x1660;
			x1659 = 1;
			x1660 = lrn_rate;
			x1657 = x1659 * x1660;
			x1658 = momentum;
			JCudaTensor x1661;
			x1661 = x1647;
			x1656.update(x1661, x1657, x1658);

			// Dealloc(X8589)
			JCudaTensor x1662;
			x1662 = x1647;
			x1662.free();

			// val X3743 = X3742 * d_Convolv(1,1)(5a2_b_cv_W)/d_X8776
			JCudaTensor x1663;
			JCudaTensor x1664, x1665;
			x1664 = x1642;
			x1665 = x784;
			x1663 = x786.backward_data(x1664, x1665);

			// V_5a2_b_cv_W <~~ X3742 * d_Convolv(1,1)(X8776)/d_5a2_b_cv_W
			float x1667, x1668;
			float x1669;
			float x1670;
			x1669 = 1;
			x1670 = lrn_rate;
			x1667 = x1669 * x1670;
			x1668 = momentum;
			JCudaTensor x1671, x1672;
			x1671 = x1642;
			x1672 = x777;
			x786.backward_filter(x1671, x1672, x1666, x1667, x1668);

			// Dealloc(X3742)
			JCudaTensor x1673;
			x1673 = x1642;
			x1673.free();

			// V_5a2_b_bn_bias <~~ X8588
			float x1675, x1676;
			float x1677;
			float x1678;
			x1677 = 1;
			x1678 = lrn_rate;
			x1675 = x1677 * x1678;
			x1676 = momentum;
			JCudaTensor x1679;
			x1679 = x1651;
			x1674.update(x1679, x1675, x1676);

			// Dealloc(X8588)
			JCudaTensor x1680;
			x1680 = x1651;
			x1680.free();

			// 5a2_b_bn_scale <~~ V_5a2_b_bn_scale
			float x1681, x1682;
			x1681 = 1;
			float x1683;
			float x1684;
			x1683 = 1;
			float x1685;
			float x1686;
			float x1687;
			float x1688;
			x1687 = 1;
			x1688 = decay;
			x1685 = x1687 * x1688;
			float x1689;
			float x1690;
			x1689 = 1;
			x1690 = lrn_rate;
			x1686 = x1689 * x1690;
			x1684 = x1685 * x1686;
			x1682 = x1683 + x1684;
			JCudaTensor x1691;
			x1691 = x1656;
			x791.update(x1691, x1681, x1682);

			// 5a2_b_cv_W <~~ V_5a2_b_cv_W
			float x1692, x1693;
			x1692 = 1;
			float x1694;
			float x1695;
			x1694 = 1;
			float x1696;
			float x1697;
			float x1698;
			float x1699;
			x1698 = 1;
			x1699 = decay;
			x1696 = x1698 * x1699;
			float x1700;
			float x1701;
			x1700 = 1;
			x1701 = lrn_rate;
			x1697 = x1700 * x1701;
			x1695 = x1696 * x1697;
			x1693 = x1694 + x1695;
			JCudaTensor x1702;
			x1702 = x1666;
			x784.update(x1702, x1692, x1693);

			// 5a2_b_bn_bias <~~ V_5a2_b_bn_bias
			float x1703, x1704;
			x1703 = 1;
			float x1705;
			float x1706;
			x1705 = 1;
			float x1707;
			float x1708;
			float x1709;
			float x1710;
			x1709 = 1;
			x1710 = decay;
			x1707 = x1709 * x1710;
			float x1711;
			float x1712;
			x1711 = 1;
			x1712 = lrn_rate;
			x1708 = x1711 * x1712;
			x1706 = x1707 * x1708;
			x1704 = x1705 + x1706;
			JCudaTensor x1713;
			x1713 = x1674;
			x792.update(x1713, x1703, x1704);

			// val X3747 = X3743 * d_ReLU()(X8776)/d_X8775
			JCudaTensor x1714;
			JCudaTensor x1715, x1716;
			x1715 = x1663;
			x1716 = x777;
			x1714 = x779.backward(x1715, x1716);

			// Dealloc(X8776)
			JCudaTensor x1717;
			x1717 = x777;
			x1717.free();

			// val X8585 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_5a2_a_bn_bias
			JCudaTensor x1718;
			JCudaTensor x1719, x1720, x1721;
			x1719 = x1714;
			x1720 = x749;
			x1721 = x774;
			JCudaTensor[] x1722 = x776.backward(x1719,x1720,x1721);
			x1718 = x1722[2];

			// val X3748 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_X8774
			JCudaTensor x1723;
			x1723 = x1722[0];

			// val X8586 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_5a2_a_bn_scale
			JCudaTensor x1727;
			x1727 = x1722[1];

			// Dealloc(X8774)
			JCudaTensor x1731;
			x1731 = x749;
			x1731.free();

			// V_5a2_a_bn_scale <~~ X8586
			float x1733, x1734;
			float x1735;
			float x1736;
			x1735 = 1;
			x1736 = lrn_rate;
			x1733 = x1735 * x1736;
			x1734 = momentum;
			JCudaTensor x1737;
			x1737 = x1727;
			x1732.update(x1737, x1733, x1734);

			// Dealloc(X8586)
			JCudaTensor x1738;
			x1738 = x1727;
			x1738.free();

			// val X3750 = (X3725 + X3748 * d_Convolv(2,0)(5a2_a_cv_W)/d_X8770)
			JCudaTensor x1739;
			JCudaTensor x1740;
			x1740 = x1553;
			JCudaTensor x1741, x1742;
			x1741 = x1723;
			x1742 = x753;
			x1739 = x755.backward_data(x1741,x1742, x1740);

			// V_5a2_a_cv_W <~~ X3748 * d_Convolv(2,0)(X8770)/d_5a2_a_cv_W
			float x1744, x1745;
			float x1746;
			float x1747;
			x1746 = 1;
			x1747 = lrn_rate;
			x1744 = x1746 * x1747;
			x1745 = momentum;
			JCudaTensor x1748, x1749;
			x1748 = x1723;
			x1749 = x747;
			x755.backward_filter(x1748, x1749, x1743, x1744, x1745);

			// Dealloc(X3748)
			JCudaTensor x1750;
			x1750 = x1723;
			x1750.free();

			// V_5a2_a_bn_bias <~~ X8585
			float x1752, x1753;
			float x1754;
			float x1755;
			x1754 = 1;
			x1755 = lrn_rate;
			x1752 = x1754 * x1755;
			x1753 = momentum;
			JCudaTensor x1756;
			x1756 = x1718;
			x1751.update(x1756, x1752, x1753);

			// Dealloc(X8585)
			JCudaTensor x1757;
			x1757 = x1718;
			x1757.free();

			// 5a2_a_bn_scale <~~ V_5a2_a_bn_scale
			float x1758, x1759;
			x1758 = 1;
			float x1760;
			float x1761;
			x1760 = 1;
			float x1762;
			float x1763;
			float x1764;
			float x1765;
			x1764 = 1;
			x1765 = decay;
			x1762 = x1764 * x1765;
			float x1766;
			float x1767;
			x1766 = 1;
			x1767 = lrn_rate;
			x1763 = x1766 * x1767;
			x1761 = x1762 * x1763;
			x1759 = x1760 + x1761;
			JCudaTensor x1768;
			x1768 = x1732;
			x774.update(x1768, x1758, x1759);

			// 5a2_a_cv_W <~~ V_5a2_a_cv_W
			float x1769, x1770;
			x1769 = 1;
			float x1771;
			float x1772;
			x1771 = 1;
			float x1773;
			float x1774;
			float x1775;
			float x1776;
			x1775 = 1;
			x1776 = decay;
			x1773 = x1775 * x1776;
			float x1777;
			float x1778;
			x1777 = 1;
			x1778 = lrn_rate;
			x1774 = x1777 * x1778;
			x1772 = x1773 * x1774;
			x1770 = x1771 + x1772;
			JCudaTensor x1779;
			x1779 = x1743;
			x753.update(x1779, x1769, x1770);

			// 5a2_a_bn_bias <~~ V_5a2_a_bn_bias
			float x1780, x1781;
			x1780 = 1;
			float x1782;
			float x1783;
			x1782 = 1;
			float x1784;
			float x1785;
			float x1786;
			float x1787;
			x1786 = 1;
			x1787 = decay;
			x1784 = x1786 * x1787;
			float x1788;
			float x1789;
			x1788 = 1;
			x1789 = lrn_rate;
			x1785 = x1788 * x1789;
			x1783 = x1784 * x1785;
			x1781 = x1782 + x1783;
			JCudaTensor x1790;
			x1790 = x1751;
			x775.update(x1790, x1780, x1781);

			// val X3820 = X3750 * d_ReLU()(X8770)/d_X8769
			JCudaTensor x1791;
			JCudaTensor x1792, x1793;
			x1792 = x1739;
			x1793 = x747;
			x1791 = x490.backward(x1792, x1793);

			// Dealloc(X8770)
			JCudaTensor x1794;
			x1794 = x747;
			x1794.free();

			// val X3830 = X3820.copy * d_ReLU()(X8768)/d_X8767
			JCudaTensor x1795;
			JCudaTensor x1796, x1797;
			x1796 = x1791;
			x1796 = x1796.clone();
			x1797 = x742;
			x1795 = x490.backward(x1796, x1797);

			// Dealloc(X8768)
			JCudaTensor x1798;
			x1798 = x742;
			x1798.free();

			// val X3831 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_X8766
			JCudaTensor x1799;
			JCudaTensor x1800, x1801, x1802;
			x1800 = x1795;
			x1801 = x729;
			x1802 = x739;
			JCudaTensor[] x1803 = x741.backward(x1800,x1801,x1802);
			x1799 = x1803[0];

			// val X8580 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_4f_c_bn_scale
			JCudaTensor x1804;
			x1804 = x1803[1];

			// val X8579 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_4f_c_bn_bias
			JCudaTensor x1808;
			x1808 = x1803[2];

			// Dealloc(X8766)
			JCudaTensor x1812;
			x1812 = x729;
			x1812.free();

			// V_4f_c_bn_bias <~~ X8579
			float x1814, x1815;
			float x1816;
			float x1817;
			x1816 = 1;
			x1817 = lrn_rate;
			x1814 = x1816 * x1817;
			x1815 = momentum;
			JCudaTensor x1818;
			x1818 = x1808;
			x1813.update(x1818, x1814, x1815);

			// Dealloc(X8579)
			JCudaTensor x1819;
			x1819 = x1808;
			x1819.free();

			// val X3832 = X3831 * d_Convolv(1,0)(4f_c_cv_W)/d_X8765
			JCudaTensor x1820;
			JCudaTensor x1821, x1822;
			x1821 = x1799;
			x1822 = x733;
			x1820 = x480.backward_data(x1821, x1822);

			// V_4f_c_cv_W <~~ X3831 * d_Convolv(1,0)(X8765)/d_4f_c_cv_W
			float x1824, x1825;
			float x1826;
			float x1827;
			x1826 = 1;
			x1827 = lrn_rate;
			x1824 = x1826 * x1827;
			x1825 = momentum;
			JCudaTensor x1828, x1829;
			x1828 = x1799;
			x1829 = x727;
			x480.backward_filter(x1828, x1829, x1823, x1824, x1825);

			// Dealloc(X3831)
			JCudaTensor x1830;
			x1830 = x1799;
			x1830.free();

			// V_4f_c_bn_scale <~~ X8580
			float x1832, x1833;
			float x1834;
			float x1835;
			x1834 = 1;
			x1835 = lrn_rate;
			x1832 = x1834 * x1835;
			x1833 = momentum;
			JCudaTensor x1836;
			x1836 = x1804;
			x1831.update(x1836, x1832, x1833);

			// Dealloc(X8580)
			JCudaTensor x1837;
			x1837 = x1804;
			x1837.free();

			// 4f_c_bn_bias <~~ V_4f_c_bn_bias
			float x1838, x1839;
			x1838 = 1;
			float x1840;
			float x1841;
			x1840 = 1;
			float x1842;
			float x1843;
			float x1844;
			float x1845;
			x1844 = 1;
			x1845 = decay;
			x1842 = x1844 * x1845;
			float x1846;
			float x1847;
			x1846 = 1;
			x1847 = lrn_rate;
			x1843 = x1846 * x1847;
			x1841 = x1842 * x1843;
			x1839 = x1840 + x1841;
			JCudaTensor x1848;
			x1848 = x1813;
			x740.update(x1848, x1838, x1839);

			// 4f_c_cv_W <~~ V_4f_c_cv_W
			float x1849, x1850;
			x1849 = 1;
			float x1851;
			float x1852;
			x1851 = 1;
			float x1853;
			float x1854;
			float x1855;
			float x1856;
			x1855 = 1;
			x1856 = decay;
			x1853 = x1855 * x1856;
			float x1857;
			float x1858;
			x1857 = 1;
			x1858 = lrn_rate;
			x1854 = x1857 * x1858;
			x1852 = x1853 * x1854;
			x1850 = x1851 + x1852;
			JCudaTensor x1859;
			x1859 = x1823;
			x733.update(x1859, x1849, x1850);

			// 4f_c_bn_scale <~~ V_4f_c_bn_scale
			float x1860, x1861;
			x1860 = 1;
			float x1862;
			float x1863;
			x1862 = 1;
			float x1864;
			float x1865;
			float x1866;
			float x1867;
			x1866 = 1;
			x1867 = decay;
			x1864 = x1866 * x1867;
			float x1868;
			float x1869;
			x1868 = 1;
			x1869 = lrn_rate;
			x1865 = x1868 * x1869;
			x1863 = x1864 * x1865;
			x1861 = x1862 + x1863;
			JCudaTensor x1870;
			x1870 = x1831;
			x739.update(x1870, x1860, x1861);

			// val X3836 = X3832 * d_ReLU()(X8765)/d_X8764
			JCudaTensor x1871;
			JCudaTensor x1872, x1873;
			x1872 = x1820;
			x1873 = x727;
			x1871 = x457.backward(x1872, x1873);

			// Dealloc(X8765)
			JCudaTensor x1874;
			x1874 = x727;
			x1874.free();

			// val X8576 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_4f_b_bn_bias
			JCudaTensor x1875;
			JCudaTensor x1876, x1877, x1878;
			x1876 = x1871;
			x1877 = x714;
			x1878 = x724;
			JCudaTensor[] x1879 = x726.backward(x1876,x1877,x1878);
			x1875 = x1879[2];

			// val X3837 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_X8763
			JCudaTensor x1880;
			x1880 = x1879[0];

			// val X8577 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_4f_b_bn_scale
			JCudaTensor x1884;
			x1884 = x1879[1];

			// Dealloc(X8763)
			JCudaTensor x1888;
			x1888 = x714;
			x1888.free();

			// V_4f_b_bn_bias <~~ X8576
			float x1890, x1891;
			float x1892;
			float x1893;
			x1892 = 1;
			x1893 = lrn_rate;
			x1890 = x1892 * x1893;
			x1891 = momentum;
			JCudaTensor x1894;
			x1894 = x1875;
			x1889.update(x1894, x1890, x1891);

			// Dealloc(X8576)
			JCudaTensor x1895;
			x1895 = x1875;
			x1895.free();

			// val X3838 = X3837 * d_Convolv(1,1)(4f_b_cv_W)/d_X8762
			JCudaTensor x1896;
			JCudaTensor x1897, x1898;
			x1897 = x1880;
			x1898 = x718;
			x1896 = x464.backward_data(x1897, x1898);

			// V_4f_b_cv_W <~~ X3837 * d_Convolv(1,1)(X8762)/d_4f_b_cv_W
			float x1900, x1901;
			float x1902;
			float x1903;
			x1902 = 1;
			x1903 = lrn_rate;
			x1900 = x1902 * x1903;
			x1901 = momentum;
			JCudaTensor x1904, x1905;
			x1904 = x1880;
			x1905 = x712;
			x464.backward_filter(x1904, x1905, x1899, x1900, x1901);

			// Dealloc(X3837)
			JCudaTensor x1906;
			x1906 = x1880;
			x1906.free();

			// V_4f_b_bn_scale <~~ X8577
			float x1908, x1909;
			float x1910;
			float x1911;
			x1910 = 1;
			x1911 = lrn_rate;
			x1908 = x1910 * x1911;
			x1909 = momentum;
			JCudaTensor x1912;
			x1912 = x1884;
			x1907.update(x1912, x1908, x1909);

			// Dealloc(X8577)
			JCudaTensor x1913;
			x1913 = x1884;
			x1913.free();

			// 4f_b_bn_bias <~~ V_4f_b_bn_bias
			float x1914, x1915;
			x1914 = 1;
			float x1916;
			float x1917;
			x1916 = 1;
			float x1918;
			float x1919;
			float x1920;
			float x1921;
			x1920 = 1;
			x1921 = decay;
			x1918 = x1920 * x1921;
			float x1922;
			float x1923;
			x1922 = 1;
			x1923 = lrn_rate;
			x1919 = x1922 * x1923;
			x1917 = x1918 * x1919;
			x1915 = x1916 + x1917;
			JCudaTensor x1924;
			x1924 = x1889;
			x725.update(x1924, x1914, x1915);

			// 4f_b_cv_W <~~ V_4f_b_cv_W
			float x1925, x1926;
			x1925 = 1;
			float x1927;
			float x1928;
			x1927 = 1;
			float x1929;
			float x1930;
			float x1931;
			float x1932;
			x1931 = 1;
			x1932 = decay;
			x1929 = x1931 * x1932;
			float x1933;
			float x1934;
			x1933 = 1;
			x1934 = lrn_rate;
			x1930 = x1933 * x1934;
			x1928 = x1929 * x1930;
			x1926 = x1927 + x1928;
			JCudaTensor x1935;
			x1935 = x1899;
			x718.update(x1935, x1925, x1926);

			// 4f_b_bn_scale <~~ V_4f_b_bn_scale
			float x1936, x1937;
			x1936 = 1;
			float x1938;
			float x1939;
			x1938 = 1;
			float x1940;
			float x1941;
			float x1942;
			float x1943;
			x1942 = 1;
			x1943 = decay;
			x1940 = x1942 * x1943;
			float x1944;
			float x1945;
			x1944 = 1;
			x1945 = lrn_rate;
			x1941 = x1944 * x1945;
			x1939 = x1940 * x1941;
			x1937 = x1938 + x1939;
			JCudaTensor x1946;
			x1946 = x1907;
			x724.update(x1946, x1936, x1937);

			// val X3842 = X3838 * d_ReLU()(X8762)/d_X8761
			JCudaTensor x1947;
			JCudaTensor x1948, x1949;
			x1948 = x1896;
			x1949 = x712;
			x1947 = x457.backward(x1948, x1949);

			// Dealloc(X8762)
			JCudaTensor x1950;
			x1950 = x712;
			x1950.free();

			// val X3843 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_X8760
			JCudaTensor x1951;
			JCudaTensor x1952, x1953, x1954;
			x1952 = x1947;
			x1953 = x699;
			x1954 = x709;
			JCudaTensor[] x1955 = x711.backward(x1952,x1953,x1954);
			x1951 = x1955[0];

			// val X8573 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_4f_a_bn_bias
			JCudaTensor x1956;
			x1956 = x1955[2];

			// val X8574 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_4f_a_bn_scale
			JCudaTensor x1960;
			x1960 = x1955[1];

			// Dealloc(X8760)
			JCudaTensor x1964;
			x1964 = x699;
			x1964.free();

			// val X3844 = X3843 * d_Convolv(1,0)(4f_a_cv_W)/d_X8759
			JCudaTensor x1965;
			JCudaTensor x1966, x1967;
			x1966 = x1951;
			x1967 = x703;
			x1965 = x504.backward_data(x1966, x1967);

			// V_4f_a_cv_W <~~ X3843 * d_Convolv(1,0)(X8759)/d_4f_a_cv_W
			float x1969, x1970;
			float x1971;
			float x1972;
			x1971 = 1;
			x1972 = lrn_rate;
			x1969 = x1971 * x1972;
			x1970 = momentum;
			JCudaTensor x1973, x1974;
			x1973 = x1951;
			x1974 = x697;
			x504.backward_filter(x1973, x1974, x1968, x1969, x1970);

			// Dealloc(X3843)
			JCudaTensor x1975;
			x1975 = x1951;
			x1975.free();

			// V_4f_a_bn_bias <~~ X8573
			float x1977, x1978;
			float x1979;
			float x1980;
			x1979 = 1;
			x1980 = lrn_rate;
			x1977 = x1979 * x1980;
			x1978 = momentum;
			JCudaTensor x1981;
			x1981 = x1956;
			x1976.update(x1981, x1977, x1978);

			// Dealloc(X8573)
			JCudaTensor x1982;
			x1982 = x1956;
			x1982.free();

			// V_4f_a_bn_scale <~~ X8574
			float x1984, x1985;
			float x1986;
			float x1987;
			x1986 = 1;
			x1987 = lrn_rate;
			x1984 = x1986 * x1987;
			x1985 = momentum;
			JCudaTensor x1988;
			x1988 = x1960;
			x1983.update(x1988, x1984, x1985);

			// Dealloc(X8574)
			JCudaTensor x1989;
			x1989 = x1960;
			x1989.free();

			// 4f_a_cv_W <~~ V_4f_a_cv_W
			float x1990, x1991;
			x1990 = 1;
			float x1992;
			float x1993;
			x1992 = 1;
			float x1994;
			float x1995;
			float x1996;
			float x1997;
			x1996 = 1;
			x1997 = decay;
			x1994 = x1996 * x1997;
			float x1998;
			float x1999;
			x1998 = 1;
			x1999 = lrn_rate;
			x1995 = x1998 * x1999;
			x1993 = x1994 * x1995;
			x1991 = x1992 + x1993;
			JCudaTensor x2000;
			x2000 = x1968;
			x703.update(x2000, x1990, x1991);

			// 4f_a_bn_bias <~~ V_4f_a_bn_bias
			float x2001, x2002;
			x2001 = 1;
			float x2003;
			float x2004;
			x2003 = 1;
			float x2005;
			float x2006;
			float x2007;
			float x2008;
			x2007 = 1;
			x2008 = decay;
			x2005 = x2007 * x2008;
			float x2009;
			float x2010;
			x2009 = 1;
			x2010 = lrn_rate;
			x2006 = x2009 * x2010;
			x2004 = x2005 * x2006;
			x2002 = x2003 + x2004;
			JCudaTensor x2011;
			x2011 = x1976;
			x710.update(x2011, x2001, x2002);

			// 4f_a_bn_scale <~~ V_4f_a_bn_scale
			float x2012, x2013;
			x2012 = 1;
			float x2014;
			float x2015;
			x2014 = 1;
			float x2016;
			float x2017;
			float x2018;
			float x2019;
			x2018 = 1;
			x2019 = decay;
			x2016 = x2018 * x2019;
			float x2020;
			float x2021;
			x2020 = 1;
			x2021 = lrn_rate;
			x2017 = x2020 * x2021;
			x2015 = x2016 * x2017;
			x2013 = x2014 + x2015;
			JCudaTensor x2022;
			x2022 = x1983;
			x709.update(x2022, x2012, x2013);

			// val X3845 = (X3844 + X3820)
			JCudaTensor x2023;
			JCudaTensor x2024, x2025;
			x2024 = x1965;
			x2025 = x1791;
			x2023 = x2024.plus_i(x2025);

			// Dealloc(X3820)
			JCudaTensor x2026;
			x2026 = x1791;
			x2026.free();

			// val X3857 = X3845 * d_ReLU()(X8759)/d_X8758
			JCudaTensor x2027;
			JCudaTensor x2028, x2029;
			x2028 = x2023;
			x2029 = x697;
			x2027 = x490.backward(x2028, x2029);

			// Dealloc(X8759)
			JCudaTensor x2030;
			x2030 = x697;
			x2030.free();

			// val X3867 = X3857.copy * d_ReLU()(X8757)/d_X8756
			JCudaTensor x2031;
			JCudaTensor x2032, x2033;
			x2032 = x2027;
			x2032 = x2032.clone();
			x2033 = x692;
			x2031 = x490.backward(x2032, x2033);

			// Dealloc(X8757)
			JCudaTensor x2034;
			x2034 = x692;
			x2034.free();

			// val X3868 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_X8755
			JCudaTensor x2035;
			JCudaTensor x2036, x2037, x2038;
			x2036 = x2031;
			x2037 = x679;
			x2038 = x689;
			JCudaTensor[] x2039 = x691.backward(x2036,x2037,x2038);
			x2035 = x2039[0];

			// val X8570 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_4e_c_bn_bias
			JCudaTensor x2040;
			x2040 = x2039[2];

			// val X8571 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_4e_c_bn_scale
			JCudaTensor x2044;
			x2044 = x2039[1];

			// Dealloc(X8755)
			JCudaTensor x2048;
			x2048 = x679;
			x2048.free();

			// V_4e_c_bn_bias <~~ X8570
			float x2050, x2051;
			float x2052;
			float x2053;
			x2052 = 1;
			x2053 = lrn_rate;
			x2050 = x2052 * x2053;
			x2051 = momentum;
			JCudaTensor x2054;
			x2054 = x2040;
			x2049.update(x2054, x2050, x2051);

			// Dealloc(X8570)
			JCudaTensor x2055;
			x2055 = x2040;
			x2055.free();

			// V_4e_c_bn_scale <~~ X8571
			float x2057, x2058;
			float x2059;
			float x2060;
			x2059 = 1;
			x2060 = lrn_rate;
			x2057 = x2059 * x2060;
			x2058 = momentum;
			JCudaTensor x2061;
			x2061 = x2044;
			x2056.update(x2061, x2057, x2058);

			// Dealloc(X8571)
			JCudaTensor x2062;
			x2062 = x2044;
			x2062.free();

			// val X3869 = X3868 * d_Convolv(1,0)(4e_c_cv_W)/d_X8754
			JCudaTensor x2063;
			JCudaTensor x2064, x2065;
			x2064 = x2035;
			x2065 = x683;
			x2063 = x480.backward_data(x2064, x2065);

			// V_4e_c_cv_W <~~ X3868 * d_Convolv(1,0)(X8754)/d_4e_c_cv_W
			float x2067, x2068;
			float x2069;
			float x2070;
			x2069 = 1;
			x2070 = lrn_rate;
			x2067 = x2069 * x2070;
			x2068 = momentum;
			JCudaTensor x2071, x2072;
			x2071 = x2035;
			x2072 = x677;
			x480.backward_filter(x2071, x2072, x2066, x2067, x2068);

			// Dealloc(X3868)
			JCudaTensor x2073;
			x2073 = x2035;
			x2073.free();

			// 4e_c_bn_bias <~~ V_4e_c_bn_bias
			float x2074, x2075;
			x2074 = 1;
			float x2076;
			float x2077;
			x2076 = 1;
			float x2078;
			float x2079;
			float x2080;
			float x2081;
			x2080 = 1;
			x2081 = decay;
			x2078 = x2080 * x2081;
			float x2082;
			float x2083;
			x2082 = 1;
			x2083 = lrn_rate;
			x2079 = x2082 * x2083;
			x2077 = x2078 * x2079;
			x2075 = x2076 + x2077;
			JCudaTensor x2084;
			x2084 = x2049;
			x690.update(x2084, x2074, x2075);

			// 4e_c_bn_scale <~~ V_4e_c_bn_scale
			float x2085, x2086;
			x2085 = 1;
			float x2087;
			float x2088;
			x2087 = 1;
			float x2089;
			float x2090;
			float x2091;
			float x2092;
			x2091 = 1;
			x2092 = decay;
			x2089 = x2091 * x2092;
			float x2093;
			float x2094;
			x2093 = 1;
			x2094 = lrn_rate;
			x2090 = x2093 * x2094;
			x2088 = x2089 * x2090;
			x2086 = x2087 + x2088;
			JCudaTensor x2095;
			x2095 = x2056;
			x689.update(x2095, x2085, x2086);

			// 4e_c_cv_W <~~ V_4e_c_cv_W
			float x2096, x2097;
			x2096 = 1;
			float x2098;
			float x2099;
			x2098 = 1;
			float x2100;
			float x2101;
			float x2102;
			float x2103;
			x2102 = 1;
			x2103 = decay;
			x2100 = x2102 * x2103;
			float x2104;
			float x2105;
			x2104 = 1;
			x2105 = lrn_rate;
			x2101 = x2104 * x2105;
			x2099 = x2100 * x2101;
			x2097 = x2098 + x2099;
			JCudaTensor x2106;
			x2106 = x2066;
			x683.update(x2106, x2096, x2097);

			// val X3873 = X3869 * d_ReLU()(X8754)/d_X8753
			JCudaTensor x2107;
			JCudaTensor x2108, x2109;
			x2108 = x2063;
			x2109 = x677;
			x2107 = x457.backward(x2108, x2109);

			// Dealloc(X8754)
			JCudaTensor x2110;
			x2110 = x677;
			x2110.free();

			// val X8568 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_4e_b_bn_scale
			JCudaTensor x2111;
			JCudaTensor x2112, x2113, x2114;
			x2112 = x2107;
			x2113 = x664;
			x2114 = x674;
			JCudaTensor[] x2115 = x676.backward(x2112,x2113,x2114);
			x2111 = x2115[1];

			// val X8567 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_4e_b_bn_bias
			JCudaTensor x2116;
			x2116 = x2115[2];

			// val X3874 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_X8752
			JCudaTensor x2120;
			x2120 = x2115[0];

			// Dealloc(X8752)
			JCudaTensor x2124;
			x2124 = x664;
			x2124.free();

			// val X3875 = X3874 * d_Convolv(1,1)(4e_b_cv_W)/d_X8751
			JCudaTensor x2125;
			JCudaTensor x2126, x2127;
			x2126 = x2120;
			x2127 = x668;
			x2125 = x464.backward_data(x2126, x2127);

			// V_4e_b_bn_bias <~~ X8567
			float x2129, x2130;
			float x2131;
			float x2132;
			x2131 = 1;
			x2132 = lrn_rate;
			x2129 = x2131 * x2132;
			x2130 = momentum;
			JCudaTensor x2133;
			x2133 = x2116;
			x2128.update(x2133, x2129, x2130);

			// Dealloc(X8567)
			JCudaTensor x2134;
			x2134 = x2116;
			x2134.free();

			// V_4e_b_cv_W <~~ X3874 * d_Convolv(1,1)(X8751)/d_4e_b_cv_W
			float x2136, x2137;
			float x2138;
			float x2139;
			x2138 = 1;
			x2139 = lrn_rate;
			x2136 = x2138 * x2139;
			x2137 = momentum;
			JCudaTensor x2140, x2141;
			x2140 = x2120;
			x2141 = x662;
			x464.backward_filter(x2140, x2141, x2135, x2136, x2137);

			// Dealloc(X3874)
			JCudaTensor x2142;
			x2142 = x2120;
			x2142.free();

			// V_4e_b_bn_scale <~~ X8568
			float x2144, x2145;
			float x2146;
			float x2147;
			x2146 = 1;
			x2147 = lrn_rate;
			x2144 = x2146 * x2147;
			x2145 = momentum;
			JCudaTensor x2148;
			x2148 = x2111;
			x2143.update(x2148, x2144, x2145);

			// Dealloc(X8568)
			JCudaTensor x2149;
			x2149 = x2111;
			x2149.free();

			// 4e_b_bn_bias <~~ V_4e_b_bn_bias
			float x2150, x2151;
			x2150 = 1;
			float x2152;
			float x2153;
			x2152 = 1;
			float x2154;
			float x2155;
			float x2156;
			float x2157;
			x2156 = 1;
			x2157 = decay;
			x2154 = x2156 * x2157;
			float x2158;
			float x2159;
			x2158 = 1;
			x2159 = lrn_rate;
			x2155 = x2158 * x2159;
			x2153 = x2154 * x2155;
			x2151 = x2152 + x2153;
			JCudaTensor x2160;
			x2160 = x2128;
			x675.update(x2160, x2150, x2151);

			// 4e_b_cv_W <~~ V_4e_b_cv_W
			float x2161, x2162;
			x2161 = 1;
			float x2163;
			float x2164;
			x2163 = 1;
			float x2165;
			float x2166;
			float x2167;
			float x2168;
			x2167 = 1;
			x2168 = decay;
			x2165 = x2167 * x2168;
			float x2169;
			float x2170;
			x2169 = 1;
			x2170 = lrn_rate;
			x2166 = x2169 * x2170;
			x2164 = x2165 * x2166;
			x2162 = x2163 + x2164;
			JCudaTensor x2171;
			x2171 = x2135;
			x668.update(x2171, x2161, x2162);

			// 4e_b_bn_scale <~~ V_4e_b_bn_scale
			float x2172, x2173;
			x2172 = 1;
			float x2174;
			float x2175;
			x2174 = 1;
			float x2176;
			float x2177;
			float x2178;
			float x2179;
			x2178 = 1;
			x2179 = decay;
			x2176 = x2178 * x2179;
			float x2180;
			float x2181;
			x2180 = 1;
			x2181 = lrn_rate;
			x2177 = x2180 * x2181;
			x2175 = x2176 * x2177;
			x2173 = x2174 + x2175;
			JCudaTensor x2182;
			x2182 = x2143;
			x674.update(x2182, x2172, x2173);

			// val X3879 = X3875 * d_ReLU()(X8751)/d_X8750
			JCudaTensor x2183;
			JCudaTensor x2184, x2185;
			x2184 = x2125;
			x2185 = x662;
			x2183 = x457.backward(x2184, x2185);

			// Dealloc(X8751)
			JCudaTensor x2186;
			x2186 = x662;
			x2186.free();

			// val X3880 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_X8749
			JCudaTensor x2187;
			JCudaTensor x2188, x2189, x2190;
			x2188 = x2183;
			x2189 = x649;
			x2190 = x659;
			JCudaTensor[] x2191 = x661.backward(x2188,x2189,x2190);
			x2187 = x2191[0];

			// val X8564 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_4e_a_bn_bias
			JCudaTensor x2192;
			x2192 = x2191[2];

			// val X8565 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_4e_a_bn_scale
			JCudaTensor x2196;
			x2196 = x2191[1];

			// Dealloc(X8749)
			JCudaTensor x2200;
			x2200 = x649;
			x2200.free();

			// val X3881 = X3880 * d_Convolv(1,0)(4e_a_cv_W)/d_X8748
			JCudaTensor x2201;
			JCudaTensor x2202, x2203;
			x2202 = x2187;
			x2203 = x653;
			x2201 = x504.backward_data(x2202, x2203);

			// V_4e_a_bn_bias <~~ X8564
			float x2205, x2206;
			float x2207;
			float x2208;
			x2207 = 1;
			x2208 = lrn_rate;
			x2205 = x2207 * x2208;
			x2206 = momentum;
			JCudaTensor x2209;
			x2209 = x2192;
			x2204.update(x2209, x2205, x2206);

			// Dealloc(X8564)
			JCudaTensor x2210;
			x2210 = x2192;
			x2210.free();

			// V_4e_a_bn_scale <~~ X8565
			float x2212, x2213;
			float x2214;
			float x2215;
			x2214 = 1;
			x2215 = lrn_rate;
			x2212 = x2214 * x2215;
			x2213 = momentum;
			JCudaTensor x2216;
			x2216 = x2196;
			x2211.update(x2216, x2212, x2213);

			// Dealloc(X8565)
			JCudaTensor x2217;
			x2217 = x2196;
			x2217.free();

			// V_4e_a_cv_W <~~ X3880 * d_Convolv(1,0)(X8748)/d_4e_a_cv_W
			float x2219, x2220;
			float x2221;
			float x2222;
			x2221 = 1;
			x2222 = lrn_rate;
			x2219 = x2221 * x2222;
			x2220 = momentum;
			JCudaTensor x2223, x2224;
			x2223 = x2187;
			x2224 = x647;
			x504.backward_filter(x2223, x2224, x2218, x2219, x2220);

			// Dealloc(X3880)
			JCudaTensor x2225;
			x2225 = x2187;
			x2225.free();

			// 4e_a_bn_bias <~~ V_4e_a_bn_bias
			float x2226, x2227;
			x2226 = 1;
			float x2228;
			float x2229;
			x2228 = 1;
			float x2230;
			float x2231;
			float x2232;
			float x2233;
			x2232 = 1;
			x2233 = decay;
			x2230 = x2232 * x2233;
			float x2234;
			float x2235;
			x2234 = 1;
			x2235 = lrn_rate;
			x2231 = x2234 * x2235;
			x2229 = x2230 * x2231;
			x2227 = x2228 + x2229;
			JCudaTensor x2236;
			x2236 = x2204;
			x660.update(x2236, x2226, x2227);

			// 4e_a_bn_scale <~~ V_4e_a_bn_scale
			float x2237, x2238;
			x2237 = 1;
			float x2239;
			float x2240;
			x2239 = 1;
			float x2241;
			float x2242;
			float x2243;
			float x2244;
			x2243 = 1;
			x2244 = decay;
			x2241 = x2243 * x2244;
			float x2245;
			float x2246;
			x2245 = 1;
			x2246 = lrn_rate;
			x2242 = x2245 * x2246;
			x2240 = x2241 * x2242;
			x2238 = x2239 + x2240;
			JCudaTensor x2247;
			x2247 = x2211;
			x659.update(x2247, x2237, x2238);

			// 4e_a_cv_W <~~ V_4e_a_cv_W
			float x2248, x2249;
			x2248 = 1;
			float x2250;
			float x2251;
			x2250 = 1;
			float x2252;
			float x2253;
			float x2254;
			float x2255;
			x2254 = 1;
			x2255 = decay;
			x2252 = x2254 * x2255;
			float x2256;
			float x2257;
			x2256 = 1;
			x2257 = lrn_rate;
			x2253 = x2256 * x2257;
			x2251 = x2252 * x2253;
			x2249 = x2250 + x2251;
			JCudaTensor x2258;
			x2258 = x2218;
			x653.update(x2258, x2248, x2249);

			// val X3882 = (X3881 + X3857)
			JCudaTensor x2259;
			JCudaTensor x2260, x2261;
			x2260 = x2201;
			x2261 = x2027;
			x2259 = x2260.plus_i(x2261);

			// Dealloc(X3857)
			JCudaTensor x2262;
			x2262 = x2027;
			x2262.free();

			// val X3894 = X3882 * d_ReLU()(X8748)/d_X8747
			JCudaTensor x2263;
			JCudaTensor x2264, x2265;
			x2264 = x2259;
			x2265 = x647;
			x2263 = x490.backward(x2264, x2265);

			// Dealloc(X8748)
			JCudaTensor x2266;
			x2266 = x647;
			x2266.free();

			// val X3904 = X3894.copy * d_ReLU()(X8746)/d_X8745
			JCudaTensor x2267;
			JCudaTensor x2268, x2269;
			x2268 = x2263;
			x2268 = x2268.clone();
			x2269 = x642;
			x2267 = x490.backward(x2268, x2269);

			// Dealloc(X8746)
			JCudaTensor x2270;
			x2270 = x642;
			x2270.free();

			// val X3905 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_X8744
			JCudaTensor x2271;
			JCudaTensor x2272, x2273, x2274;
			x2272 = x2267;
			x2273 = x629;
			x2274 = x639;
			JCudaTensor[] x2275 = x641.backward(x2272,x2273,x2274);
			x2271 = x2275[0];

			// val X8561 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_4d_c_bn_bias
			JCudaTensor x2276;
			x2276 = x2275[2];

			// val X8562 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_4d_c_bn_scale
			JCudaTensor x2280;
			x2280 = x2275[1];

			// Dealloc(X8744)
			JCudaTensor x2284;
			x2284 = x629;
			x2284.free();

			// V_4d_c_bn_scale <~~ X8562
			float x2286, x2287;
			float x2288;
			float x2289;
			x2288 = 1;
			x2289 = lrn_rate;
			x2286 = x2288 * x2289;
			x2287 = momentum;
			JCudaTensor x2290;
			x2290 = x2280;
			x2285.update(x2290, x2286, x2287);

			// Dealloc(X8562)
			JCudaTensor x2291;
			x2291 = x2280;
			x2291.free();

			// val X3906 = X3905 * d_Convolv(1,0)(4d_c_cv_W)/d_X8743
			JCudaTensor x2292;
			JCudaTensor x2293, x2294;
			x2293 = x2271;
			x2294 = x633;
			x2292 = x480.backward_data(x2293, x2294);

			// V_4d_c_cv_W <~~ X3905 * d_Convolv(1,0)(X8743)/d_4d_c_cv_W
			float x2296, x2297;
			float x2298;
			float x2299;
			x2298 = 1;
			x2299 = lrn_rate;
			x2296 = x2298 * x2299;
			x2297 = momentum;
			JCudaTensor x2300, x2301;
			x2300 = x2271;
			x2301 = x627;
			x480.backward_filter(x2300, x2301, x2295, x2296, x2297);

			// Dealloc(X3905)
			JCudaTensor x2302;
			x2302 = x2271;
			x2302.free();

			// V_4d_c_bn_bias <~~ X8561
			float x2304, x2305;
			float x2306;
			float x2307;
			x2306 = 1;
			x2307 = lrn_rate;
			x2304 = x2306 * x2307;
			x2305 = momentum;
			JCudaTensor x2308;
			x2308 = x2276;
			x2303.update(x2308, x2304, x2305);

			// Dealloc(X8561)
			JCudaTensor x2309;
			x2309 = x2276;
			x2309.free();

			// 4d_c_bn_scale <~~ V_4d_c_bn_scale
			float x2310, x2311;
			x2310 = 1;
			float x2312;
			float x2313;
			x2312 = 1;
			float x2314;
			float x2315;
			float x2316;
			float x2317;
			x2316 = 1;
			x2317 = decay;
			x2314 = x2316 * x2317;
			float x2318;
			float x2319;
			x2318 = 1;
			x2319 = lrn_rate;
			x2315 = x2318 * x2319;
			x2313 = x2314 * x2315;
			x2311 = x2312 + x2313;
			JCudaTensor x2320;
			x2320 = x2285;
			x639.update(x2320, x2310, x2311);

			// 4d_c_cv_W <~~ V_4d_c_cv_W
			float x2321, x2322;
			x2321 = 1;
			float x2323;
			float x2324;
			x2323 = 1;
			float x2325;
			float x2326;
			float x2327;
			float x2328;
			x2327 = 1;
			x2328 = decay;
			x2325 = x2327 * x2328;
			float x2329;
			float x2330;
			x2329 = 1;
			x2330 = lrn_rate;
			x2326 = x2329 * x2330;
			x2324 = x2325 * x2326;
			x2322 = x2323 + x2324;
			JCudaTensor x2331;
			x2331 = x2295;
			x633.update(x2331, x2321, x2322);

			// 4d_c_bn_bias <~~ V_4d_c_bn_bias
			float x2332, x2333;
			x2332 = 1;
			float x2334;
			float x2335;
			x2334 = 1;
			float x2336;
			float x2337;
			float x2338;
			float x2339;
			x2338 = 1;
			x2339 = decay;
			x2336 = x2338 * x2339;
			float x2340;
			float x2341;
			x2340 = 1;
			x2341 = lrn_rate;
			x2337 = x2340 * x2341;
			x2335 = x2336 * x2337;
			x2333 = x2334 + x2335;
			JCudaTensor x2342;
			x2342 = x2303;
			x640.update(x2342, x2332, x2333);

			// val X3910 = X3906 * d_ReLU()(X8743)/d_X8742
			JCudaTensor x2343;
			JCudaTensor x2344, x2345;
			x2344 = x2292;
			x2345 = x627;
			x2343 = x457.backward(x2344, x2345);

			// Dealloc(X8743)
			JCudaTensor x2346;
			x2346 = x627;
			x2346.free();

			// val X8559 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_4d_b_bn_scale
			JCudaTensor x2347;
			JCudaTensor x2348, x2349, x2350;
			x2348 = x2343;
			x2349 = x614;
			x2350 = x624;
			JCudaTensor[] x2351 = x626.backward(x2348,x2349,x2350);
			x2347 = x2351[1];

			// val X3911 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_X8741
			JCudaTensor x2352;
			x2352 = x2351[0];

			// val X8558 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_4d_b_bn_bias
			JCudaTensor x2356;
			x2356 = x2351[2];

			// Dealloc(X8741)
			JCudaTensor x2360;
			x2360 = x614;
			x2360.free();

			// V_4d_b_bn_bias <~~ X8558
			float x2362, x2363;
			float x2364;
			float x2365;
			x2364 = 1;
			x2365 = lrn_rate;
			x2362 = x2364 * x2365;
			x2363 = momentum;
			JCudaTensor x2366;
			x2366 = x2356;
			x2361.update(x2366, x2362, x2363);

			// Dealloc(X8558)
			JCudaTensor x2367;
			x2367 = x2356;
			x2367.free();

			// val X3912 = X3911 * d_Convolv(1,1)(4d_b_cv_W)/d_X8740
			JCudaTensor x2368;
			JCudaTensor x2369, x2370;
			x2369 = x2352;
			x2370 = x618;
			x2368 = x464.backward_data(x2369, x2370);

			// V_4d_b_cv_W <~~ X3911 * d_Convolv(1,1)(X8740)/d_4d_b_cv_W
			float x2372, x2373;
			float x2374;
			float x2375;
			x2374 = 1;
			x2375 = lrn_rate;
			x2372 = x2374 * x2375;
			x2373 = momentum;
			JCudaTensor x2376, x2377;
			x2376 = x2352;
			x2377 = x612;
			x464.backward_filter(x2376, x2377, x2371, x2372, x2373);

			// Dealloc(X3911)
			JCudaTensor x2378;
			x2378 = x2352;
			x2378.free();

			// V_4d_b_bn_scale <~~ X8559
			float x2380, x2381;
			float x2382;
			float x2383;
			x2382 = 1;
			x2383 = lrn_rate;
			x2380 = x2382 * x2383;
			x2381 = momentum;
			JCudaTensor x2384;
			x2384 = x2347;
			x2379.update(x2384, x2380, x2381);

			// Dealloc(X8559)
			JCudaTensor x2385;
			x2385 = x2347;
			x2385.free();

			// 4d_b_bn_bias <~~ V_4d_b_bn_bias
			float x2386, x2387;
			x2386 = 1;
			float x2388;
			float x2389;
			x2388 = 1;
			float x2390;
			float x2391;
			float x2392;
			float x2393;
			x2392 = 1;
			x2393 = decay;
			x2390 = x2392 * x2393;
			float x2394;
			float x2395;
			x2394 = 1;
			x2395 = lrn_rate;
			x2391 = x2394 * x2395;
			x2389 = x2390 * x2391;
			x2387 = x2388 + x2389;
			JCudaTensor x2396;
			x2396 = x2361;
			x625.update(x2396, x2386, x2387);

			// 4d_b_cv_W <~~ V_4d_b_cv_W
			float x2397, x2398;
			x2397 = 1;
			float x2399;
			float x2400;
			x2399 = 1;
			float x2401;
			float x2402;
			float x2403;
			float x2404;
			x2403 = 1;
			x2404 = decay;
			x2401 = x2403 * x2404;
			float x2405;
			float x2406;
			x2405 = 1;
			x2406 = lrn_rate;
			x2402 = x2405 * x2406;
			x2400 = x2401 * x2402;
			x2398 = x2399 + x2400;
			JCudaTensor x2407;
			x2407 = x2371;
			x618.update(x2407, x2397, x2398);

			// 4d_b_bn_scale <~~ V_4d_b_bn_scale
			float x2408, x2409;
			x2408 = 1;
			float x2410;
			float x2411;
			x2410 = 1;
			float x2412;
			float x2413;
			float x2414;
			float x2415;
			x2414 = 1;
			x2415 = decay;
			x2412 = x2414 * x2415;
			float x2416;
			float x2417;
			x2416 = 1;
			x2417 = lrn_rate;
			x2413 = x2416 * x2417;
			x2411 = x2412 * x2413;
			x2409 = x2410 + x2411;
			JCudaTensor x2418;
			x2418 = x2379;
			x624.update(x2418, x2408, x2409);

			// val X3916 = X3912 * d_ReLU()(X8740)/d_X8739
			JCudaTensor x2419;
			JCudaTensor x2420, x2421;
			x2420 = x2368;
			x2421 = x612;
			x2419 = x457.backward(x2420, x2421);

			// Dealloc(X8740)
			JCudaTensor x2422;
			x2422 = x612;
			x2422.free();

			// val X3917 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_X8738
			JCudaTensor x2423;
			JCudaTensor x2424, x2425, x2426;
			x2424 = x2419;
			x2425 = x599;
			x2426 = x609;
			JCudaTensor[] x2427 = x611.backward(x2424,x2425,x2426);
			x2423 = x2427[0];

			// val X8555 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_4d_a_bn_bias
			JCudaTensor x2428;
			x2428 = x2427[2];

			// val X8556 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_4d_a_bn_scale
			JCudaTensor x2432;
			x2432 = x2427[1];

			// Dealloc(X8738)
			JCudaTensor x2436;
			x2436 = x599;
			x2436.free();

			// V_4d_a_bn_bias <~~ X8555
			float x2438, x2439;
			float x2440;
			float x2441;
			x2440 = 1;
			x2441 = lrn_rate;
			x2438 = x2440 * x2441;
			x2439 = momentum;
			JCudaTensor x2442;
			x2442 = x2428;
			x2437.update(x2442, x2438, x2439);

			// Dealloc(X8555)
			JCudaTensor x2443;
			x2443 = x2428;
			x2443.free();

			// val X3918 = X3917 * d_Convolv(1,0)(4d_a_cv_W)/d_X8737
			JCudaTensor x2444;
			JCudaTensor x2445, x2446;
			x2445 = x2423;
			x2446 = x603;
			x2444 = x504.backward_data(x2445, x2446);

			// V_4d_a_bn_scale <~~ X8556
			float x2448, x2449;
			float x2450;
			float x2451;
			x2450 = 1;
			x2451 = lrn_rate;
			x2448 = x2450 * x2451;
			x2449 = momentum;
			JCudaTensor x2452;
			x2452 = x2432;
			x2447.update(x2452, x2448, x2449);

			// Dealloc(X8556)
			JCudaTensor x2453;
			x2453 = x2432;
			x2453.free();

			// V_4d_a_cv_W <~~ X3917 * d_Convolv(1,0)(X8737)/d_4d_a_cv_W
			float x2455, x2456;
			float x2457;
			float x2458;
			x2457 = 1;
			x2458 = lrn_rate;
			x2455 = x2457 * x2458;
			x2456 = momentum;
			JCudaTensor x2459, x2460;
			x2459 = x2423;
			x2460 = x597;
			x504.backward_filter(x2459, x2460, x2454, x2455, x2456);

			// Dealloc(X3917)
			JCudaTensor x2461;
			x2461 = x2423;
			x2461.free();

			// 4d_a_bn_bias <~~ V_4d_a_bn_bias
			float x2462, x2463;
			x2462 = 1;
			float x2464;
			float x2465;
			x2464 = 1;
			float x2466;
			float x2467;
			float x2468;
			float x2469;
			x2468 = 1;
			x2469 = decay;
			x2466 = x2468 * x2469;
			float x2470;
			float x2471;
			x2470 = 1;
			x2471 = lrn_rate;
			x2467 = x2470 * x2471;
			x2465 = x2466 * x2467;
			x2463 = x2464 + x2465;
			JCudaTensor x2472;
			x2472 = x2437;
			x610.update(x2472, x2462, x2463);

			// 4d_a_bn_scale <~~ V_4d_a_bn_scale
			float x2473, x2474;
			x2473 = 1;
			float x2475;
			float x2476;
			x2475 = 1;
			float x2477;
			float x2478;
			float x2479;
			float x2480;
			x2479 = 1;
			x2480 = decay;
			x2477 = x2479 * x2480;
			float x2481;
			float x2482;
			x2481 = 1;
			x2482 = lrn_rate;
			x2478 = x2481 * x2482;
			x2476 = x2477 * x2478;
			x2474 = x2475 + x2476;
			JCudaTensor x2483;
			x2483 = x2447;
			x609.update(x2483, x2473, x2474);

			// 4d_a_cv_W <~~ V_4d_a_cv_W
			float x2484, x2485;
			x2484 = 1;
			float x2486;
			float x2487;
			x2486 = 1;
			float x2488;
			float x2489;
			float x2490;
			float x2491;
			x2490 = 1;
			x2491 = decay;
			x2488 = x2490 * x2491;
			float x2492;
			float x2493;
			x2492 = 1;
			x2493 = lrn_rate;
			x2489 = x2492 * x2493;
			x2487 = x2488 * x2489;
			x2485 = x2486 + x2487;
			JCudaTensor x2494;
			x2494 = x2454;
			x603.update(x2494, x2484, x2485);

			// val X3919 = (X3918 + X3894)
			JCudaTensor x2495;
			JCudaTensor x2496, x2497;
			x2496 = x2444;
			x2497 = x2263;
			x2495 = x2496.plus_i(x2497);

			// Dealloc(X3894)
			JCudaTensor x2498;
			x2498 = x2263;
			x2498.free();

			// val X3931 = X3919 * d_ReLU()(X8737)/d_X8736
			JCudaTensor x2499;
			JCudaTensor x2500, x2501;
			x2500 = x2495;
			x2501 = x597;
			x2499 = x490.backward(x2500, x2501);

			// Dealloc(X8737)
			JCudaTensor x2502;
			x2502 = x597;
			x2502.free();

			// val X3941 = X3931.copy * d_ReLU()(X8735)/d_X8734
			JCudaTensor x2503;
			JCudaTensor x2504, x2505;
			x2504 = x2499;
			x2504 = x2504.clone();
			x2505 = x592;
			x2503 = x490.backward(x2504, x2505);

			// Dealloc(X8735)
			JCudaTensor x2506;
			x2506 = x592;
			x2506.free();

			// val X3942 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_X8733
			JCudaTensor x2507;
			JCudaTensor x2508, x2509, x2510;
			x2508 = x2503;
			x2509 = x579;
			x2510 = x589;
			JCudaTensor[] x2511 = x591.backward(x2508,x2509,x2510);
			x2507 = x2511[0];

			// val X8552 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_4c_c_bn_bias
			JCudaTensor x2512;
			x2512 = x2511[2];

			// val X8553 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_4c_c_bn_scale
			JCudaTensor x2516;
			x2516 = x2511[1];

			// Dealloc(X8733)
			JCudaTensor x2520;
			x2520 = x579;
			x2520.free();

			// V_4c_c_bn_scale <~~ X8553
			float x2522, x2523;
			float x2524;
			float x2525;
			x2524 = 1;
			x2525 = lrn_rate;
			x2522 = x2524 * x2525;
			x2523 = momentum;
			JCudaTensor x2526;
			x2526 = x2516;
			x2521.update(x2526, x2522, x2523);

			// Dealloc(X8553)
			JCudaTensor x2527;
			x2527 = x2516;
			x2527.free();

			// val X3943 = X3942 * d_Convolv(1,0)(4c_c_cv_W)/d_X8732
			JCudaTensor x2528;
			JCudaTensor x2529, x2530;
			x2529 = x2507;
			x2530 = x583;
			x2528 = x480.backward_data(x2529, x2530);

			// V_4c_c_cv_W <~~ X3942 * d_Convolv(1,0)(X8732)/d_4c_c_cv_W
			float x2532, x2533;
			float x2534;
			float x2535;
			x2534 = 1;
			x2535 = lrn_rate;
			x2532 = x2534 * x2535;
			x2533 = momentum;
			JCudaTensor x2536, x2537;
			x2536 = x2507;
			x2537 = x577;
			x480.backward_filter(x2536, x2537, x2531, x2532, x2533);

			// Dealloc(X3942)
			JCudaTensor x2538;
			x2538 = x2507;
			x2538.free();

			// V_4c_c_bn_bias <~~ X8552
			float x2540, x2541;
			float x2542;
			float x2543;
			x2542 = 1;
			x2543 = lrn_rate;
			x2540 = x2542 * x2543;
			x2541 = momentum;
			JCudaTensor x2544;
			x2544 = x2512;
			x2539.update(x2544, x2540, x2541);

			// Dealloc(X8552)
			JCudaTensor x2545;
			x2545 = x2512;
			x2545.free();

			// 4c_c_bn_scale <~~ V_4c_c_bn_scale
			float x2546, x2547;
			x2546 = 1;
			float x2548;
			float x2549;
			x2548 = 1;
			float x2550;
			float x2551;
			float x2552;
			float x2553;
			x2552 = 1;
			x2553 = decay;
			x2550 = x2552 * x2553;
			float x2554;
			float x2555;
			x2554 = 1;
			x2555 = lrn_rate;
			x2551 = x2554 * x2555;
			x2549 = x2550 * x2551;
			x2547 = x2548 + x2549;
			JCudaTensor x2556;
			x2556 = x2521;
			x589.update(x2556, x2546, x2547);

			// 4c_c_cv_W <~~ V_4c_c_cv_W
			float x2557, x2558;
			x2557 = 1;
			float x2559;
			float x2560;
			x2559 = 1;
			float x2561;
			float x2562;
			float x2563;
			float x2564;
			x2563 = 1;
			x2564 = decay;
			x2561 = x2563 * x2564;
			float x2565;
			float x2566;
			x2565 = 1;
			x2566 = lrn_rate;
			x2562 = x2565 * x2566;
			x2560 = x2561 * x2562;
			x2558 = x2559 + x2560;
			JCudaTensor x2567;
			x2567 = x2531;
			x583.update(x2567, x2557, x2558);

			// 4c_c_bn_bias <~~ V_4c_c_bn_bias
			float x2568, x2569;
			x2568 = 1;
			float x2570;
			float x2571;
			x2570 = 1;
			float x2572;
			float x2573;
			float x2574;
			float x2575;
			x2574 = 1;
			x2575 = decay;
			x2572 = x2574 * x2575;
			float x2576;
			float x2577;
			x2576 = 1;
			x2577 = lrn_rate;
			x2573 = x2576 * x2577;
			x2571 = x2572 * x2573;
			x2569 = x2570 + x2571;
			JCudaTensor x2578;
			x2578 = x2539;
			x590.update(x2578, x2568, x2569);

			// val X3947 = X3943 * d_ReLU()(X8732)/d_X8731
			JCudaTensor x2579;
			JCudaTensor x2580, x2581;
			x2580 = x2528;
			x2581 = x577;
			x2579 = x457.backward(x2580, x2581);

			// Dealloc(X8732)
			JCudaTensor x2582;
			x2582 = x577;
			x2582.free();

			// val X3948 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_X8730
			JCudaTensor x2583;
			JCudaTensor x2584, x2585, x2586;
			x2584 = x2579;
			x2585 = x564;
			x2586 = x574;
			JCudaTensor[] x2587 = x576.backward(x2584,x2585,x2586);
			x2583 = x2587[0];

			// val X8550 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_4c_b_bn_scale
			JCudaTensor x2588;
			x2588 = x2587[1];

			// val X8549 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_4c_b_bn_bias
			JCudaTensor x2592;
			x2592 = x2587[2];

			// Dealloc(X8730)
			JCudaTensor x2596;
			x2596 = x564;
			x2596.free();

			// V_4c_b_bn_scale <~~ X8550
			float x2598, x2599;
			float x2600;
			float x2601;
			x2600 = 1;
			x2601 = lrn_rate;
			x2598 = x2600 * x2601;
			x2599 = momentum;
			JCudaTensor x2602;
			x2602 = x2588;
			x2597.update(x2602, x2598, x2599);

			// Dealloc(X8550)
			JCudaTensor x2603;
			x2603 = x2588;
			x2603.free();

			// val X3949 = X3948 * d_Convolv(1,1)(4c_b_cv_W)/d_X8729
			JCudaTensor x2604;
			JCudaTensor x2605, x2606;
			x2605 = x2583;
			x2606 = x568;
			x2604 = x464.backward_data(x2605, x2606);

			// V_4c_b_cv_W <~~ X3948 * d_Convolv(1,1)(X8729)/d_4c_b_cv_W
			float x2608, x2609;
			float x2610;
			float x2611;
			x2610 = 1;
			x2611 = lrn_rate;
			x2608 = x2610 * x2611;
			x2609 = momentum;
			JCudaTensor x2612, x2613;
			x2612 = x2583;
			x2613 = x562;
			x464.backward_filter(x2612, x2613, x2607, x2608, x2609);

			// Dealloc(X3948)
			JCudaTensor x2614;
			x2614 = x2583;
			x2614.free();

			// V_4c_b_bn_bias <~~ X8549
			float x2616, x2617;
			float x2618;
			float x2619;
			x2618 = 1;
			x2619 = lrn_rate;
			x2616 = x2618 * x2619;
			x2617 = momentum;
			JCudaTensor x2620;
			x2620 = x2592;
			x2615.update(x2620, x2616, x2617);

			// Dealloc(X8549)
			JCudaTensor x2621;
			x2621 = x2592;
			x2621.free();

			// 4c_b_bn_scale <~~ V_4c_b_bn_scale
			float x2622, x2623;
			x2622 = 1;
			float x2624;
			float x2625;
			x2624 = 1;
			float x2626;
			float x2627;
			float x2628;
			float x2629;
			x2628 = 1;
			x2629 = decay;
			x2626 = x2628 * x2629;
			float x2630;
			float x2631;
			x2630 = 1;
			x2631 = lrn_rate;
			x2627 = x2630 * x2631;
			x2625 = x2626 * x2627;
			x2623 = x2624 + x2625;
			JCudaTensor x2632;
			x2632 = x2597;
			x574.update(x2632, x2622, x2623);

			// 4c_b_cv_W <~~ V_4c_b_cv_W
			float x2633, x2634;
			x2633 = 1;
			float x2635;
			float x2636;
			x2635 = 1;
			float x2637;
			float x2638;
			float x2639;
			float x2640;
			x2639 = 1;
			x2640 = decay;
			x2637 = x2639 * x2640;
			float x2641;
			float x2642;
			x2641 = 1;
			x2642 = lrn_rate;
			x2638 = x2641 * x2642;
			x2636 = x2637 * x2638;
			x2634 = x2635 + x2636;
			JCudaTensor x2643;
			x2643 = x2607;
			x568.update(x2643, x2633, x2634);

			// 4c_b_bn_bias <~~ V_4c_b_bn_bias
			float x2644, x2645;
			x2644 = 1;
			float x2646;
			float x2647;
			x2646 = 1;
			float x2648;
			float x2649;
			float x2650;
			float x2651;
			x2650 = 1;
			x2651 = decay;
			x2648 = x2650 * x2651;
			float x2652;
			float x2653;
			x2652 = 1;
			x2653 = lrn_rate;
			x2649 = x2652 * x2653;
			x2647 = x2648 * x2649;
			x2645 = x2646 + x2647;
			JCudaTensor x2654;
			x2654 = x2615;
			x575.update(x2654, x2644, x2645);

			// val X3953 = X3949 * d_ReLU()(X8729)/d_X8728
			JCudaTensor x2655;
			JCudaTensor x2656, x2657;
			x2656 = x2604;
			x2657 = x562;
			x2655 = x457.backward(x2656, x2657);

			// Dealloc(X8729)
			JCudaTensor x2658;
			x2658 = x562;
			x2658.free();

			// val X8547 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_4c_a_bn_scale
			JCudaTensor x2659;
			JCudaTensor x2660, x2661, x2662;
			x2660 = x2655;
			x2661 = x549;
			x2662 = x559;
			JCudaTensor[] x2663 = x561.backward(x2660,x2661,x2662);
			x2659 = x2663[1];

			// val X3954 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_X8727
			JCudaTensor x2664;
			x2664 = x2663[0];

			// val X8546 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_4c_a_bn_bias
			JCudaTensor x2668;
			x2668 = x2663[2];

			// Dealloc(X8727)
			JCudaTensor x2672;
			x2672 = x549;
			x2672.free();

			// val X3955 = X3954 * d_Convolv(1,0)(4c_a_cv_W)/d_X8726
			JCudaTensor x2673;
			JCudaTensor x2674, x2675;
			x2674 = x2664;
			x2675 = x553;
			x2673 = x504.backward_data(x2674, x2675);

			// V_4c_a_cv_W <~~ X3954 * d_Convolv(1,0)(X8726)/d_4c_a_cv_W
			float x2677, x2678;
			float x2679;
			float x2680;
			x2679 = 1;
			x2680 = lrn_rate;
			x2677 = x2679 * x2680;
			x2678 = momentum;
			JCudaTensor x2681, x2682;
			x2681 = x2664;
			x2682 = x547;
			x504.backward_filter(x2681, x2682, x2676, x2677, x2678);

			// Dealloc(X3954)
			JCudaTensor x2683;
			x2683 = x2664;
			x2683.free();

			// V_4c_a_bn_bias <~~ X8546
			float x2685, x2686;
			float x2687;
			float x2688;
			x2687 = 1;
			x2688 = lrn_rate;
			x2685 = x2687 * x2688;
			x2686 = momentum;
			JCudaTensor x2689;
			x2689 = x2668;
			x2684.update(x2689, x2685, x2686);

			// Dealloc(X8546)
			JCudaTensor x2690;
			x2690 = x2668;
			x2690.free();

			// V_4c_a_bn_scale <~~ X8547
			float x2692, x2693;
			float x2694;
			float x2695;
			x2694 = 1;
			x2695 = lrn_rate;
			x2692 = x2694 * x2695;
			x2693 = momentum;
			JCudaTensor x2696;
			x2696 = x2659;
			x2691.update(x2696, x2692, x2693);

			// Dealloc(X8547)
			JCudaTensor x2697;
			x2697 = x2659;
			x2697.free();

			// 4c_a_cv_W <~~ V_4c_a_cv_W
			float x2698, x2699;
			x2698 = 1;
			float x2700;
			float x2701;
			x2700 = 1;
			float x2702;
			float x2703;
			float x2704;
			float x2705;
			x2704 = 1;
			x2705 = decay;
			x2702 = x2704 * x2705;
			float x2706;
			float x2707;
			x2706 = 1;
			x2707 = lrn_rate;
			x2703 = x2706 * x2707;
			x2701 = x2702 * x2703;
			x2699 = x2700 + x2701;
			JCudaTensor x2708;
			x2708 = x2676;
			x553.update(x2708, x2698, x2699);

			// 4c_a_bn_bias <~~ V_4c_a_bn_bias
			float x2709, x2710;
			x2709 = 1;
			float x2711;
			float x2712;
			x2711 = 1;
			float x2713;
			float x2714;
			float x2715;
			float x2716;
			x2715 = 1;
			x2716 = decay;
			x2713 = x2715 * x2716;
			float x2717;
			float x2718;
			x2717 = 1;
			x2718 = lrn_rate;
			x2714 = x2717 * x2718;
			x2712 = x2713 * x2714;
			x2710 = x2711 + x2712;
			JCudaTensor x2719;
			x2719 = x2684;
			x560.update(x2719, x2709, x2710);

			// 4c_a_bn_scale <~~ V_4c_a_bn_scale
			float x2720, x2721;
			x2720 = 1;
			float x2722;
			float x2723;
			x2722 = 1;
			float x2724;
			float x2725;
			float x2726;
			float x2727;
			x2726 = 1;
			x2727 = decay;
			x2724 = x2726 * x2727;
			float x2728;
			float x2729;
			x2728 = 1;
			x2729 = lrn_rate;
			x2725 = x2728 * x2729;
			x2723 = x2724 * x2725;
			x2721 = x2722 + x2723;
			JCudaTensor x2730;
			x2730 = x2691;
			x559.update(x2730, x2720, x2721);

			// val X3956 = (X3955 + X3931)
			JCudaTensor x2731;
			JCudaTensor x2732, x2733;
			x2732 = x2673;
			x2733 = x2499;
			x2731 = x2732.plus_i(x2733);

			// Dealloc(X3931)
			JCudaTensor x2734;
			x2734 = x2499;
			x2734.free();

			// val X3968 = X3956 * d_ReLU()(X8726)/d_X8725
			JCudaTensor x2735;
			JCudaTensor x2736, x2737;
			x2736 = x2731;
			x2737 = x547;
			x2735 = x490.backward(x2736, x2737);

			// Dealloc(X8726)
			JCudaTensor x2738;
			x2738 = x547;
			x2738.free();

			// val X3978 = X3968.copy * d_ReLU()(X8724)/d_X8723
			JCudaTensor x2739;
			JCudaTensor x2740, x2741;
			x2740 = x2735;
			x2740 = x2740.clone();
			x2741 = x542;
			x2739 = x490.backward(x2740, x2741);

			// Dealloc(X8724)
			JCudaTensor x2742;
			x2742 = x542;
			x2742.free();

			// val X8544 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_4b_c_bn_scale
			JCudaTensor x2743;
			JCudaTensor x2744, x2745, x2746;
			x2744 = x2739;
			x2745 = x529;
			x2746 = x539;
			JCudaTensor[] x2747 = x541.backward(x2744,x2745,x2746);
			x2743 = x2747[1];

			// val X3979 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_X8722
			JCudaTensor x2748;
			x2748 = x2747[0];

			// val X8543 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_4b_c_bn_bias
			JCudaTensor x2752;
			x2752 = x2747[2];

			// Dealloc(X8722)
			JCudaTensor x2756;
			x2756 = x529;
			x2756.free();

			// val X3980 = X3979 * d_Convolv(1,0)(4b_c_cv_W)/d_X8721
			JCudaTensor x2757;
			JCudaTensor x2758, x2759;
			x2758 = x2748;
			x2759 = x533;
			x2757 = x480.backward_data(x2758, x2759);

			// V_4b_c_cv_W <~~ X3979 * d_Convolv(1,0)(X8721)/d_4b_c_cv_W
			float x2761, x2762;
			float x2763;
			float x2764;
			x2763 = 1;
			x2764 = lrn_rate;
			x2761 = x2763 * x2764;
			x2762 = momentum;
			JCudaTensor x2765, x2766;
			x2765 = x2748;
			x2766 = x527;
			x480.backward_filter(x2765, x2766, x2760, x2761, x2762);

			// Dealloc(X3979)
			JCudaTensor x2767;
			x2767 = x2748;
			x2767.free();

			// V_4b_c_bn_bias <~~ X8543
			float x2769, x2770;
			float x2771;
			float x2772;
			x2771 = 1;
			x2772 = lrn_rate;
			x2769 = x2771 * x2772;
			x2770 = momentum;
			JCudaTensor x2773;
			x2773 = x2752;
			x2768.update(x2773, x2769, x2770);

			// Dealloc(X8543)
			JCudaTensor x2774;
			x2774 = x2752;
			x2774.free();

			// V_4b_c_bn_scale <~~ X8544
			float x2776, x2777;
			float x2778;
			float x2779;
			x2778 = 1;
			x2779 = lrn_rate;
			x2776 = x2778 * x2779;
			x2777 = momentum;
			JCudaTensor x2780;
			x2780 = x2743;
			x2775.update(x2780, x2776, x2777);

			// Dealloc(X8544)
			JCudaTensor x2781;
			x2781 = x2743;
			x2781.free();

			// 4b_c_cv_W <~~ V_4b_c_cv_W
			float x2782, x2783;
			x2782 = 1;
			float x2784;
			float x2785;
			x2784 = 1;
			float x2786;
			float x2787;
			float x2788;
			float x2789;
			x2788 = 1;
			x2789 = decay;
			x2786 = x2788 * x2789;
			float x2790;
			float x2791;
			x2790 = 1;
			x2791 = lrn_rate;
			x2787 = x2790 * x2791;
			x2785 = x2786 * x2787;
			x2783 = x2784 + x2785;
			JCudaTensor x2792;
			x2792 = x2760;
			x533.update(x2792, x2782, x2783);

			// 4b_c_bn_bias <~~ V_4b_c_bn_bias
			float x2793, x2794;
			x2793 = 1;
			float x2795;
			float x2796;
			x2795 = 1;
			float x2797;
			float x2798;
			float x2799;
			float x2800;
			x2799 = 1;
			x2800 = decay;
			x2797 = x2799 * x2800;
			float x2801;
			float x2802;
			x2801 = 1;
			x2802 = lrn_rate;
			x2798 = x2801 * x2802;
			x2796 = x2797 * x2798;
			x2794 = x2795 + x2796;
			JCudaTensor x2803;
			x2803 = x2768;
			x540.update(x2803, x2793, x2794);

			// 4b_c_bn_scale <~~ V_4b_c_bn_scale
			float x2804, x2805;
			x2804 = 1;
			float x2806;
			float x2807;
			x2806 = 1;
			float x2808;
			float x2809;
			float x2810;
			float x2811;
			x2810 = 1;
			x2811 = decay;
			x2808 = x2810 * x2811;
			float x2812;
			float x2813;
			x2812 = 1;
			x2813 = lrn_rate;
			x2809 = x2812 * x2813;
			x2807 = x2808 * x2809;
			x2805 = x2806 + x2807;
			JCudaTensor x2814;
			x2814 = x2775;
			x539.update(x2814, x2804, x2805);

			// val X3984 = X3980 * d_ReLU()(X8721)/d_X8720
			JCudaTensor x2815;
			JCudaTensor x2816, x2817;
			x2816 = x2757;
			x2817 = x527;
			x2815 = x457.backward(x2816, x2817);

			// Dealloc(X8721)
			JCudaTensor x2818;
			x2818 = x527;
			x2818.free();

			// val X3985 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_X8719
			JCudaTensor x2819;
			JCudaTensor x2820, x2821, x2822;
			x2820 = x2815;
			x2821 = x514;
			x2822 = x524;
			JCudaTensor[] x2823 = x526.backward(x2820,x2821,x2822);
			x2819 = x2823[0];

			// val X8540 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_4b_b_bn_bias
			JCudaTensor x2824;
			x2824 = x2823[2];

			// val X8541 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_4b_b_bn_scale
			JCudaTensor x2828;
			x2828 = x2823[1];

			// Dealloc(X8719)
			JCudaTensor x2832;
			x2832 = x514;
			x2832.free();

			// val X3986 = X3985 * d_Convolv(1,1)(4b_b_cv_W)/d_X8718
			JCudaTensor x2833;
			JCudaTensor x2834, x2835;
			x2834 = x2819;
			x2835 = x518;
			x2833 = x464.backward_data(x2834, x2835);

			// V_4b_b_cv_W <~~ X3985 * d_Convolv(1,1)(X8718)/d_4b_b_cv_W
			float x2837, x2838;
			float x2839;
			float x2840;
			x2839 = 1;
			x2840 = lrn_rate;
			x2837 = x2839 * x2840;
			x2838 = momentum;
			JCudaTensor x2841, x2842;
			x2841 = x2819;
			x2842 = x512;
			x464.backward_filter(x2841, x2842, x2836, x2837, x2838);

			// Dealloc(X3985)
			JCudaTensor x2843;
			x2843 = x2819;
			x2843.free();

			// V_4b_b_bn_scale <~~ X8541
			float x2845, x2846;
			float x2847;
			float x2848;
			x2847 = 1;
			x2848 = lrn_rate;
			x2845 = x2847 * x2848;
			x2846 = momentum;
			JCudaTensor x2849;
			x2849 = x2828;
			x2844.update(x2849, x2845, x2846);

			// Dealloc(X8541)
			JCudaTensor x2850;
			x2850 = x2828;
			x2850.free();

			// V_4b_b_bn_bias <~~ X8540
			float x2852, x2853;
			float x2854;
			float x2855;
			x2854 = 1;
			x2855 = lrn_rate;
			x2852 = x2854 * x2855;
			x2853 = momentum;
			JCudaTensor x2856;
			x2856 = x2824;
			x2851.update(x2856, x2852, x2853);

			// Dealloc(X8540)
			JCudaTensor x2857;
			x2857 = x2824;
			x2857.free();

			// 4b_b_cv_W <~~ V_4b_b_cv_W
			float x2858, x2859;
			x2858 = 1;
			float x2860;
			float x2861;
			x2860 = 1;
			float x2862;
			float x2863;
			float x2864;
			float x2865;
			x2864 = 1;
			x2865 = decay;
			x2862 = x2864 * x2865;
			float x2866;
			float x2867;
			x2866 = 1;
			x2867 = lrn_rate;
			x2863 = x2866 * x2867;
			x2861 = x2862 * x2863;
			x2859 = x2860 + x2861;
			JCudaTensor x2868;
			x2868 = x2836;
			x518.update(x2868, x2858, x2859);

			// 4b_b_bn_scale <~~ V_4b_b_bn_scale
			float x2869, x2870;
			x2869 = 1;
			float x2871;
			float x2872;
			x2871 = 1;
			float x2873;
			float x2874;
			float x2875;
			float x2876;
			x2875 = 1;
			x2876 = decay;
			x2873 = x2875 * x2876;
			float x2877;
			float x2878;
			x2877 = 1;
			x2878 = lrn_rate;
			x2874 = x2877 * x2878;
			x2872 = x2873 * x2874;
			x2870 = x2871 + x2872;
			JCudaTensor x2879;
			x2879 = x2844;
			x524.update(x2879, x2869, x2870);

			// 4b_b_bn_bias <~~ V_4b_b_bn_bias
			float x2880, x2881;
			x2880 = 1;
			float x2882;
			float x2883;
			x2882 = 1;
			float x2884;
			float x2885;
			float x2886;
			float x2887;
			x2886 = 1;
			x2887 = decay;
			x2884 = x2886 * x2887;
			float x2888;
			float x2889;
			x2888 = 1;
			x2889 = lrn_rate;
			x2885 = x2888 * x2889;
			x2883 = x2884 * x2885;
			x2881 = x2882 + x2883;
			JCudaTensor x2890;
			x2890 = x2851;
			x525.update(x2890, x2880, x2881);

			// val X3990 = X3986 * d_ReLU()(X8718)/d_X8717
			JCudaTensor x2891;
			JCudaTensor x2892, x2893;
			x2892 = x2833;
			x2893 = x512;
			x2891 = x457.backward(x2892, x2893);

			// Dealloc(X8718)
			JCudaTensor x2894;
			x2894 = x512;
			x2894.free();

			// val X8537 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_4b_a_bn_bias
			JCudaTensor x2895;
			JCudaTensor x2896, x2897, x2898;
			x2896 = x2891;
			x2897 = x498;
			x2898 = x509;
			JCudaTensor[] x2899 = x511.backward(x2896,x2897,x2898);
			x2895 = x2899[2];

			// val X3991 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_X8716
			JCudaTensor x2900;
			x2900 = x2899[0];

			// val X8538 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_4b_a_bn_scale
			JCudaTensor x2904;
			x2904 = x2899[1];

			// Dealloc(X8716)
			JCudaTensor x2908;
			x2908 = x498;
			x2908.free();

			// V_4b_a_bn_bias <~~ X8537
			float x2910, x2911;
			float x2912;
			float x2913;
			x2912 = 1;
			x2913 = lrn_rate;
			x2910 = x2912 * x2913;
			x2911 = momentum;
			JCudaTensor x2914;
			x2914 = x2895;
			x2909.update(x2914, x2910, x2911);

			// Dealloc(X8537)
			JCudaTensor x2915;
			x2915 = x2895;
			x2915.free();

			// val X3992 = X3991 * d_Convolv(1,0)(4b_a_cv_W)/d_X8715
			JCudaTensor x2916;
			JCudaTensor x2917, x2918;
			x2917 = x2900;
			x2918 = x502;
			x2916 = x504.backward_data(x2917, x2918);

			// V_4b_a_cv_W <~~ X3991 * d_Convolv(1,0)(X8715)/d_4b_a_cv_W
			float x2920, x2921;
			float x2922;
			float x2923;
			x2922 = 1;
			x2923 = lrn_rate;
			x2920 = x2922 * x2923;
			x2921 = momentum;
			JCudaTensor x2924, x2925;
			x2924 = x2900;
			x2925 = x496;
			x504.backward_filter(x2924, x2925, x2919, x2920, x2921);

			// Dealloc(X3991)
			JCudaTensor x2926;
			x2926 = x2900;
			x2926.free();

			// V_4b_a_bn_scale <~~ X8538
			float x2928, x2929;
			float x2930;
			float x2931;
			x2930 = 1;
			x2931 = lrn_rate;
			x2928 = x2930 * x2931;
			x2929 = momentum;
			JCudaTensor x2932;
			x2932 = x2904;
			x2927.update(x2932, x2928, x2929);

			// Dealloc(X8538)
			JCudaTensor x2933;
			x2933 = x2904;
			x2933.free();

			// 4b_a_bn_bias <~~ V_4b_a_bn_bias
			float x2934, x2935;
			x2934 = 1;
			float x2936;
			float x2937;
			x2936 = 1;
			float x2938;
			float x2939;
			float x2940;
			float x2941;
			x2940 = 1;
			x2941 = decay;
			x2938 = x2940 * x2941;
			float x2942;
			float x2943;
			x2942 = 1;
			x2943 = lrn_rate;
			x2939 = x2942 * x2943;
			x2937 = x2938 * x2939;
			x2935 = x2936 + x2937;
			JCudaTensor x2944;
			x2944 = x2909;
			x510.update(x2944, x2934, x2935);

			// 4b_a_cv_W <~~ V_4b_a_cv_W
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
			x2955 = x2919;
			x502.update(x2955, x2945, x2946);

			// 4b_a_bn_scale <~~ V_4b_a_bn_scale
			float x2956, x2957;
			x2956 = 1;
			float x2958;
			float x2959;
			x2958 = 1;
			float x2960;
			float x2961;
			float x2962;
			float x2963;
			x2962 = 1;
			x2963 = decay;
			x2960 = x2962 * x2963;
			float x2964;
			float x2965;
			x2964 = 1;
			x2965 = lrn_rate;
			x2961 = x2964 * x2965;
			x2959 = x2960 * x2961;
			x2957 = x2958 + x2959;
			JCudaTensor x2966;
			x2966 = x2927;
			x509.update(x2966, x2956, x2957);

			// val X3993 = (X3992 + X3968)
			JCudaTensor x2967;
			JCudaTensor x2968, x2969;
			x2968 = x2916;
			x2969 = x2735;
			x2967 = x2968.plus_i(x2969);

			// Dealloc(X3968)
			JCudaTensor x2970;
			x2970 = x2735;
			x2970.free();

			// val X4008 = X3993 * d_ReLU()(X8715)/d_X8714
			JCudaTensor x2971;
			JCudaTensor x2972, x2973;
			x2972 = x2967;
			x2973 = x496;
			x2971 = x490.backward(x2972, x2973);

			// Dealloc(X8715)
			JCudaTensor x2974;
			x2974 = x496;
			x2974.free();

			// val X4012 = X4008.copy * d_ReLU()(X8704)/d_X8703
			JCudaTensor x2975;
			JCudaTensor x2976, x2977;
			x2976 = x2971;
			x2976 = x2976.clone();
			x2977 = x488;
			x2975 = x490.backward(x2976, x2977);

			// Dealloc(X8704)
			JCudaTensor x2978;
			x2978 = x488;
			x2978.free();

			// val X4024 = X4008.copy * d_ReLU()(X8713)/d_X8712
			JCudaTensor x2979;
			JCudaTensor x2980, x2981;
			x2980 = x2971;
			x2980 = x2980.clone();
			x2981 = x491;
			x2979 = x490.backward(x2980, x2981);

			// Dealloc(X4008)
			JCudaTensor x2982;
			x2982 = x2971;
			x2982.free();

			// Dealloc(X8713)
			JCudaTensor x2983;
			x2983 = x491;
			x2983.free();

			// val X8526 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_4a1_bn_scale
			JCudaTensor x2984;
			JCudaTensor x2985, x2986, x2987;
			x2985 = x2975;
			x2986 = x427;
			x2987 = x445;
			JCudaTensor[] x2988 = x447.backward(x2985,x2986,x2987);
			x2984 = x2988[1];

			// val X8525 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_4a1_bn_bias
			JCudaTensor x2989;
			x2989 = x2988[2];

			// val X4013 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_X8702
			JCudaTensor x2993;
			x2993 = x2988[0];

			// Dealloc(X8702)
			JCudaTensor x2997;
			x2997 = x427;
			x2997.free();

			// val X8534 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_4a2_c_bn_bias
			JCudaTensor x2998;
			JCudaTensor x2999, x3000, x3001;
			x2999 = x2979;
			x3000 = x474;
			x3001 = x485;
			JCudaTensor[] x3002 = x487.backward(x2999,x3000,x3001);
			x2998 = x3002[2];

			// val X4025 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_X8711
			JCudaTensor x3003;
			x3003 = x3002[0];

			// val X8535 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_4a2_c_bn_scale
			JCudaTensor x3007;
			x3007 = x3002[1];

			// Dealloc(X8711)
			JCudaTensor x3011;
			x3011 = x474;
			x3011.free();

			// V_4a2_c_bn_scale <~~ X8535
			float x3013, x3014;
			float x3015;
			float x3016;
			x3015 = 1;
			x3016 = lrn_rate;
			x3013 = x3015 * x3016;
			x3014 = momentum;
			JCudaTensor x3017;
			x3017 = x3007;
			x3012.update(x3017, x3013, x3014);

			// Dealloc(X8535)
			JCudaTensor x3018;
			x3018 = x3007;
			x3018.free();

			// V_4a1_cv_W <~~ X4013 * d_Convolv(2,0)(X8701)/d_4a1_cv_W
			float x3020, x3021;
			float x3022;
			float x3023;
			x3022 = 1;
			x3023 = lrn_rate;
			x3020 = x3022 * x3023;
			x3021 = momentum;
			JCudaTensor x3024, x3025;
			x3024 = x2993;
			x3025 = x425;
			x433.backward_filter(x3024, x3025, x3019, x3020, x3021);

			// V_4a1_bn_bias <~~ X8525
			float x3027, x3028;
			float x3029;
			float x3030;
			x3029 = 1;
			x3030 = lrn_rate;
			x3027 = x3029 * x3030;
			x3028 = momentum;
			JCudaTensor x3031;
			x3031 = x2989;
			x3026.update(x3031, x3027, x3028);

			// Dealloc(X8525)
			JCudaTensor x3032;
			x3032 = x2989;
			x3032.free();

			// V_4a2_c_cv_W <~~ X4025 * d_Convolv(1,0)(X8710)/d_4a2_c_cv_W
			float x3034, x3035;
			float x3036;
			float x3037;
			x3036 = 1;
			x3037 = lrn_rate;
			x3034 = x3036 * x3037;
			x3035 = momentum;
			JCudaTensor x3038, x3039;
			x3038 = x3003;
			x3039 = x472;
			x480.backward_filter(x3038, x3039, x3033, x3034, x3035);

			// V_4a1_bn_scale <~~ X8526
			float x3041, x3042;
			float x3043;
			float x3044;
			x3043 = 1;
			x3044 = lrn_rate;
			x3041 = x3043 * x3044;
			x3042 = momentum;
			JCudaTensor x3045;
			x3045 = x2984;
			x3040.update(x3045, x3041, x3042);

			// Dealloc(X8526)
			JCudaTensor x3046;
			x3046 = x2984;
			x3046.free();

			// val X4014 = X4013 * d_Convolv(2,0)(4a1_cv_W)/d_X8701
			JCudaTensor x3047;
			JCudaTensor x3048, x3049;
			x3048 = x2993;
			x3049 = x431;
			x3047 = x433.backward_data(x3048, x3049);

			// Dealloc(X4013)
			JCudaTensor x3050;
			x3050 = x2993;
			x3050.free();

			// val X4026 = X4025 * d_Convolv(1,0)(4a2_c_cv_W)/d_X8710
			JCudaTensor x3051;
			JCudaTensor x3052, x3053;
			x3052 = x3003;
			x3053 = x478;
			x3051 = x480.backward_data(x3052, x3053);

			// Dealloc(X4025)
			JCudaTensor x3054;
			x3054 = x3003;
			x3054.free();

			// V_4a2_c_bn_bias <~~ X8534
			float x3056, x3057;
			float x3058;
			float x3059;
			x3058 = 1;
			x3059 = lrn_rate;
			x3056 = x3058 * x3059;
			x3057 = momentum;
			JCudaTensor x3060;
			x3060 = x2998;
			x3055.update(x3060, x3056, x3057);

			// Dealloc(X8534)
			JCudaTensor x3061;
			x3061 = x2998;
			x3061.free();

			// 4a1_cv_W <~~ V_4a1_cv_W
			float x3062, x3063;
			x3062 = 1;
			float x3064;
			float x3065;
			x3064 = 1;
			float x3066;
			float x3067;
			float x3068;
			float x3069;
			x3068 = 1;
			x3069 = decay;
			x3066 = x3068 * x3069;
			float x3070;
			float x3071;
			x3070 = 1;
			x3071 = lrn_rate;
			x3067 = x3070 * x3071;
			x3065 = x3066 * x3067;
			x3063 = x3064 + x3065;
			JCudaTensor x3072;
			x3072 = x3019;
			x431.update(x3072, x3062, x3063);

			// 4a1_bn_scale <~~ V_4a1_bn_scale
			float x3073, x3074;
			x3073 = 1;
			float x3075;
			float x3076;
			x3075 = 1;
			float x3077;
			float x3078;
			float x3079;
			float x3080;
			x3079 = 1;
			x3080 = decay;
			x3077 = x3079 * x3080;
			float x3081;
			float x3082;
			x3081 = 1;
			x3082 = lrn_rate;
			x3078 = x3081 * x3082;
			x3076 = x3077 * x3078;
			x3074 = x3075 + x3076;
			JCudaTensor x3083;
			x3083 = x3040;
			x445.update(x3083, x3073, x3074);

			// 4a2_c_bn_scale <~~ V_4a2_c_bn_scale
			float x3084, x3085;
			x3084 = 1;
			float x3086;
			float x3087;
			x3086 = 1;
			float x3088;
			float x3089;
			float x3090;
			float x3091;
			x3090 = 1;
			x3091 = decay;
			x3088 = x3090 * x3091;
			float x3092;
			float x3093;
			x3092 = 1;
			x3093 = lrn_rate;
			x3089 = x3092 * x3093;
			x3087 = x3088 * x3089;
			x3085 = x3086 + x3087;
			JCudaTensor x3094;
			x3094 = x3012;
			x485.update(x3094, x3084, x3085);

			// 4a2_c_cv_W <~~ V_4a2_c_cv_W
			float x3095, x3096;
			x3095 = 1;
			float x3097;
			float x3098;
			x3097 = 1;
			float x3099;
			float x3100;
			float x3101;
			float x3102;
			x3101 = 1;
			x3102 = decay;
			x3099 = x3101 * x3102;
			float x3103;
			float x3104;
			x3103 = 1;
			x3104 = lrn_rate;
			x3100 = x3103 * x3104;
			x3098 = x3099 * x3100;
			x3096 = x3097 + x3098;
			JCudaTensor x3105;
			x3105 = x3033;
			x478.update(x3105, x3095, x3096);

			// 4a1_bn_bias <~~ V_4a1_bn_bias
			float x3106, x3107;
			x3106 = 1;
			float x3108;
			float x3109;
			x3108 = 1;
			float x3110;
			float x3111;
			float x3112;
			float x3113;
			x3112 = 1;
			x3113 = decay;
			x3110 = x3112 * x3113;
			float x3114;
			float x3115;
			x3114 = 1;
			x3115 = lrn_rate;
			x3111 = x3114 * x3115;
			x3109 = x3110 * x3111;
			x3107 = x3108 + x3109;
			JCudaTensor x3116;
			x3116 = x3026;
			x446.update(x3116, x3106, x3107);

			// 4a2_c_bn_bias <~~ V_4a2_c_bn_bias
			float x3117, x3118;
			x3117 = 1;
			float x3119;
			float x3120;
			x3119 = 1;
			float x3121;
			float x3122;
			float x3123;
			float x3124;
			x3123 = 1;
			x3124 = decay;
			x3121 = x3123 * x3124;
			float x3125;
			float x3126;
			x3125 = 1;
			x3126 = lrn_rate;
			x3122 = x3125 * x3126;
			x3120 = x3121 * x3122;
			x3118 = x3119 + x3120;
			JCudaTensor x3127;
			x3127 = x3055;
			x486.update(x3127, x3117, x3118);

			// val X4030 = X4026 * d_ReLU()(X8710)/d_X8709
			JCudaTensor x3128;
			JCudaTensor x3129, x3130;
			x3129 = x3051;
			x3130 = x472;
			x3128 = x457.backward(x3129, x3130);

			// Dealloc(X8710)
			JCudaTensor x3131;
			x3131 = x472;
			x3131.free();

			// val X8531 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_4a2_b_bn_bias
			JCudaTensor x3132;
			JCudaTensor x3133, x3134, x3135;
			x3133 = x3128;
			x3134 = x458;
			x3135 = x469;
			JCudaTensor[] x3136 = x471.backward(x3133,x3134,x3135);
			x3132 = x3136[2];

			// val X4031 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_X8708
			JCudaTensor x3137;
			x3137 = x3136[0];

			// val X8532 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_4a2_b_bn_scale
			JCudaTensor x3141;
			x3141 = x3136[1];

			// Dealloc(X8708)
			JCudaTensor x3145;
			x3145 = x458;
			x3145.free();

			// V_4a2_b_bn_bias <~~ X8531
			float x3147, x3148;
			float x3149;
			float x3150;
			x3149 = 1;
			x3150 = lrn_rate;
			x3147 = x3149 * x3150;
			x3148 = momentum;
			JCudaTensor x3151;
			x3151 = x3132;
			x3146.update(x3151, x3147, x3148);

			// Dealloc(X8531)
			JCudaTensor x3152;
			x3152 = x3132;
			x3152.free();

			// V_4a2_b_bn_scale <~~ X8532
			float x3154, x3155;
			float x3156;
			float x3157;
			x3156 = 1;
			x3157 = lrn_rate;
			x3154 = x3156 * x3157;
			x3155 = momentum;
			JCudaTensor x3158;
			x3158 = x3141;
			x3153.update(x3158, x3154, x3155);

			// Dealloc(X8532)
			JCudaTensor x3159;
			x3159 = x3141;
			x3159.free();

			// val X4032 = X4031 * d_Convolv(1,1)(4a2_b_cv_W)/d_X8707
			JCudaTensor x3160;
			JCudaTensor x3161, x3162;
			x3161 = x3137;
			x3162 = x462;
			x3160 = x464.backward_data(x3161, x3162);

			// V_4a2_b_cv_W <~~ X4031 * d_Convolv(1,1)(X8707)/d_4a2_b_cv_W
			float x3164, x3165;
			float x3166;
			float x3167;
			x3166 = 1;
			x3167 = lrn_rate;
			x3164 = x3166 * x3167;
			x3165 = momentum;
			JCudaTensor x3168, x3169;
			x3168 = x3137;
			x3169 = x455;
			x464.backward_filter(x3168, x3169, x3163, x3164, x3165);

			// Dealloc(X4031)
			JCudaTensor x3170;
			x3170 = x3137;
			x3170.free();

			// 4a2_b_bn_bias <~~ V_4a2_b_bn_bias
			float x3171, x3172;
			x3171 = 1;
			float x3173;
			float x3174;
			x3173 = 1;
			float x3175;
			float x3176;
			float x3177;
			float x3178;
			x3177 = 1;
			x3178 = decay;
			x3175 = x3177 * x3178;
			float x3179;
			float x3180;
			x3179 = 1;
			x3180 = lrn_rate;
			x3176 = x3179 * x3180;
			x3174 = x3175 * x3176;
			x3172 = x3173 + x3174;
			JCudaTensor x3181;
			x3181 = x3146;
			x470.update(x3181, x3171, x3172);

			// 4a2_b_bn_scale <~~ V_4a2_b_bn_scale
			float x3182, x3183;
			x3182 = 1;
			float x3184;
			float x3185;
			x3184 = 1;
			float x3186;
			float x3187;
			float x3188;
			float x3189;
			x3188 = 1;
			x3189 = decay;
			x3186 = x3188 * x3189;
			float x3190;
			float x3191;
			x3190 = 1;
			x3191 = lrn_rate;
			x3187 = x3190 * x3191;
			x3185 = x3186 * x3187;
			x3183 = x3184 + x3185;
			JCudaTensor x3192;
			x3192 = x3153;
			x469.update(x3192, x3182, x3183);

			// 4a2_b_cv_W <~~ V_4a2_b_cv_W
			float x3193, x3194;
			x3193 = 1;
			float x3195;
			float x3196;
			x3195 = 1;
			float x3197;
			float x3198;
			float x3199;
			float x3200;
			x3199 = 1;
			x3200 = decay;
			x3197 = x3199 * x3200;
			float x3201;
			float x3202;
			x3201 = 1;
			x3202 = lrn_rate;
			x3198 = x3201 * x3202;
			x3196 = x3197 * x3198;
			x3194 = x3195 + x3196;
			JCudaTensor x3203;
			x3203 = x3163;
			x462.update(x3203, x3193, x3194);

			// val X4036 = X4032 * d_ReLU()(X8707)/d_X8706
			JCudaTensor x3204;
			JCudaTensor x3205, x3206;
			x3205 = x3160;
			x3206 = x455;
			x3204 = x457.backward(x3205, x3206);

			// Dealloc(X8707)
			JCudaTensor x3207;
			x3207 = x455;
			x3207.free();

			// val X8528 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_4a2_a_bn_bias
			JCudaTensor x3208;
			JCudaTensor x3209, x3210, x3211;
			x3209 = x3204;
			x3210 = x434;
			x3211 = x452;
			JCudaTensor[] x3212 = x454.backward(x3209,x3210,x3211);
			x3208 = x3212[2];

			// val X8529 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_4a2_a_bn_scale
			JCudaTensor x3213;
			x3213 = x3212[1];

			// val X4037 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_X8705
			JCudaTensor x3217;
			x3217 = x3212[0];

			// Dealloc(X8705)
			JCudaTensor x3221;
			x3221 = x434;
			x3221.free();

			// V_4a2_a_bn_scale <~~ X8529
			float x3223, x3224;
			float x3225;
			float x3226;
			x3225 = 1;
			x3226 = lrn_rate;
			x3223 = x3225 * x3226;
			x3224 = momentum;
			JCudaTensor x3227;
			x3227 = x3213;
			x3222.update(x3227, x3223, x3224);

			// Dealloc(X8529)
			JCudaTensor x3228;
			x3228 = x3213;
			x3228.free();

			// val X4039 = (X4014 + X4037 * d_Convolv(2,0)(4a2_a_cv_W)/d_X8701)
			JCudaTensor x3229;
			JCudaTensor x3230;
			x3230 = x3047;
			JCudaTensor x3231, x3232;
			x3231 = x3217;
			x3232 = x438;
			x3229 = x440.backward_data(x3231,x3232, x3230);

			// V_4a2_a_cv_W <~~ X4037 * d_Convolv(2,0)(X8701)/d_4a2_a_cv_W
			float x3234, x3235;
			float x3236;
			float x3237;
			x3236 = 1;
			x3237 = lrn_rate;
			x3234 = x3236 * x3237;
			x3235 = momentum;
			JCudaTensor x3238, x3239;
			x3238 = x3217;
			x3239 = x425;
			x440.backward_filter(x3238, x3239, x3233, x3234, x3235);

			// Dealloc(X4037)
			JCudaTensor x3240;
			x3240 = x3217;
			x3240.free();

			// V_4a2_a_bn_bias <~~ X8528
			float x3242, x3243;
			float x3244;
			float x3245;
			x3244 = 1;
			x3245 = lrn_rate;
			x3242 = x3244 * x3245;
			x3243 = momentum;
			JCudaTensor x3246;
			x3246 = x3208;
			x3241.update(x3246, x3242, x3243);

			// Dealloc(X8528)
			JCudaTensor x3247;
			x3247 = x3208;
			x3247.free();

			// 4a2_a_bn_scale <~~ V_4a2_a_bn_scale
			float x3248, x3249;
			x3248 = 1;
			float x3250;
			float x3251;
			x3250 = 1;
			float x3252;
			float x3253;
			float x3254;
			float x3255;
			x3254 = 1;
			x3255 = decay;
			x3252 = x3254 * x3255;
			float x3256;
			float x3257;
			x3256 = 1;
			x3257 = lrn_rate;
			x3253 = x3256 * x3257;
			x3251 = x3252 * x3253;
			x3249 = x3250 + x3251;
			JCudaTensor x3258;
			x3258 = x3222;
			x452.update(x3258, x3248, x3249);

			// 4a2_a_cv_W <~~ V_4a2_a_cv_W
			float x3259, x3260;
			x3259 = 1;
			float x3261;
			float x3262;
			x3261 = 1;
			float x3263;
			float x3264;
			float x3265;
			float x3266;
			x3265 = 1;
			x3266 = decay;
			x3263 = x3265 * x3266;
			float x3267;
			float x3268;
			x3267 = 1;
			x3268 = lrn_rate;
			x3264 = x3267 * x3268;
			x3262 = x3263 * x3264;
			x3260 = x3261 + x3262;
			JCudaTensor x3269;
			x3269 = x3233;
			x438.update(x3269, x3259, x3260);

			// 4a2_a_bn_bias <~~ V_4a2_a_bn_bias
			float x3270, x3271;
			x3270 = 1;
			float x3272;
			float x3273;
			x3272 = 1;
			float x3274;
			float x3275;
			float x3276;
			float x3277;
			x3276 = 1;
			x3277 = decay;
			x3274 = x3276 * x3277;
			float x3278;
			float x3279;
			x3278 = 1;
			x3279 = lrn_rate;
			x3275 = x3278 * x3279;
			x3273 = x3274 * x3275;
			x3271 = x3272 + x3273;
			JCudaTensor x3280;
			x3280 = x3241;
			x453.update(x3280, x3270, x3271);

			// val X4087 = X4039 * d_ReLU()(X8701)/d_X8700
			JCudaTensor x3281;
			JCudaTensor x3282, x3283;
			x3282 = x3229;
			x3283 = x425;
			x3281 = x268.backward(x3282, x3283);

			// Dealloc(X8701)
			JCudaTensor x3284;
			x3284 = x425;
			x3284.free();

			// val X4097 = X4087.copy * d_ReLU()(X8699)/d_X8698
			JCudaTensor x3285;
			JCudaTensor x3286, x3287;
			x3286 = x3281;
			x3286 = x3286.clone();
			x3287 = x420;
			x3285 = x268.backward(x3286, x3287);

			// Dealloc(X8699)
			JCudaTensor x3288;
			x3288 = x420;
			x3288.free();

			// val X8522 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_3d_c_bn_bias
			JCudaTensor x3289;
			JCudaTensor x3290, x3291, x3292;
			x3290 = x3285;
			x3291 = x407;
			x3292 = x417;
			JCudaTensor[] x3293 = x419.backward(x3290,x3291,x3292);
			x3289 = x3293[2];

			// val X8523 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_3d_c_bn_scale
			JCudaTensor x3294;
			x3294 = x3293[1];

			// val X4098 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_X8697
			JCudaTensor x3298;
			x3298 = x3293[0];

			// Dealloc(X8697)
			JCudaTensor x3302;
			x3302 = x407;
			x3302.free();

			// V_3d_c_bn_bias <~~ X8522
			float x3304, x3305;
			float x3306;
			float x3307;
			x3306 = 1;
			x3307 = lrn_rate;
			x3304 = x3306 * x3307;
			x3305 = momentum;
			JCudaTensor x3308;
			x3308 = x3289;
			x3303.update(x3308, x3304, x3305);

			// Dealloc(X8522)
			JCudaTensor x3309;
			x3309 = x3289;
			x3309.free();

			// V_3d_c_bn_scale <~~ X8523
			float x3311, x3312;
			float x3313;
			float x3314;
			x3313 = 1;
			x3314 = lrn_rate;
			x3311 = x3313 * x3314;
			x3312 = momentum;
			JCudaTensor x3315;
			x3315 = x3294;
			x3310.update(x3315, x3311, x3312);

			// Dealloc(X8523)
			JCudaTensor x3316;
			x3316 = x3294;
			x3316.free();

			// val X4099 = X4098 * d_Convolv(1,0)(3d_c_cv_W)/d_X8696
			JCudaTensor x3317;
			JCudaTensor x3318, x3319;
			x3318 = x3298;
			x3319 = x411;
			x3317 = x258.backward_data(x3318, x3319);

			// V_3d_c_cv_W <~~ X4098 * d_Convolv(1,0)(X8696)/d_3d_c_cv_W
			float x3321, x3322;
			float x3323;
			float x3324;
			x3323 = 1;
			x3324 = lrn_rate;
			x3321 = x3323 * x3324;
			x3322 = momentum;
			JCudaTensor x3325, x3326;
			x3325 = x3298;
			x3326 = x405;
			x258.backward_filter(x3325, x3326, x3320, x3321, x3322);

			// Dealloc(X4098)
			JCudaTensor x3327;
			x3327 = x3298;
			x3327.free();

			// 3d_c_bn_bias <~~ V_3d_c_bn_bias
			float x3328, x3329;
			x3328 = 1;
			float x3330;
			float x3331;
			x3330 = 1;
			float x3332;
			float x3333;
			float x3334;
			float x3335;
			x3334 = 1;
			x3335 = decay;
			x3332 = x3334 * x3335;
			float x3336;
			float x3337;
			x3336 = 1;
			x3337 = lrn_rate;
			x3333 = x3336 * x3337;
			x3331 = x3332 * x3333;
			x3329 = x3330 + x3331;
			JCudaTensor x3338;
			x3338 = x3303;
			x418.update(x3338, x3328, x3329);

			// 3d_c_bn_scale <~~ V_3d_c_bn_scale
			float x3339, x3340;
			x3339 = 1;
			float x3341;
			float x3342;
			x3341 = 1;
			float x3343;
			float x3344;
			float x3345;
			float x3346;
			x3345 = 1;
			x3346 = decay;
			x3343 = x3345 * x3346;
			float x3347;
			float x3348;
			x3347 = 1;
			x3348 = lrn_rate;
			x3344 = x3347 * x3348;
			x3342 = x3343 * x3344;
			x3340 = x3341 + x3342;
			JCudaTensor x3349;
			x3349 = x3310;
			x417.update(x3349, x3339, x3340);

			// 3d_c_cv_W <~~ V_3d_c_cv_W
			float x3350, x3351;
			x3350 = 1;
			float x3352;
			float x3353;
			x3352 = 1;
			float x3354;
			float x3355;
			float x3356;
			float x3357;
			x3356 = 1;
			x3357 = decay;
			x3354 = x3356 * x3357;
			float x3358;
			float x3359;
			x3358 = 1;
			x3359 = lrn_rate;
			x3355 = x3358 * x3359;
			x3353 = x3354 * x3355;
			x3351 = x3352 + x3353;
			JCudaTensor x3360;
			x3360 = x3320;
			x411.update(x3360, x3350, x3351);

			// val X4103 = X4099 * d_ReLU()(X8696)/d_X8695
			JCudaTensor x3361;
			JCudaTensor x3362, x3363;
			x3362 = x3317;
			x3363 = x405;
			x3361 = x235.backward(x3362, x3363);

			// Dealloc(X8696)
			JCudaTensor x3364;
			x3364 = x405;
			x3364.free();

			// val X4104 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_X8694
			JCudaTensor x3365;
			JCudaTensor x3366, x3367, x3368;
			x3366 = x3361;
			x3367 = x392;
			x3368 = x402;
			JCudaTensor[] x3369 = x404.backward(x3366,x3367,x3368);
			x3365 = x3369[0];

			// val X8519 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_3d_b_bn_bias
			JCudaTensor x3370;
			x3370 = x3369[2];

			// val X8520 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_3d_b_bn_scale
			JCudaTensor x3374;
			x3374 = x3369[1];

			// Dealloc(X8694)
			JCudaTensor x3378;
			x3378 = x392;
			x3378.free();

			// V_3d_b_bn_bias <~~ X8519
			float x3380, x3381;
			float x3382;
			float x3383;
			x3382 = 1;
			x3383 = lrn_rate;
			x3380 = x3382 * x3383;
			x3381 = momentum;
			JCudaTensor x3384;
			x3384 = x3370;
			x3379.update(x3384, x3380, x3381);

			// Dealloc(X8519)
			JCudaTensor x3385;
			x3385 = x3370;
			x3385.free();

			// val X4105 = X4104 * d_Convolv(1,1)(3d_b_cv_W)/d_X8693
			JCudaTensor x3386;
			JCudaTensor x3387, x3388;
			x3387 = x3365;
			x3388 = x396;
			x3386 = x242.backward_data(x3387, x3388);

			// V_3d_b_cv_W <~~ X4104 * d_Convolv(1,1)(X8693)/d_3d_b_cv_W
			float x3390, x3391;
			float x3392;
			float x3393;
			x3392 = 1;
			x3393 = lrn_rate;
			x3390 = x3392 * x3393;
			x3391 = momentum;
			JCudaTensor x3394, x3395;
			x3394 = x3365;
			x3395 = x390;
			x242.backward_filter(x3394, x3395, x3389, x3390, x3391);

			// Dealloc(X4104)
			JCudaTensor x3396;
			x3396 = x3365;
			x3396.free();

			// V_3d_b_bn_scale <~~ X8520
			float x3398, x3399;
			float x3400;
			float x3401;
			x3400 = 1;
			x3401 = lrn_rate;
			x3398 = x3400 * x3401;
			x3399 = momentum;
			JCudaTensor x3402;
			x3402 = x3374;
			x3397.update(x3402, x3398, x3399);

			// Dealloc(X8520)
			JCudaTensor x3403;
			x3403 = x3374;
			x3403.free();

			// 3d_b_bn_bias <~~ V_3d_b_bn_bias
			float x3404, x3405;
			x3404 = 1;
			float x3406;
			float x3407;
			x3406 = 1;
			float x3408;
			float x3409;
			float x3410;
			float x3411;
			x3410 = 1;
			x3411 = decay;
			x3408 = x3410 * x3411;
			float x3412;
			float x3413;
			x3412 = 1;
			x3413 = lrn_rate;
			x3409 = x3412 * x3413;
			x3407 = x3408 * x3409;
			x3405 = x3406 + x3407;
			JCudaTensor x3414;
			x3414 = x3379;
			x403.update(x3414, x3404, x3405);

			// 3d_b_cv_W <~~ V_3d_b_cv_W
			float x3415, x3416;
			x3415 = 1;
			float x3417;
			float x3418;
			x3417 = 1;
			float x3419;
			float x3420;
			float x3421;
			float x3422;
			x3421 = 1;
			x3422 = decay;
			x3419 = x3421 * x3422;
			float x3423;
			float x3424;
			x3423 = 1;
			x3424 = lrn_rate;
			x3420 = x3423 * x3424;
			x3418 = x3419 * x3420;
			x3416 = x3417 + x3418;
			JCudaTensor x3425;
			x3425 = x3389;
			x396.update(x3425, x3415, x3416);

			// 3d_b_bn_scale <~~ V_3d_b_bn_scale
			float x3426, x3427;
			x3426 = 1;
			float x3428;
			float x3429;
			x3428 = 1;
			float x3430;
			float x3431;
			float x3432;
			float x3433;
			x3432 = 1;
			x3433 = decay;
			x3430 = x3432 * x3433;
			float x3434;
			float x3435;
			x3434 = 1;
			x3435 = lrn_rate;
			x3431 = x3434 * x3435;
			x3429 = x3430 * x3431;
			x3427 = x3428 + x3429;
			JCudaTensor x3436;
			x3436 = x3397;
			x402.update(x3436, x3426, x3427);

			// val X4109 = X4105 * d_ReLU()(X8693)/d_X8692
			JCudaTensor x3437;
			JCudaTensor x3438, x3439;
			x3438 = x3386;
			x3439 = x390;
			x3437 = x235.backward(x3438, x3439);

			// Dealloc(X8693)
			JCudaTensor x3440;
			x3440 = x390;
			x3440.free();

			// val X8517 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_3d_a_bn_scale
			JCudaTensor x3441;
			JCudaTensor x3442, x3443, x3444;
			x3442 = x3437;
			x3443 = x377;
			x3444 = x387;
			JCudaTensor[] x3445 = x389.backward(x3442,x3443,x3444);
			x3441 = x3445[1];

			// val X4110 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_X8691
			JCudaTensor x3446;
			x3446 = x3445[0];

			// val X8516 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_3d_a_bn_bias
			JCudaTensor x3450;
			x3450 = x3445[2];

			// Dealloc(X8691)
			JCudaTensor x3454;
			x3454 = x377;
			x3454.free();

			// V_3d_a_bn_bias <~~ X8516
			float x3456, x3457;
			float x3458;
			float x3459;
			x3458 = 1;
			x3459 = lrn_rate;
			x3456 = x3458 * x3459;
			x3457 = momentum;
			JCudaTensor x3460;
			x3460 = x3450;
			x3455.update(x3460, x3456, x3457);

			// Dealloc(X8516)
			JCudaTensor x3461;
			x3461 = x3450;
			x3461.free();

			// V_3d_a_bn_scale <~~ X8517
			float x3463, x3464;
			float x3465;
			float x3466;
			x3465 = 1;
			x3466 = lrn_rate;
			x3463 = x3465 * x3466;
			x3464 = momentum;
			JCudaTensor x3467;
			x3467 = x3441;
			x3462.update(x3467, x3463, x3464);

			// Dealloc(X8517)
			JCudaTensor x3468;
			x3468 = x3441;
			x3468.free();

			// val X4111 = X4110 * d_Convolv(1,0)(3d_a_cv_W)/d_X8690
			JCudaTensor x3469;
			JCudaTensor x3470, x3471;
			x3470 = x3446;
			x3471 = x381;
			x3469 = x282.backward_data(x3470, x3471);

			// V_3d_a_cv_W <~~ X4110 * d_Convolv(1,0)(X8690)/d_3d_a_cv_W
			float x3473, x3474;
			float x3475;
			float x3476;
			x3475 = 1;
			x3476 = lrn_rate;
			x3473 = x3475 * x3476;
			x3474 = momentum;
			JCudaTensor x3477, x3478;
			x3477 = x3446;
			x3478 = x375;
			x282.backward_filter(x3477, x3478, x3472, x3473, x3474);

			// Dealloc(X4110)
			JCudaTensor x3479;
			x3479 = x3446;
			x3479.free();

			// 3d_a_bn_bias <~~ V_3d_a_bn_bias
			float x3480, x3481;
			x3480 = 1;
			float x3482;
			float x3483;
			x3482 = 1;
			float x3484;
			float x3485;
			float x3486;
			float x3487;
			x3486 = 1;
			x3487 = decay;
			x3484 = x3486 * x3487;
			float x3488;
			float x3489;
			x3488 = 1;
			x3489 = lrn_rate;
			x3485 = x3488 * x3489;
			x3483 = x3484 * x3485;
			x3481 = x3482 + x3483;
			JCudaTensor x3490;
			x3490 = x3455;
			x388.update(x3490, x3480, x3481);

			// 3d_a_bn_scale <~~ V_3d_a_bn_scale
			float x3491, x3492;
			x3491 = 1;
			float x3493;
			float x3494;
			x3493 = 1;
			float x3495;
			float x3496;
			float x3497;
			float x3498;
			x3497 = 1;
			x3498 = decay;
			x3495 = x3497 * x3498;
			float x3499;
			float x3500;
			x3499 = 1;
			x3500 = lrn_rate;
			x3496 = x3499 * x3500;
			x3494 = x3495 * x3496;
			x3492 = x3493 + x3494;
			JCudaTensor x3501;
			x3501 = x3462;
			x387.update(x3501, x3491, x3492);

			// 3d_a_cv_W <~~ V_3d_a_cv_W
			float x3502, x3503;
			x3502 = 1;
			float x3504;
			float x3505;
			x3504 = 1;
			float x3506;
			float x3507;
			float x3508;
			float x3509;
			x3508 = 1;
			x3509 = decay;
			x3506 = x3508 * x3509;
			float x3510;
			float x3511;
			x3510 = 1;
			x3511 = lrn_rate;
			x3507 = x3510 * x3511;
			x3505 = x3506 * x3507;
			x3503 = x3504 + x3505;
			JCudaTensor x3512;
			x3512 = x3472;
			x381.update(x3512, x3502, x3503);

			// val X4112 = (X4111 + X4087)
			JCudaTensor x3513;
			JCudaTensor x3514, x3515;
			x3514 = x3469;
			x3515 = x3281;
			x3513 = x3514.plus_i(x3515);

			// Dealloc(X4087)
			JCudaTensor x3516;
			x3516 = x3281;
			x3516.free();

			// val X4124 = X4112 * d_ReLU()(X8690)/d_X8689
			JCudaTensor x3517;
			JCudaTensor x3518, x3519;
			x3518 = x3513;
			x3519 = x375;
			x3517 = x268.backward(x3518, x3519);

			// Dealloc(X8690)
			JCudaTensor x3520;
			x3520 = x375;
			x3520.free();

			// val X4134 = X4124.copy * d_ReLU()(X8688)/d_X8687
			JCudaTensor x3521;
			JCudaTensor x3522, x3523;
			x3522 = x3517;
			x3522 = x3522.clone();
			x3523 = x370;
			x3521 = x268.backward(x3522, x3523);

			// Dealloc(X8688)
			JCudaTensor x3524;
			x3524 = x370;
			x3524.free();

			// val X4135 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_X8686
			JCudaTensor x3525;
			JCudaTensor x3526, x3527, x3528;
			x3526 = x3521;
			x3527 = x357;
			x3528 = x367;
			JCudaTensor[] x3529 = x369.backward(x3526,x3527,x3528);
			x3525 = x3529[0];

			// val X8514 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_3c_c_bn_scale
			JCudaTensor x3530;
			x3530 = x3529[1];

			// val X8513 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_3c_c_bn_bias
			JCudaTensor x3534;
			x3534 = x3529[2];

			// Dealloc(X8686)
			JCudaTensor x3538;
			x3538 = x357;
			x3538.free();

			// V_3c_c_bn_scale <~~ X8514
			float x3540, x3541;
			float x3542;
			float x3543;
			x3542 = 1;
			x3543 = lrn_rate;
			x3540 = x3542 * x3543;
			x3541 = momentum;
			JCudaTensor x3544;
			x3544 = x3530;
			x3539.update(x3544, x3540, x3541);

			// Dealloc(X8514)
			JCudaTensor x3545;
			x3545 = x3530;
			x3545.free();

			// val X4136 = X4135 * d_Convolv(1,0)(3c_c_cv_W)/d_X8685
			JCudaTensor x3546;
			JCudaTensor x3547, x3548;
			x3547 = x3525;
			x3548 = x361;
			x3546 = x258.backward_data(x3547, x3548);

			// V_3c_c_bn_bias <~~ X8513
			float x3550, x3551;
			float x3552;
			float x3553;
			x3552 = 1;
			x3553 = lrn_rate;
			x3550 = x3552 * x3553;
			x3551 = momentum;
			JCudaTensor x3554;
			x3554 = x3534;
			x3549.update(x3554, x3550, x3551);

			// Dealloc(X8513)
			JCudaTensor x3555;
			x3555 = x3534;
			x3555.free();

			// V_3c_c_cv_W <~~ X4135 * d_Convolv(1,0)(X8685)/d_3c_c_cv_W
			float x3557, x3558;
			float x3559;
			float x3560;
			x3559 = 1;
			x3560 = lrn_rate;
			x3557 = x3559 * x3560;
			x3558 = momentum;
			JCudaTensor x3561, x3562;
			x3561 = x3525;
			x3562 = x355;
			x258.backward_filter(x3561, x3562, x3556, x3557, x3558);

			// Dealloc(X4135)
			JCudaTensor x3563;
			x3563 = x3525;
			x3563.free();

			// 3c_c_bn_scale <~~ V_3c_c_bn_scale
			float x3564, x3565;
			x3564 = 1;
			float x3566;
			float x3567;
			x3566 = 1;
			float x3568;
			float x3569;
			float x3570;
			float x3571;
			x3570 = 1;
			x3571 = decay;
			x3568 = x3570 * x3571;
			float x3572;
			float x3573;
			x3572 = 1;
			x3573 = lrn_rate;
			x3569 = x3572 * x3573;
			x3567 = x3568 * x3569;
			x3565 = x3566 + x3567;
			JCudaTensor x3574;
			x3574 = x3539;
			x367.update(x3574, x3564, x3565);

			// 3c_c_bn_bias <~~ V_3c_c_bn_bias
			float x3575, x3576;
			x3575 = 1;
			float x3577;
			float x3578;
			x3577 = 1;
			float x3579;
			float x3580;
			float x3581;
			float x3582;
			x3581 = 1;
			x3582 = decay;
			x3579 = x3581 * x3582;
			float x3583;
			float x3584;
			x3583 = 1;
			x3584 = lrn_rate;
			x3580 = x3583 * x3584;
			x3578 = x3579 * x3580;
			x3576 = x3577 + x3578;
			JCudaTensor x3585;
			x3585 = x3549;
			x368.update(x3585, x3575, x3576);

			// 3c_c_cv_W <~~ V_3c_c_cv_W
			float x3586, x3587;
			x3586 = 1;
			float x3588;
			float x3589;
			x3588 = 1;
			float x3590;
			float x3591;
			float x3592;
			float x3593;
			x3592 = 1;
			x3593 = decay;
			x3590 = x3592 * x3593;
			float x3594;
			float x3595;
			x3594 = 1;
			x3595 = lrn_rate;
			x3591 = x3594 * x3595;
			x3589 = x3590 * x3591;
			x3587 = x3588 + x3589;
			JCudaTensor x3596;
			x3596 = x3556;
			x361.update(x3596, x3586, x3587);

			// val X4140 = X4136 * d_ReLU()(X8685)/d_X8684
			JCudaTensor x3597;
			JCudaTensor x3598, x3599;
			x3598 = x3546;
			x3599 = x355;
			x3597 = x235.backward(x3598, x3599);

			// Dealloc(X8685)
			JCudaTensor x3600;
			x3600 = x355;
			x3600.free();

			// val X4141 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_X8683
			JCudaTensor x3601;
			JCudaTensor x3602, x3603, x3604;
			x3602 = x3597;
			x3603 = x342;
			x3604 = x352;
			JCudaTensor[] x3605 = x354.backward(x3602,x3603,x3604);
			x3601 = x3605[0];

			// val X8510 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_3c_b_bn_bias
			JCudaTensor x3606;
			x3606 = x3605[2];

			// val X8511 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_3c_b_bn_scale
			JCudaTensor x3610;
			x3610 = x3605[1];

			// Dealloc(X8683)
			JCudaTensor x3614;
			x3614 = x342;
			x3614.free();

			// val X4142 = X4141 * d_Convolv(1,1)(3c_b_cv_W)/d_X8682
			JCudaTensor x3615;
			JCudaTensor x3616, x3617;
			x3616 = x3601;
			x3617 = x346;
			x3615 = x242.backward_data(x3616, x3617);

			// V_3c_b_bn_scale <~~ X8511
			float x3619, x3620;
			float x3621;
			float x3622;
			x3621 = 1;
			x3622 = lrn_rate;
			x3619 = x3621 * x3622;
			x3620 = momentum;
			JCudaTensor x3623;
			x3623 = x3610;
			x3618.update(x3623, x3619, x3620);

			// Dealloc(X8511)
			JCudaTensor x3624;
			x3624 = x3610;
			x3624.free();

			// V_3c_b_bn_bias <~~ X8510
			float x3626, x3627;
			float x3628;
			float x3629;
			x3628 = 1;
			x3629 = lrn_rate;
			x3626 = x3628 * x3629;
			x3627 = momentum;
			JCudaTensor x3630;
			x3630 = x3606;
			x3625.update(x3630, x3626, x3627);

			// Dealloc(X8510)
			JCudaTensor x3631;
			x3631 = x3606;
			x3631.free();

			// V_3c_b_cv_W <~~ X4141 * d_Convolv(1,1)(X8682)/d_3c_b_cv_W
			float x3633, x3634;
			float x3635;
			float x3636;
			x3635 = 1;
			x3636 = lrn_rate;
			x3633 = x3635 * x3636;
			x3634 = momentum;
			JCudaTensor x3637, x3638;
			x3637 = x3601;
			x3638 = x340;
			x242.backward_filter(x3637, x3638, x3632, x3633, x3634);

			// Dealloc(X4141)
			JCudaTensor x3639;
			x3639 = x3601;
			x3639.free();

			// 3c_b_bn_scale <~~ V_3c_b_bn_scale
			float x3640, x3641;
			x3640 = 1;
			float x3642;
			float x3643;
			x3642 = 1;
			float x3644;
			float x3645;
			float x3646;
			float x3647;
			x3646 = 1;
			x3647 = decay;
			x3644 = x3646 * x3647;
			float x3648;
			float x3649;
			x3648 = 1;
			x3649 = lrn_rate;
			x3645 = x3648 * x3649;
			x3643 = x3644 * x3645;
			x3641 = x3642 + x3643;
			JCudaTensor x3650;
			x3650 = x3618;
			x352.update(x3650, x3640, x3641);

			// 3c_b_bn_bias <~~ V_3c_b_bn_bias
			float x3651, x3652;
			x3651 = 1;
			float x3653;
			float x3654;
			x3653 = 1;
			float x3655;
			float x3656;
			float x3657;
			float x3658;
			x3657 = 1;
			x3658 = decay;
			x3655 = x3657 * x3658;
			float x3659;
			float x3660;
			x3659 = 1;
			x3660 = lrn_rate;
			x3656 = x3659 * x3660;
			x3654 = x3655 * x3656;
			x3652 = x3653 + x3654;
			JCudaTensor x3661;
			x3661 = x3625;
			x353.update(x3661, x3651, x3652);

			// 3c_b_cv_W <~~ V_3c_b_cv_W
			float x3662, x3663;
			x3662 = 1;
			float x3664;
			float x3665;
			x3664 = 1;
			float x3666;
			float x3667;
			float x3668;
			float x3669;
			x3668 = 1;
			x3669 = decay;
			x3666 = x3668 * x3669;
			float x3670;
			float x3671;
			x3670 = 1;
			x3671 = lrn_rate;
			x3667 = x3670 * x3671;
			x3665 = x3666 * x3667;
			x3663 = x3664 + x3665;
			JCudaTensor x3672;
			x3672 = x3632;
			x346.update(x3672, x3662, x3663);

			// val X4146 = X4142 * d_ReLU()(X8682)/d_X8681
			JCudaTensor x3673;
			JCudaTensor x3674, x3675;
			x3674 = x3615;
			x3675 = x340;
			x3673 = x235.backward(x3674, x3675);

			// Dealloc(X8682)
			JCudaTensor x3676;
			x3676 = x340;
			x3676.free();

			// val X4147 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_X8680
			JCudaTensor x3677;
			JCudaTensor x3678, x3679, x3680;
			x3678 = x3673;
			x3679 = x327;
			x3680 = x337;
			JCudaTensor[] x3681 = x339.backward(x3678,x3679,x3680);
			x3677 = x3681[0];

			// val X8507 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_3c_a_bn_bias
			JCudaTensor x3682;
			x3682 = x3681[2];

			// val X8508 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_3c_a_bn_scale
			JCudaTensor x3686;
			x3686 = x3681[1];

			// Dealloc(X8680)
			JCudaTensor x3690;
			x3690 = x327;
			x3690.free();

			// val X4148 = X4147 * d_Convolv(1,0)(3c_a_cv_W)/d_X8679
			JCudaTensor x3691;
			JCudaTensor x3692, x3693;
			x3692 = x3677;
			x3693 = x331;
			x3691 = x282.backward_data(x3692, x3693);

			// V_3c_a_bn_scale <~~ X8508
			float x3695, x3696;
			float x3697;
			float x3698;
			x3697 = 1;
			x3698 = lrn_rate;
			x3695 = x3697 * x3698;
			x3696 = momentum;
			JCudaTensor x3699;
			x3699 = x3686;
			x3694.update(x3699, x3695, x3696);

			// Dealloc(X8508)
			JCudaTensor x3700;
			x3700 = x3686;
			x3700.free();

			// V_3c_a_bn_bias <~~ X8507
			float x3702, x3703;
			float x3704;
			float x3705;
			x3704 = 1;
			x3705 = lrn_rate;
			x3702 = x3704 * x3705;
			x3703 = momentum;
			JCudaTensor x3706;
			x3706 = x3682;
			x3701.update(x3706, x3702, x3703);

			// Dealloc(X8507)
			JCudaTensor x3707;
			x3707 = x3682;
			x3707.free();

			// V_3c_a_cv_W <~~ X4147 * d_Convolv(1,0)(X8679)/d_3c_a_cv_W
			float x3709, x3710;
			float x3711;
			float x3712;
			x3711 = 1;
			x3712 = lrn_rate;
			x3709 = x3711 * x3712;
			x3710 = momentum;
			JCudaTensor x3713, x3714;
			x3713 = x3677;
			x3714 = x325;
			x282.backward_filter(x3713, x3714, x3708, x3709, x3710);

			// Dealloc(X4147)
			JCudaTensor x3715;
			x3715 = x3677;
			x3715.free();

			// 3c_a_bn_scale <~~ V_3c_a_bn_scale
			float x3716, x3717;
			x3716 = 1;
			float x3718;
			float x3719;
			x3718 = 1;
			float x3720;
			float x3721;
			float x3722;
			float x3723;
			x3722 = 1;
			x3723 = decay;
			x3720 = x3722 * x3723;
			float x3724;
			float x3725;
			x3724 = 1;
			x3725 = lrn_rate;
			x3721 = x3724 * x3725;
			x3719 = x3720 * x3721;
			x3717 = x3718 + x3719;
			JCudaTensor x3726;
			x3726 = x3694;
			x337.update(x3726, x3716, x3717);

			// 3c_a_bn_bias <~~ V_3c_a_bn_bias
			float x3727, x3728;
			x3727 = 1;
			float x3729;
			float x3730;
			x3729 = 1;
			float x3731;
			float x3732;
			float x3733;
			float x3734;
			x3733 = 1;
			x3734 = decay;
			x3731 = x3733 * x3734;
			float x3735;
			float x3736;
			x3735 = 1;
			x3736 = lrn_rate;
			x3732 = x3735 * x3736;
			x3730 = x3731 * x3732;
			x3728 = x3729 + x3730;
			JCudaTensor x3737;
			x3737 = x3701;
			x338.update(x3737, x3727, x3728);

			// 3c_a_cv_W <~~ V_3c_a_cv_W
			float x3738, x3739;
			x3738 = 1;
			float x3740;
			float x3741;
			x3740 = 1;
			float x3742;
			float x3743;
			float x3744;
			float x3745;
			x3744 = 1;
			x3745 = decay;
			x3742 = x3744 * x3745;
			float x3746;
			float x3747;
			x3746 = 1;
			x3747 = lrn_rate;
			x3743 = x3746 * x3747;
			x3741 = x3742 * x3743;
			x3739 = x3740 + x3741;
			JCudaTensor x3748;
			x3748 = x3708;
			x331.update(x3748, x3738, x3739);

			// val X4149 = (X4148 + X4124)
			JCudaTensor x3749;
			JCudaTensor x3750, x3751;
			x3750 = x3691;
			x3751 = x3517;
			x3749 = x3750.plus_i(x3751);

			// Dealloc(X4124)
			JCudaTensor x3752;
			x3752 = x3517;
			x3752.free();

			// val X4161 = X4149 * d_ReLU()(X8679)/d_X8678
			JCudaTensor x3753;
			JCudaTensor x3754, x3755;
			x3754 = x3749;
			x3755 = x325;
			x3753 = x268.backward(x3754, x3755);

			// Dealloc(X8679)
			JCudaTensor x3756;
			x3756 = x325;
			x3756.free();

			// val X4171 = X4161.copy * d_ReLU()(X8677)/d_X8676
			JCudaTensor x3757;
			JCudaTensor x3758, x3759;
			x3758 = x3753;
			x3758 = x3758.clone();
			x3759 = x320;
			x3757 = x268.backward(x3758, x3759);

			// Dealloc(X8677)
			JCudaTensor x3760;
			x3760 = x320;
			x3760.free();

			// val X8504 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_3b_c_bn_bias
			JCudaTensor x3761;
			JCudaTensor x3762, x3763, x3764;
			x3762 = x3757;
			x3763 = x307;
			x3764 = x317;
			JCudaTensor[] x3765 = x319.backward(x3762,x3763,x3764);
			x3761 = x3765[2];

			// val X4172 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_X8675
			JCudaTensor x3766;
			x3766 = x3765[0];

			// val X8505 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_3b_c_bn_scale
			JCudaTensor x3770;
			x3770 = x3765[1];

			// Dealloc(X8675)
			JCudaTensor x3774;
			x3774 = x307;
			x3774.free();

			// val X4173 = X4172 * d_Convolv(1,0)(3b_c_cv_W)/d_X8674
			JCudaTensor x3775;
			JCudaTensor x3776, x3777;
			x3776 = x3766;
			x3777 = x311;
			x3775 = x258.backward_data(x3776, x3777);

			// V_3b_c_cv_W <~~ X4172 * d_Convolv(1,0)(X8674)/d_3b_c_cv_W
			float x3779, x3780;
			float x3781;
			float x3782;
			x3781 = 1;
			x3782 = lrn_rate;
			x3779 = x3781 * x3782;
			x3780 = momentum;
			JCudaTensor x3783, x3784;
			x3783 = x3766;
			x3784 = x305;
			x258.backward_filter(x3783, x3784, x3778, x3779, x3780);

			// Dealloc(X4172)
			JCudaTensor x3785;
			x3785 = x3766;
			x3785.free();

			// V_3b_c_bn_bias <~~ X8504
			float x3787, x3788;
			float x3789;
			float x3790;
			x3789 = 1;
			x3790 = lrn_rate;
			x3787 = x3789 * x3790;
			x3788 = momentum;
			JCudaTensor x3791;
			x3791 = x3761;
			x3786.update(x3791, x3787, x3788);

			// Dealloc(X8504)
			JCudaTensor x3792;
			x3792 = x3761;
			x3792.free();

			// V_3b_c_bn_scale <~~ X8505
			float x3794, x3795;
			float x3796;
			float x3797;
			x3796 = 1;
			x3797 = lrn_rate;
			x3794 = x3796 * x3797;
			x3795 = momentum;
			JCudaTensor x3798;
			x3798 = x3770;
			x3793.update(x3798, x3794, x3795);

			// Dealloc(X8505)
			JCudaTensor x3799;
			x3799 = x3770;
			x3799.free();

			// 3b_c_cv_W <~~ V_3b_c_cv_W
			float x3800, x3801;
			x3800 = 1;
			float x3802;
			float x3803;
			x3802 = 1;
			float x3804;
			float x3805;
			float x3806;
			float x3807;
			x3806 = 1;
			x3807 = decay;
			x3804 = x3806 * x3807;
			float x3808;
			float x3809;
			x3808 = 1;
			x3809 = lrn_rate;
			x3805 = x3808 * x3809;
			x3803 = x3804 * x3805;
			x3801 = x3802 + x3803;
			JCudaTensor x3810;
			x3810 = x3778;
			x311.update(x3810, x3800, x3801);

			// 3b_c_bn_bias <~~ V_3b_c_bn_bias
			float x3811, x3812;
			x3811 = 1;
			float x3813;
			float x3814;
			x3813 = 1;
			float x3815;
			float x3816;
			float x3817;
			float x3818;
			x3817 = 1;
			x3818 = decay;
			x3815 = x3817 * x3818;
			float x3819;
			float x3820;
			x3819 = 1;
			x3820 = lrn_rate;
			x3816 = x3819 * x3820;
			x3814 = x3815 * x3816;
			x3812 = x3813 + x3814;
			JCudaTensor x3821;
			x3821 = x3786;
			x318.update(x3821, x3811, x3812);

			// 3b_c_bn_scale <~~ V_3b_c_bn_scale
			float x3822, x3823;
			x3822 = 1;
			float x3824;
			float x3825;
			x3824 = 1;
			float x3826;
			float x3827;
			float x3828;
			float x3829;
			x3828 = 1;
			x3829 = decay;
			x3826 = x3828 * x3829;
			float x3830;
			float x3831;
			x3830 = 1;
			x3831 = lrn_rate;
			x3827 = x3830 * x3831;
			x3825 = x3826 * x3827;
			x3823 = x3824 + x3825;
			JCudaTensor x3832;
			x3832 = x3793;
			x317.update(x3832, x3822, x3823);

			// val X4177 = X4173 * d_ReLU()(X8674)/d_X8673
			JCudaTensor x3833;
			JCudaTensor x3834, x3835;
			x3834 = x3775;
			x3835 = x305;
			x3833 = x235.backward(x3834, x3835);

			// Dealloc(X8674)
			JCudaTensor x3836;
			x3836 = x305;
			x3836.free();

			// val X4178 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_X8672
			JCudaTensor x3837;
			JCudaTensor x3838, x3839, x3840;
			x3838 = x3833;
			x3839 = x292;
			x3840 = x302;
			JCudaTensor[] x3841 = x304.backward(x3838,x3839,x3840);
			x3837 = x3841[0];

			// val X8502 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_3b_b_bn_scale
			JCudaTensor x3842;
			x3842 = x3841[1];

			// val X8501 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_3b_b_bn_bias
			JCudaTensor x3846;
			x3846 = x3841[2];

			// Dealloc(X8672)
			JCudaTensor x3850;
			x3850 = x292;
			x3850.free();

			// V_3b_b_bn_bias <~~ X8501
			float x3852, x3853;
			float x3854;
			float x3855;
			x3854 = 1;
			x3855 = lrn_rate;
			x3852 = x3854 * x3855;
			x3853 = momentum;
			JCudaTensor x3856;
			x3856 = x3846;
			x3851.update(x3856, x3852, x3853);

			// Dealloc(X8501)
			JCudaTensor x3857;
			x3857 = x3846;
			x3857.free();

			// val X4179 = X4178 * d_Convolv(1,1)(3b_b_cv_W)/d_X8671
			JCudaTensor x3858;
			JCudaTensor x3859, x3860;
			x3859 = x3837;
			x3860 = x296;
			x3858 = x242.backward_data(x3859, x3860);

			// V_3b_b_cv_W <~~ X4178 * d_Convolv(1,1)(X8671)/d_3b_b_cv_W
			float x3862, x3863;
			float x3864;
			float x3865;
			x3864 = 1;
			x3865 = lrn_rate;
			x3862 = x3864 * x3865;
			x3863 = momentum;
			JCudaTensor x3866, x3867;
			x3866 = x3837;
			x3867 = x290;
			x242.backward_filter(x3866, x3867, x3861, x3862, x3863);

			// Dealloc(X4178)
			JCudaTensor x3868;
			x3868 = x3837;
			x3868.free();

			// V_3b_b_bn_scale <~~ X8502
			float x3870, x3871;
			float x3872;
			float x3873;
			x3872 = 1;
			x3873 = lrn_rate;
			x3870 = x3872 * x3873;
			x3871 = momentum;
			JCudaTensor x3874;
			x3874 = x3842;
			x3869.update(x3874, x3870, x3871);

			// Dealloc(X8502)
			JCudaTensor x3875;
			x3875 = x3842;
			x3875.free();

			// 3b_b_bn_bias <~~ V_3b_b_bn_bias
			float x3876, x3877;
			x3876 = 1;
			float x3878;
			float x3879;
			x3878 = 1;
			float x3880;
			float x3881;
			float x3882;
			float x3883;
			x3882 = 1;
			x3883 = decay;
			x3880 = x3882 * x3883;
			float x3884;
			float x3885;
			x3884 = 1;
			x3885 = lrn_rate;
			x3881 = x3884 * x3885;
			x3879 = x3880 * x3881;
			x3877 = x3878 + x3879;
			JCudaTensor x3886;
			x3886 = x3851;
			x303.update(x3886, x3876, x3877);

			// 3b_b_cv_W <~~ V_3b_b_cv_W
			float x3887, x3888;
			x3887 = 1;
			float x3889;
			float x3890;
			x3889 = 1;
			float x3891;
			float x3892;
			float x3893;
			float x3894;
			x3893 = 1;
			x3894 = decay;
			x3891 = x3893 * x3894;
			float x3895;
			float x3896;
			x3895 = 1;
			x3896 = lrn_rate;
			x3892 = x3895 * x3896;
			x3890 = x3891 * x3892;
			x3888 = x3889 + x3890;
			JCudaTensor x3897;
			x3897 = x3861;
			x296.update(x3897, x3887, x3888);

			// 3b_b_bn_scale <~~ V_3b_b_bn_scale
			float x3898, x3899;
			x3898 = 1;
			float x3900;
			float x3901;
			x3900 = 1;
			float x3902;
			float x3903;
			float x3904;
			float x3905;
			x3904 = 1;
			x3905 = decay;
			x3902 = x3904 * x3905;
			float x3906;
			float x3907;
			x3906 = 1;
			x3907 = lrn_rate;
			x3903 = x3906 * x3907;
			x3901 = x3902 * x3903;
			x3899 = x3900 + x3901;
			JCudaTensor x3908;
			x3908 = x3869;
			x302.update(x3908, x3898, x3899);

			// val X4183 = X4179 * d_ReLU()(X8671)/d_X8670
			JCudaTensor x3909;
			JCudaTensor x3910, x3911;
			x3910 = x3858;
			x3911 = x290;
			x3909 = x235.backward(x3910, x3911);

			// Dealloc(X8671)
			JCudaTensor x3912;
			x3912 = x290;
			x3912.free();

			// val X8498 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_3b_a_bn_bias
			JCudaTensor x3913;
			JCudaTensor x3914, x3915, x3916;
			x3914 = x3909;
			x3915 = x276;
			x3916 = x287;
			JCudaTensor[] x3917 = x289.backward(x3914,x3915,x3916);
			x3913 = x3917[2];

			// val X4184 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_X8669
			JCudaTensor x3918;
			x3918 = x3917[0];

			// val X8499 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_3b_a_bn_scale
			JCudaTensor x3922;
			x3922 = x3917[1];

			// Dealloc(X8669)
			JCudaTensor x3926;
			x3926 = x276;
			x3926.free();

			// V_3b_a_bn_bias <~~ X8498
			float x3928, x3929;
			float x3930;
			float x3931;
			x3930 = 1;
			x3931 = lrn_rate;
			x3928 = x3930 * x3931;
			x3929 = momentum;
			JCudaTensor x3932;
			x3932 = x3913;
			x3927.update(x3932, x3928, x3929);

			// Dealloc(X8498)
			JCudaTensor x3933;
			x3933 = x3913;
			x3933.free();

			// V_3b_a_bn_scale <~~ X8499
			float x3935, x3936;
			float x3937;
			float x3938;
			x3937 = 1;
			x3938 = lrn_rate;
			x3935 = x3937 * x3938;
			x3936 = momentum;
			JCudaTensor x3939;
			x3939 = x3922;
			x3934.update(x3939, x3935, x3936);

			// Dealloc(X8499)
			JCudaTensor x3940;
			x3940 = x3922;
			x3940.free();

			// val X4185 = X4184 * d_Convolv(1,0)(3b_a_cv_W)/d_X8668
			JCudaTensor x3941;
			JCudaTensor x3942, x3943;
			x3942 = x3918;
			x3943 = x280;
			x3941 = x282.backward_data(x3942, x3943);

			// V_3b_a_cv_W <~~ X4184 * d_Convolv(1,0)(X8668)/d_3b_a_cv_W
			float x3945, x3946;
			float x3947;
			float x3948;
			x3947 = 1;
			x3948 = lrn_rate;
			x3945 = x3947 * x3948;
			x3946 = momentum;
			JCudaTensor x3949, x3950;
			x3949 = x3918;
			x3950 = x274;
			x282.backward_filter(x3949, x3950, x3944, x3945, x3946);

			// Dealloc(X4184)
			JCudaTensor x3951;
			x3951 = x3918;
			x3951.free();

			// 3b_a_bn_bias <~~ V_3b_a_bn_bias
			float x3952, x3953;
			x3952 = 1;
			float x3954;
			float x3955;
			x3954 = 1;
			float x3956;
			float x3957;
			float x3958;
			float x3959;
			x3958 = 1;
			x3959 = decay;
			x3956 = x3958 * x3959;
			float x3960;
			float x3961;
			x3960 = 1;
			x3961 = lrn_rate;
			x3957 = x3960 * x3961;
			x3955 = x3956 * x3957;
			x3953 = x3954 + x3955;
			JCudaTensor x3962;
			x3962 = x3927;
			x288.update(x3962, x3952, x3953);

			// 3b_a_bn_scale <~~ V_3b_a_bn_scale
			float x3963, x3964;
			x3963 = 1;
			float x3965;
			float x3966;
			x3965 = 1;
			float x3967;
			float x3968;
			float x3969;
			float x3970;
			x3969 = 1;
			x3970 = decay;
			x3967 = x3969 * x3970;
			float x3971;
			float x3972;
			x3971 = 1;
			x3972 = lrn_rate;
			x3968 = x3971 * x3972;
			x3966 = x3967 * x3968;
			x3964 = x3965 + x3966;
			JCudaTensor x3973;
			x3973 = x3934;
			x287.update(x3973, x3963, x3964);

			// 3b_a_cv_W <~~ V_3b_a_cv_W
			float x3974, x3975;
			x3974 = 1;
			float x3976;
			float x3977;
			x3976 = 1;
			float x3978;
			float x3979;
			float x3980;
			float x3981;
			x3980 = 1;
			x3981 = decay;
			x3978 = x3980 * x3981;
			float x3982;
			float x3983;
			x3982 = 1;
			x3983 = lrn_rate;
			x3979 = x3982 * x3983;
			x3977 = x3978 * x3979;
			x3975 = x3976 + x3977;
			JCudaTensor x3984;
			x3984 = x3944;
			x280.update(x3984, x3974, x3975);

			// val X4186 = (X4185 + X4161)
			JCudaTensor x3985;
			JCudaTensor x3986, x3987;
			x3986 = x3941;
			x3987 = x3753;
			x3985 = x3986.plus_i(x3987);

			// Dealloc(X4161)
			JCudaTensor x3988;
			x3988 = x3753;
			x3988.free();

			// val X4201 = X4186 * d_ReLU()(X8668)/d_X8667
			JCudaTensor x3989;
			JCudaTensor x3990, x3991;
			x3990 = x3985;
			x3991 = x274;
			x3989 = x268.backward(x3990, x3991);

			// Dealloc(X8668)
			JCudaTensor x3992;
			x3992 = x274;
			x3992.free();

			// val X4205 = X4201.copy * d_ReLU()(X8657)/d_X8656
			JCudaTensor x3993;
			JCudaTensor x3994, x3995;
			x3994 = x3989;
			x3994 = x3994.clone();
			x3995 = x266;
			x3993 = x268.backward(x3994, x3995);

			// Dealloc(X8657)
			JCudaTensor x3996;
			x3996 = x266;
			x3996.free();

			// val X4217 = X4201.copy * d_ReLU()(X8666)/d_X8665
			JCudaTensor x3997;
			JCudaTensor x3998, x3999;
			x3998 = x3989;
			x3998 = x3998.clone();
			x3999 = x269;
			x3997 = x268.backward(x3998, x3999);

			// Dealloc(X4201)
			JCudaTensor x4000;
			x4000 = x3989;
			x4000.free();

			// Dealloc(X8666)
			JCudaTensor x4001;
			x4001 = x269;
			x4001.free();

			// val X4206 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_X8655
			JCudaTensor x4002;
			JCudaTensor x4003, x4004, x4005;
			x4003 = x3993;
			x4004 = x212;
			x4005 = x223;
			JCudaTensor[] x4006 = x225.backward(x4003,x4004,x4005);
			x4002 = x4006[0];

			// val X8495 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_3a2_c_bn_bias
			JCudaTensor x4007;
			JCudaTensor x4008, x4009, x4010;
			x4008 = x3997;
			x4009 = x252;
			x4010 = x263;
			JCudaTensor[] x4011 = x265.backward(x4008,x4009,x4010);
			x4007 = x4011[2];

			// val X8487 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_3a1_bn_scale
			JCudaTensor x4012;
			x4012 = x4006[1];

			// val X4218 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_X8664
			JCudaTensor x4016;
			x4016 = x4011[0];

			// val X8486 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_3a1_bn_bias
			JCudaTensor x4020;
			x4020 = x4006[2];

			// Dealloc(X8655)
			JCudaTensor x4024;
			x4024 = x212;
			x4024.free();

			// val X8496 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_3a2_c_bn_scale
			JCudaTensor x4025;
			x4025 = x4011[1];

			// Dealloc(X8664)
			JCudaTensor x4029;
			x4029 = x252;
			x4029.free();

			// val X4207 = X4206 * d_Convolv(2,0)(3a1_cv_W)/d_X8654
			JCudaTensor x4030;
			JCudaTensor x4031, x4032;
			x4031 = x4002;
			x4032 = x216;
			x4030 = x218.backward_data(x4031, x4032);

			// V_3a1_cv_W <~~ X4206 * d_Convolv(2,0)(X8654)/d_3a1_cv_W
			float x4034, x4035;
			float x4036;
			float x4037;
			x4036 = 1;
			x4037 = lrn_rate;
			x4034 = x4036 * x4037;
			x4035 = momentum;
			JCudaTensor x4038, x4039;
			x4038 = x4002;
			x4039 = x203;
			x218.backward_filter(x4038, x4039, x4033, x4034, x4035);

			// Dealloc(X4206)
			JCudaTensor x4040;
			x4040 = x4002;
			x4040.free();

			// V_3a2_c_cv_W <~~ X4218 * d_Convolv(1,0)(X8663)/d_3a2_c_cv_W
			float x4042, x4043;
			float x4044;
			float x4045;
			x4044 = 1;
			x4045 = lrn_rate;
			x4042 = x4044 * x4045;
			x4043 = momentum;
			JCudaTensor x4046, x4047;
			x4046 = x4016;
			x4047 = x250;
			x258.backward_filter(x4046, x4047, x4041, x4042, x4043);

			// V_3a1_bn_bias <~~ X8486
			float x4049, x4050;
			float x4051;
			float x4052;
			x4051 = 1;
			x4052 = lrn_rate;
			x4049 = x4051 * x4052;
			x4050 = momentum;
			JCudaTensor x4053;
			x4053 = x4020;
			x4048.update(x4053, x4049, x4050);

			// Dealloc(X8486)
			JCudaTensor x4054;
			x4054 = x4020;
			x4054.free();

			// val X4219 = X4218 * d_Convolv(1,0)(3a2_c_cv_W)/d_X8663
			JCudaTensor x4055;
			JCudaTensor x4056, x4057;
			x4056 = x4016;
			x4057 = x256;
			x4055 = x258.backward_data(x4056, x4057);

			// Dealloc(X4218)
			JCudaTensor x4058;
			x4058 = x4016;
			x4058.free();

			// V_3a2_c_bn_bias <~~ X8495
			float x4060, x4061;
			float x4062;
			float x4063;
			x4062 = 1;
			x4063 = lrn_rate;
			x4060 = x4062 * x4063;
			x4061 = momentum;
			JCudaTensor x4064;
			x4064 = x4007;
			x4059.update(x4064, x4060, x4061);

			// Dealloc(X8495)
			JCudaTensor x4065;
			x4065 = x4007;
			x4065.free();

			// V_3a2_c_bn_scale <~~ X8496
			float x4067, x4068;
			float x4069;
			float x4070;
			x4069 = 1;
			x4070 = lrn_rate;
			x4067 = x4069 * x4070;
			x4068 = momentum;
			JCudaTensor x4071;
			x4071 = x4025;
			x4066.update(x4071, x4067, x4068);

			// Dealloc(X8496)
			JCudaTensor x4072;
			x4072 = x4025;
			x4072.free();

			// V_3a1_bn_scale <~~ X8487
			float x4074, x4075;
			float x4076;
			float x4077;
			x4076 = 1;
			x4077 = lrn_rate;
			x4074 = x4076 * x4077;
			x4075 = momentum;
			JCudaTensor x4078;
			x4078 = x4012;
			x4073.update(x4078, x4074, x4075);

			// Dealloc(X8487)
			JCudaTensor x4079;
			x4079 = x4012;
			x4079.free();

			// 3a1_bn_bias <~~ V_3a1_bn_bias
			float x4080, x4081;
			x4080 = 1;
			float x4082;
			float x4083;
			x4082 = 1;
			float x4084;
			float x4085;
			float x4086;
			float x4087;
			x4086 = 1;
			x4087 = decay;
			x4084 = x4086 * x4087;
			float x4088;
			float x4089;
			x4088 = 1;
			x4089 = lrn_rate;
			x4085 = x4088 * x4089;
			x4083 = x4084 * x4085;
			x4081 = x4082 + x4083;
			JCudaTensor x4090;
			x4090 = x4048;
			x224.update(x4090, x4080, x4081);

			// 3a1_bn_scale <~~ V_3a1_bn_scale
			float x4091, x4092;
			x4091 = 1;
			float x4093;
			float x4094;
			x4093 = 1;
			float x4095;
			float x4096;
			float x4097;
			float x4098;
			x4097 = 1;
			x4098 = decay;
			x4095 = x4097 * x4098;
			float x4099;
			float x4100;
			x4099 = 1;
			x4100 = lrn_rate;
			x4096 = x4099 * x4100;
			x4094 = x4095 * x4096;
			x4092 = x4093 + x4094;
			JCudaTensor x4101;
			x4101 = x4073;
			x223.update(x4101, x4091, x4092);

			// 3a2_c_bn_scale <~~ V_3a2_c_bn_scale
			float x4102, x4103;
			x4102 = 1;
			float x4104;
			float x4105;
			x4104 = 1;
			float x4106;
			float x4107;
			float x4108;
			float x4109;
			x4108 = 1;
			x4109 = decay;
			x4106 = x4108 * x4109;
			float x4110;
			float x4111;
			x4110 = 1;
			x4111 = lrn_rate;
			x4107 = x4110 * x4111;
			x4105 = x4106 * x4107;
			x4103 = x4104 + x4105;
			JCudaTensor x4112;
			x4112 = x4066;
			x263.update(x4112, x4102, x4103);

			// 3a2_c_cv_W <~~ V_3a2_c_cv_W
			float x4113, x4114;
			x4113 = 1;
			float x4115;
			float x4116;
			x4115 = 1;
			float x4117;
			float x4118;
			float x4119;
			float x4120;
			x4119 = 1;
			x4120 = decay;
			x4117 = x4119 * x4120;
			float x4121;
			float x4122;
			x4121 = 1;
			x4122 = lrn_rate;
			x4118 = x4121 * x4122;
			x4116 = x4117 * x4118;
			x4114 = x4115 + x4116;
			JCudaTensor x4123;
			x4123 = x4041;
			x256.update(x4123, x4113, x4114);

			// 3a2_c_bn_bias <~~ V_3a2_c_bn_bias
			float x4124, x4125;
			x4124 = 1;
			float x4126;
			float x4127;
			x4126 = 1;
			float x4128;
			float x4129;
			float x4130;
			float x4131;
			x4130 = 1;
			x4131 = decay;
			x4128 = x4130 * x4131;
			float x4132;
			float x4133;
			x4132 = 1;
			x4133 = lrn_rate;
			x4129 = x4132 * x4133;
			x4127 = x4128 * x4129;
			x4125 = x4126 + x4127;
			JCudaTensor x4134;
			x4134 = x4059;
			x264.update(x4134, x4124, x4125);

			// 3a1_cv_W <~~ V_3a1_cv_W
			float x4135, x4136;
			x4135 = 1;
			float x4137;
			float x4138;
			x4137 = 1;
			float x4139;
			float x4140;
			float x4141;
			float x4142;
			x4141 = 1;
			x4142 = decay;
			x4139 = x4141 * x4142;
			float x4143;
			float x4144;
			x4143 = 1;
			x4144 = lrn_rate;
			x4140 = x4143 * x4144;
			x4138 = x4139 * x4140;
			x4136 = x4137 + x4138;
			JCudaTensor x4145;
			x4145 = x4033;
			x216.update(x4145, x4135, x4136);

			// val X4223 = X4219 * d_ReLU()(X8663)/d_X8662
			JCudaTensor x4146;
			JCudaTensor x4147, x4148;
			x4147 = x4055;
			x4148 = x250;
			x4146 = x235.backward(x4147, x4148);

			// Dealloc(X8663)
			JCudaTensor x4149;
			x4149 = x250;
			x4149.free();

			// val X8493 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_3a2_b_bn_scale
			JCudaTensor x4150;
			JCudaTensor x4151, x4152, x4153;
			x4151 = x4146;
			x4152 = x236;
			x4153 = x247;
			JCudaTensor[] x4154 = x249.backward(x4151,x4152,x4153);
			x4150 = x4154[1];

			// val X4224 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_X8661
			JCudaTensor x4155;
			x4155 = x4154[0];

			// val X8492 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_3a2_b_bn_bias
			JCudaTensor x4159;
			x4159 = x4154[2];

			// Dealloc(X8661)
			JCudaTensor x4163;
			x4163 = x236;
			x4163.free();

			// val X4225 = X4224 * d_Convolv(1,1)(3a2_b_cv_W)/d_X8660
			JCudaTensor x4164;
			JCudaTensor x4165, x4166;
			x4165 = x4155;
			x4166 = x240;
			x4164 = x242.backward_data(x4165, x4166);

			// V_3a2_b_cv_W <~~ X4224 * d_Convolv(1,1)(X8660)/d_3a2_b_cv_W
			float x4168, x4169;
			float x4170;
			float x4171;
			x4170 = 1;
			x4171 = lrn_rate;
			x4168 = x4170 * x4171;
			x4169 = momentum;
			JCudaTensor x4172, x4173;
			x4172 = x4155;
			x4173 = x233;
			x242.backward_filter(x4172, x4173, x4167, x4168, x4169);

			// Dealloc(X4224)
			JCudaTensor x4174;
			x4174 = x4155;
			x4174.free();

			// V_3a2_b_bn_bias <~~ X8492
			float x4176, x4177;
			float x4178;
			float x4179;
			x4178 = 1;
			x4179 = lrn_rate;
			x4176 = x4178 * x4179;
			x4177 = momentum;
			JCudaTensor x4180;
			x4180 = x4159;
			x4175.update(x4180, x4176, x4177);

			// Dealloc(X8492)
			JCudaTensor x4181;
			x4181 = x4159;
			x4181.free();

			// V_3a2_b_bn_scale <~~ X8493
			float x4183, x4184;
			float x4185;
			float x4186;
			x4185 = 1;
			x4186 = lrn_rate;
			x4183 = x4185 * x4186;
			x4184 = momentum;
			JCudaTensor x4187;
			x4187 = x4150;
			x4182.update(x4187, x4183, x4184);

			// Dealloc(X8493)
			JCudaTensor x4188;
			x4188 = x4150;
			x4188.free();

			// 3a2_b_cv_W <~~ V_3a2_b_cv_W
			float x4189, x4190;
			x4189 = 1;
			float x4191;
			float x4192;
			x4191 = 1;
			float x4193;
			float x4194;
			float x4195;
			float x4196;
			x4195 = 1;
			x4196 = decay;
			x4193 = x4195 * x4196;
			float x4197;
			float x4198;
			x4197 = 1;
			x4198 = lrn_rate;
			x4194 = x4197 * x4198;
			x4192 = x4193 * x4194;
			x4190 = x4191 + x4192;
			JCudaTensor x4199;
			x4199 = x4167;
			x240.update(x4199, x4189, x4190);

			// 3a2_b_bn_bias <~~ V_3a2_b_bn_bias
			float x4200, x4201;
			x4200 = 1;
			float x4202;
			float x4203;
			x4202 = 1;
			float x4204;
			float x4205;
			float x4206;
			float x4207;
			x4206 = 1;
			x4207 = decay;
			x4204 = x4206 * x4207;
			float x4208;
			float x4209;
			x4208 = 1;
			x4209 = lrn_rate;
			x4205 = x4208 * x4209;
			x4203 = x4204 * x4205;
			x4201 = x4202 + x4203;
			JCudaTensor x4210;
			x4210 = x4175;
			x248.update(x4210, x4200, x4201);

			// 3a2_b_bn_scale <~~ V_3a2_b_bn_scale
			float x4211, x4212;
			x4211 = 1;
			float x4213;
			float x4214;
			x4213 = 1;
			float x4215;
			float x4216;
			float x4217;
			float x4218;
			x4217 = 1;
			x4218 = decay;
			x4215 = x4217 * x4218;
			float x4219;
			float x4220;
			x4219 = 1;
			x4220 = lrn_rate;
			x4216 = x4219 * x4220;
			x4214 = x4215 * x4216;
			x4212 = x4213 + x4214;
			JCudaTensor x4221;
			x4221 = x4182;
			x247.update(x4221, x4211, x4212);

			// val X4229 = X4225 * d_ReLU()(X8660)/d_X8659
			JCudaTensor x4222;
			JCudaTensor x4223, x4224;
			x4223 = x4164;
			x4224 = x233;
			x4222 = x235.backward(x4223, x4224);

			// Dealloc(X8660)
			JCudaTensor x4225;
			x4225 = x233;
			x4225.free();

			// val X8489 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_3a2_a_bn_bias
			JCudaTensor x4226;
			JCudaTensor x4227, x4228, x4229;
			x4227 = x4222;
			x4228 = x205;
			x4229 = x230;
			JCudaTensor[] x4230 = x232.backward(x4227,x4228,x4229);
			x4226 = x4230[2];

			// val X4230 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_X8658
			JCudaTensor x4231;
			x4231 = x4230[0];

			// val X8490 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_3a2_a_bn_scale
			JCudaTensor x4235;
			x4235 = x4230[1];

			// Dealloc(X8658)
			JCudaTensor x4239;
			x4239 = x205;
			x4239.free();

			// V_3a2_a_bn_bias <~~ X8489
			float x4241, x4242;
			float x4243;
			float x4244;
			x4243 = 1;
			x4244 = lrn_rate;
			x4241 = x4243 * x4244;
			x4242 = momentum;
			JCudaTensor x4245;
			x4245 = x4226;
			x4240.update(x4245, x4241, x4242);

			// Dealloc(X8489)
			JCudaTensor x4246;
			x4246 = x4226;
			x4246.free();

			// val X4232 = (X4207 + X4230 * d_Convolv(2,0)(3a2_a_cv_W)/d_X8654)
			JCudaTensor x4247;
			JCudaTensor x4248;
			x4248 = x4030;
			JCudaTensor x4249, x4250;
			x4249 = x4231;
			x4250 = x209;
			x4247 = x211.backward_data(x4249,x4250, x4248);

			// V_3a2_a_cv_W <~~ X4230 * d_Convolv(2,0)(X8654)/d_3a2_a_cv_W
			float x4252, x4253;
			float x4254;
			float x4255;
			x4254 = 1;
			x4255 = lrn_rate;
			x4252 = x4254 * x4255;
			x4253 = momentum;
			JCudaTensor x4256, x4257;
			x4256 = x4231;
			x4257 = x203;
			x211.backward_filter(x4256, x4257, x4251, x4252, x4253);

			// Dealloc(X4230)
			JCudaTensor x4258;
			x4258 = x4231;
			x4258.free();

			// V_3a2_a_bn_scale <~~ X8490
			float x4260, x4261;
			float x4262;
			float x4263;
			x4262 = 1;
			x4263 = lrn_rate;
			x4260 = x4262 * x4263;
			x4261 = momentum;
			JCudaTensor x4264;
			x4264 = x4235;
			x4259.update(x4264, x4260, x4261);

			// Dealloc(X8490)
			JCudaTensor x4265;
			x4265 = x4235;
			x4265.free();

			// 3a2_a_bn_bias <~~ V_3a2_a_bn_bias
			float x4266, x4267;
			x4266 = 1;
			float x4268;
			float x4269;
			x4268 = 1;
			float x4270;
			float x4271;
			float x4272;
			float x4273;
			x4272 = 1;
			x4273 = decay;
			x4270 = x4272 * x4273;
			float x4274;
			float x4275;
			x4274 = 1;
			x4275 = lrn_rate;
			x4271 = x4274 * x4275;
			x4269 = x4270 * x4271;
			x4267 = x4268 + x4269;
			JCudaTensor x4276;
			x4276 = x4240;
			x231.update(x4276, x4266, x4267);

			// 3a2_a_cv_W <~~ V_3a2_a_cv_W
			float x4277, x4278;
			x4277 = 1;
			float x4279;
			float x4280;
			x4279 = 1;
			float x4281;
			float x4282;
			float x4283;
			float x4284;
			x4283 = 1;
			x4284 = decay;
			x4281 = x4283 * x4284;
			float x4285;
			float x4286;
			x4285 = 1;
			x4286 = lrn_rate;
			x4282 = x4285 * x4286;
			x4280 = x4281 * x4282;
			x4278 = x4279 + x4280;
			JCudaTensor x4287;
			x4287 = x4251;
			x209.update(x4287, x4277, x4278);

			// 3a2_a_bn_scale <~~ V_3a2_a_bn_scale
			float x4288, x4289;
			x4288 = 1;
			float x4290;
			float x4291;
			x4290 = 1;
			float x4292;
			float x4293;
			float x4294;
			float x4295;
			x4294 = 1;
			x4295 = decay;
			x4292 = x4294 * x4295;
			float x4296;
			float x4297;
			x4296 = 1;
			x4297 = lrn_rate;
			x4293 = x4296 * x4297;
			x4291 = x4292 * x4293;
			x4289 = x4290 + x4291;
			JCudaTensor x4298;
			x4298 = x4259;
			x230.update(x4298, x4288, x4289);

			// val X4269 = X4232 * d_ReLU()(X8654)/d_X8653
			JCudaTensor x4299;
			JCudaTensor x4300, x4301;
			x4300 = x4247;
			x4301 = x203;
			x4299 = x96.backward(x4300, x4301);

			// Dealloc(X8654)
			JCudaTensor x4302;
			x4302 = x203;
			x4302.free();

			// val X4279 = X4269.copy * d_ReLU()(X8652)/d_X8651
			JCudaTensor x4303;
			JCudaTensor x4304, x4305;
			x4304 = x4299;
			x4304 = x4304.clone();
			x4305 = x198;
			x4303 = x96.backward(x4304, x4305);

			// Dealloc(X8652)
			JCudaTensor x4306;
			x4306 = x198;
			x4306.free();

			// val X8483 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_2c_c_bn_bias
			JCudaTensor x4307;
			JCudaTensor x4308, x4309, x4310;
			x4308 = x4303;
			x4309 = x185;
			x4310 = x195;
			JCudaTensor[] x4311 = x197.backward(x4308,x4309,x4310);
			x4307 = x4311[2];

			// val X8484 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_2c_c_bn_scale
			JCudaTensor x4312;
			x4312 = x4311[1];

			// val X4280 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_X8650
			JCudaTensor x4316;
			x4316 = x4311[0];

			// Dealloc(X8650)
			JCudaTensor x4320;
			x4320 = x185;
			x4320.free();

			// val X4281 = X4280 * d_Convolv(1,0)(2c_c_cv_W)/d_X8649
			JCudaTensor x4321;
			JCudaTensor x4322, x4323;
			x4322 = x4316;
			x4323 = x189;
			x4321 = x47.backward_data(x4322, x4323);

			// V_2c_c_bn_scale <~~ X8484
			float x4325, x4326;
			float x4327;
			float x4328;
			x4327 = 1;
			x4328 = lrn_rate;
			x4325 = x4327 * x4328;
			x4326 = momentum;
			JCudaTensor x4329;
			x4329 = x4312;
			x4324.update(x4329, x4325, x4326);

			// Dealloc(X8484)
			JCudaTensor x4330;
			x4330 = x4312;
			x4330.free();

			// V_2c_c_bn_bias <~~ X8483
			float x4332, x4333;
			float x4334;
			float x4335;
			x4334 = 1;
			x4335 = lrn_rate;
			x4332 = x4334 * x4335;
			x4333 = momentum;
			JCudaTensor x4336;
			x4336 = x4307;
			x4331.update(x4336, x4332, x4333);

			// Dealloc(X8483)
			JCudaTensor x4337;
			x4337 = x4307;
			x4337.free();

			// V_2c_c_cv_W <~~ X4280 * d_Convolv(1,0)(X8649)/d_2c_c_cv_W
			float x4339, x4340;
			float x4341;
			float x4342;
			x4341 = 1;
			x4342 = lrn_rate;
			x4339 = x4341 * x4342;
			x4340 = momentum;
			JCudaTensor x4343, x4344;
			x4343 = x4316;
			x4344 = x183;
			x47.backward_filter(x4343, x4344, x4338, x4339, x4340);

			// Dealloc(X4280)
			JCudaTensor x4345;
			x4345 = x4316;
			x4345.free();

			// 2c_c_bn_scale <~~ V_2c_c_bn_scale
			float x4346, x4347;
			x4346 = 1;
			float x4348;
			float x4349;
			x4348 = 1;
			float x4350;
			float x4351;
			float x4352;
			float x4353;
			x4352 = 1;
			x4353 = decay;
			x4350 = x4352 * x4353;
			float x4354;
			float x4355;
			x4354 = 1;
			x4355 = lrn_rate;
			x4351 = x4354 * x4355;
			x4349 = x4350 * x4351;
			x4347 = x4348 + x4349;
			JCudaTensor x4356;
			x4356 = x4324;
			x195.update(x4356, x4346, x4347);

			// 2c_c_bn_bias <~~ V_2c_c_bn_bias
			float x4357, x4358;
			x4357 = 1;
			float x4359;
			float x4360;
			x4359 = 1;
			float x4361;
			float x4362;
			float x4363;
			float x4364;
			x4363 = 1;
			x4364 = decay;
			x4361 = x4363 * x4364;
			float x4365;
			float x4366;
			x4365 = 1;
			x4366 = lrn_rate;
			x4362 = x4365 * x4366;
			x4360 = x4361 * x4362;
			x4358 = x4359 + x4360;
			JCudaTensor x4367;
			x4367 = x4331;
			x196.update(x4367, x4357, x4358);

			// 2c_c_cv_W <~~ V_2c_c_cv_W
			float x4368, x4369;
			x4368 = 1;
			float x4370;
			float x4371;
			x4370 = 1;
			float x4372;
			float x4373;
			float x4374;
			float x4375;
			x4374 = 1;
			x4375 = decay;
			x4372 = x4374 * x4375;
			float x4376;
			float x4377;
			x4376 = 1;
			x4377 = lrn_rate;
			x4373 = x4376 * x4377;
			x4371 = x4372 * x4373;
			x4369 = x4370 + x4371;
			JCudaTensor x4378;
			x4378 = x4338;
			x189.update(x4378, x4368, x4369);

			// val X4285 = X4281 * d_ReLU()(X8649)/d_X8648
			JCudaTensor x4379;
			JCudaTensor x4380, x4381;
			x4380 = x4321;
			x4381 = x183;
			x4379 = x64.backward(x4380, x4381);

			// Dealloc(X8649)
			JCudaTensor x4382;
			x4382 = x183;
			x4382.free();

			// val X4286 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_X8647
			JCudaTensor x4383;
			JCudaTensor x4384, x4385, x4386;
			x4384 = x4379;
			x4385 = x170;
			x4386 = x180;
			JCudaTensor[] x4387 = x182.backward(x4384,x4385,x4386);
			x4383 = x4387[0];

			// val X8481 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_2c_b_bn_scale
			JCudaTensor x4388;
			x4388 = x4387[1];

			// val X8480 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_2c_b_bn_bias
			JCudaTensor x4392;
			x4392 = x4387[2];

			// Dealloc(X8647)
			JCudaTensor x4396;
			x4396 = x170;
			x4396.free();

			// val X4287 = X4286 * d_Convolv(1,1)(2c_b_cv_W)/d_X8646
			JCudaTensor x4397;
			JCudaTensor x4398, x4399;
			x4398 = x4383;
			x4399 = x174;
			x4397 = x71.backward_data(x4398, x4399);

			// V_2c_b_bn_bias <~~ X8480
			float x4401, x4402;
			float x4403;
			float x4404;
			x4403 = 1;
			x4404 = lrn_rate;
			x4401 = x4403 * x4404;
			x4402 = momentum;
			JCudaTensor x4405;
			x4405 = x4392;
			x4400.update(x4405, x4401, x4402);

			// Dealloc(X8480)
			JCudaTensor x4406;
			x4406 = x4392;
			x4406.free();

			// V_2c_b_cv_W <~~ X4286 * d_Convolv(1,1)(X8646)/d_2c_b_cv_W
			float x4408, x4409;
			float x4410;
			float x4411;
			x4410 = 1;
			x4411 = lrn_rate;
			x4408 = x4410 * x4411;
			x4409 = momentum;
			JCudaTensor x4412, x4413;
			x4412 = x4383;
			x4413 = x168;
			x71.backward_filter(x4412, x4413, x4407, x4408, x4409);

			// Dealloc(X4286)
			JCudaTensor x4414;
			x4414 = x4383;
			x4414.free();

			// V_2c_b_bn_scale <~~ X8481
			float x4416, x4417;
			float x4418;
			float x4419;
			x4418 = 1;
			x4419 = lrn_rate;
			x4416 = x4418 * x4419;
			x4417 = momentum;
			JCudaTensor x4420;
			x4420 = x4388;
			x4415.update(x4420, x4416, x4417);

			// Dealloc(X8481)
			JCudaTensor x4421;
			x4421 = x4388;
			x4421.free();

			// 2c_b_bn_bias <~~ V_2c_b_bn_bias
			float x4422, x4423;
			x4422 = 1;
			float x4424;
			float x4425;
			x4424 = 1;
			float x4426;
			float x4427;
			float x4428;
			float x4429;
			x4428 = 1;
			x4429 = decay;
			x4426 = x4428 * x4429;
			float x4430;
			float x4431;
			x4430 = 1;
			x4431 = lrn_rate;
			x4427 = x4430 * x4431;
			x4425 = x4426 * x4427;
			x4423 = x4424 + x4425;
			JCudaTensor x4432;
			x4432 = x4400;
			x181.update(x4432, x4422, x4423);

			// 2c_b_cv_W <~~ V_2c_b_cv_W
			float x4433, x4434;
			x4433 = 1;
			float x4435;
			float x4436;
			x4435 = 1;
			float x4437;
			float x4438;
			float x4439;
			float x4440;
			x4439 = 1;
			x4440 = decay;
			x4437 = x4439 * x4440;
			float x4441;
			float x4442;
			x4441 = 1;
			x4442 = lrn_rate;
			x4438 = x4441 * x4442;
			x4436 = x4437 * x4438;
			x4434 = x4435 + x4436;
			JCudaTensor x4443;
			x4443 = x4407;
			x174.update(x4443, x4433, x4434);

			// 2c_b_bn_scale <~~ V_2c_b_bn_scale
			float x4444, x4445;
			x4444 = 1;
			float x4446;
			float x4447;
			x4446 = 1;
			float x4448;
			float x4449;
			float x4450;
			float x4451;
			x4450 = 1;
			x4451 = decay;
			x4448 = x4450 * x4451;
			float x4452;
			float x4453;
			x4452 = 1;
			x4453 = lrn_rate;
			x4449 = x4452 * x4453;
			x4447 = x4448 * x4449;
			x4445 = x4446 + x4447;
			JCudaTensor x4454;
			x4454 = x4415;
			x180.update(x4454, x4444, x4445);

			// val X4291 = X4287 * d_ReLU()(X8646)/d_X8645
			JCudaTensor x4455;
			JCudaTensor x4456, x4457;
			x4456 = x4397;
			x4457 = x168;
			x4455 = x64.backward(x4456, x4457);

			// Dealloc(X8646)
			JCudaTensor x4458;
			x4458 = x168;
			x4458.free();

			// val X8477 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_2c_a_bn_bias
			JCudaTensor x4459;
			JCudaTensor x4460, x4461, x4462;
			x4460 = x4455;
			x4461 = x155;
			x4462 = x165;
			JCudaTensor[] x4463 = x167.backward(x4460,x4461,x4462);
			x4459 = x4463[2];

			// val X4292 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_X8644
			JCudaTensor x4464;
			x4464 = x4463[0];

			// val X8478 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_2c_a_bn_scale
			JCudaTensor x4468;
			x4468 = x4463[1];

			// Dealloc(X8644)
			JCudaTensor x4472;
			x4472 = x155;
			x4472.free();

			// val X4293 = X4292 * d_Convolv(1,0)(2c_a_cv_W)/d_X8643
			JCudaTensor x4473;
			JCudaTensor x4474, x4475;
			x4474 = x4464;
			x4475 = x159;
			x4473 = x110.backward_data(x4474, x4475);

			// V_2c_a_cv_W <~~ X4292 * d_Convolv(1,0)(X8643)/d_2c_a_cv_W
			float x4477, x4478;
			float x4479;
			float x4480;
			x4479 = 1;
			x4480 = lrn_rate;
			x4477 = x4479 * x4480;
			x4478 = momentum;
			JCudaTensor x4481, x4482;
			x4481 = x4464;
			x4482 = x153;
			x110.backward_filter(x4481, x4482, x4476, x4477, x4478);

			// Dealloc(X4292)
			JCudaTensor x4483;
			x4483 = x4464;
			x4483.free();

			// V_2c_a_bn_bias <~~ X8477
			float x4485, x4486;
			float x4487;
			float x4488;
			x4487 = 1;
			x4488 = lrn_rate;
			x4485 = x4487 * x4488;
			x4486 = momentum;
			JCudaTensor x4489;
			x4489 = x4459;
			x4484.update(x4489, x4485, x4486);

			// Dealloc(X8477)
			JCudaTensor x4490;
			x4490 = x4459;
			x4490.free();

			// V_2c_a_bn_scale <~~ X8478
			float x4492, x4493;
			float x4494;
			float x4495;
			x4494 = 1;
			x4495 = lrn_rate;
			x4492 = x4494 * x4495;
			x4493 = momentum;
			JCudaTensor x4496;
			x4496 = x4468;
			x4491.update(x4496, x4492, x4493);

			// Dealloc(X8478)
			JCudaTensor x4497;
			x4497 = x4468;
			x4497.free();

			// 2c_a_cv_W <~~ V_2c_a_cv_W
			float x4498, x4499;
			x4498 = 1;
			float x4500;
			float x4501;
			x4500 = 1;
			float x4502;
			float x4503;
			float x4504;
			float x4505;
			x4504 = 1;
			x4505 = decay;
			x4502 = x4504 * x4505;
			float x4506;
			float x4507;
			x4506 = 1;
			x4507 = lrn_rate;
			x4503 = x4506 * x4507;
			x4501 = x4502 * x4503;
			x4499 = x4500 + x4501;
			JCudaTensor x4508;
			x4508 = x4476;
			x159.update(x4508, x4498, x4499);

			// 2c_a_bn_bias <~~ V_2c_a_bn_bias
			float x4509, x4510;
			x4509 = 1;
			float x4511;
			float x4512;
			x4511 = 1;
			float x4513;
			float x4514;
			float x4515;
			float x4516;
			x4515 = 1;
			x4516 = decay;
			x4513 = x4515 * x4516;
			float x4517;
			float x4518;
			x4517 = 1;
			x4518 = lrn_rate;
			x4514 = x4517 * x4518;
			x4512 = x4513 * x4514;
			x4510 = x4511 + x4512;
			JCudaTensor x4519;
			x4519 = x4484;
			x166.update(x4519, x4509, x4510);

			// 2c_a_bn_scale <~~ V_2c_a_bn_scale
			float x4520, x4521;
			x4520 = 1;
			float x4522;
			float x4523;
			x4522 = 1;
			float x4524;
			float x4525;
			float x4526;
			float x4527;
			x4526 = 1;
			x4527 = decay;
			x4524 = x4526 * x4527;
			float x4528;
			float x4529;
			x4528 = 1;
			x4529 = lrn_rate;
			x4525 = x4528 * x4529;
			x4523 = x4524 * x4525;
			x4521 = x4522 + x4523;
			JCudaTensor x4530;
			x4530 = x4491;
			x165.update(x4530, x4520, x4521);

			// val X4294 = (X4293 + X4269)
			JCudaTensor x4531;
			JCudaTensor x4532, x4533;
			x4532 = x4473;
			x4533 = x4299;
			x4531 = x4532.plus_i(x4533);

			// Dealloc(X4269)
			JCudaTensor x4534;
			x4534 = x4299;
			x4534.free();

			// val X4306 = X4294 * d_ReLU()(X8643)/d_X8642
			JCudaTensor x4535;
			JCudaTensor x4536, x4537;
			x4536 = x4531;
			x4537 = x153;
			x4535 = x96.backward(x4536, x4537);

			// Dealloc(X8643)
			JCudaTensor x4538;
			x4538 = x153;
			x4538.free();

			// val X4316 = X4306.copy * d_ReLU()(X8641)/d_X8640
			JCudaTensor x4539;
			JCudaTensor x4540, x4541;
			x4540 = x4535;
			x4540 = x4540.clone();
			x4541 = x148;
			x4539 = x96.backward(x4540, x4541);

			// Dealloc(X8641)
			JCudaTensor x4542;
			x4542 = x148;
			x4542.free();

			// val X4317 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_X8639
			JCudaTensor x4543;
			JCudaTensor x4544, x4545, x4546;
			x4544 = x4539;
			x4545 = x135;
			x4546 = x145;
			JCudaTensor[] x4547 = x147.backward(x4544,x4545,x4546);
			x4543 = x4547[0];

			// val X8474 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_2b_c_bn_bias
			JCudaTensor x4548;
			x4548 = x4547[2];

			// val X8475 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_2b_c_bn_scale
			JCudaTensor x4552;
			x4552 = x4547[1];

			// Dealloc(X8639)
			JCudaTensor x4556;
			x4556 = x135;
			x4556.free();

			// V_2b_c_bn_scale <~~ X8475
			float x4558, x4559;
			float x4560;
			float x4561;
			x4560 = 1;
			x4561 = lrn_rate;
			x4558 = x4560 * x4561;
			x4559 = momentum;
			JCudaTensor x4562;
			x4562 = x4552;
			x4557.update(x4562, x4558, x4559);

			// Dealloc(X8475)
			JCudaTensor x4563;
			x4563 = x4552;
			x4563.free();

			// val X4318 = X4317 * d_Convolv(1,0)(2b_c_cv_W)/d_X8638
			JCudaTensor x4564;
			JCudaTensor x4565, x4566;
			x4565 = x4543;
			x4566 = x139;
			x4564 = x47.backward_data(x4565, x4566);

			// V_2b_c_cv_W <~~ X4317 * d_Convolv(1,0)(X8638)/d_2b_c_cv_W
			float x4568, x4569;
			float x4570;
			float x4571;
			x4570 = 1;
			x4571 = lrn_rate;
			x4568 = x4570 * x4571;
			x4569 = momentum;
			JCudaTensor x4572, x4573;
			x4572 = x4543;
			x4573 = x133;
			x47.backward_filter(x4572, x4573, x4567, x4568, x4569);

			// Dealloc(X4317)
			JCudaTensor x4574;
			x4574 = x4543;
			x4574.free();

			// V_2b_c_bn_bias <~~ X8474
			float x4576, x4577;
			float x4578;
			float x4579;
			x4578 = 1;
			x4579 = lrn_rate;
			x4576 = x4578 * x4579;
			x4577 = momentum;
			JCudaTensor x4580;
			x4580 = x4548;
			x4575.update(x4580, x4576, x4577);

			// Dealloc(X8474)
			JCudaTensor x4581;
			x4581 = x4548;
			x4581.free();

			// 2b_c_bn_scale <~~ V_2b_c_bn_scale
			float x4582, x4583;
			x4582 = 1;
			float x4584;
			float x4585;
			x4584 = 1;
			float x4586;
			float x4587;
			float x4588;
			float x4589;
			x4588 = 1;
			x4589 = decay;
			x4586 = x4588 * x4589;
			float x4590;
			float x4591;
			x4590 = 1;
			x4591 = lrn_rate;
			x4587 = x4590 * x4591;
			x4585 = x4586 * x4587;
			x4583 = x4584 + x4585;
			JCudaTensor x4592;
			x4592 = x4557;
			x145.update(x4592, x4582, x4583);

			// 2b_c_cv_W <~~ V_2b_c_cv_W
			float x4593, x4594;
			x4593 = 1;
			float x4595;
			float x4596;
			x4595 = 1;
			float x4597;
			float x4598;
			float x4599;
			float x4600;
			x4599 = 1;
			x4600 = decay;
			x4597 = x4599 * x4600;
			float x4601;
			float x4602;
			x4601 = 1;
			x4602 = lrn_rate;
			x4598 = x4601 * x4602;
			x4596 = x4597 * x4598;
			x4594 = x4595 + x4596;
			JCudaTensor x4603;
			x4603 = x4567;
			x139.update(x4603, x4593, x4594);

			// 2b_c_bn_bias <~~ V_2b_c_bn_bias
			float x4604, x4605;
			x4604 = 1;
			float x4606;
			float x4607;
			x4606 = 1;
			float x4608;
			float x4609;
			float x4610;
			float x4611;
			x4610 = 1;
			x4611 = decay;
			x4608 = x4610 * x4611;
			float x4612;
			float x4613;
			x4612 = 1;
			x4613 = lrn_rate;
			x4609 = x4612 * x4613;
			x4607 = x4608 * x4609;
			x4605 = x4606 + x4607;
			JCudaTensor x4614;
			x4614 = x4575;
			x146.update(x4614, x4604, x4605);

			// val X4322 = X4318 * d_ReLU()(X8638)/d_X8637
			JCudaTensor x4615;
			JCudaTensor x4616, x4617;
			x4616 = x4564;
			x4617 = x133;
			x4615 = x64.backward(x4616, x4617);

			// Dealloc(X8638)
			JCudaTensor x4618;
			x4618 = x133;
			x4618.free();

			// val X4323 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_X8636
			JCudaTensor x4619;
			JCudaTensor x4620, x4621, x4622;
			x4620 = x4615;
			x4621 = x120;
			x4622 = x130;
			JCudaTensor[] x4623 = x132.backward(x4620,x4621,x4622);
			x4619 = x4623[0];

			// val X8472 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_2b_b_bn_scale
			JCudaTensor x4624;
			x4624 = x4623[1];

			// val X8471 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_2b_b_bn_bias
			JCudaTensor x4628;
			x4628 = x4623[2];

			// Dealloc(X8636)
			JCudaTensor x4632;
			x4632 = x120;
			x4632.free();

			// V_2b_b_bn_scale <~~ X8472
			float x4634, x4635;
			float x4636;
			float x4637;
			x4636 = 1;
			x4637 = lrn_rate;
			x4634 = x4636 * x4637;
			x4635 = momentum;
			JCudaTensor x4638;
			x4638 = x4624;
			x4633.update(x4638, x4634, x4635);

			// Dealloc(X8472)
			JCudaTensor x4639;
			x4639 = x4624;
			x4639.free();

			// V_2b_b_bn_bias <~~ X8471
			float x4641, x4642;
			float x4643;
			float x4644;
			x4643 = 1;
			x4644 = lrn_rate;
			x4641 = x4643 * x4644;
			x4642 = momentum;
			JCudaTensor x4645;
			x4645 = x4628;
			x4640.update(x4645, x4641, x4642);

			// Dealloc(X8471)
			JCudaTensor x4646;
			x4646 = x4628;
			x4646.free();

			// val X4324 = X4323 * d_Convolv(1,1)(2b_b_cv_W)/d_X8635
			JCudaTensor x4647;
			JCudaTensor x4648, x4649;
			x4648 = x4619;
			x4649 = x124;
			x4647 = x71.backward_data(x4648, x4649);

			// V_2b_b_cv_W <~~ X4323 * d_Convolv(1,1)(X8635)/d_2b_b_cv_W
			float x4651, x4652;
			float x4653;
			float x4654;
			x4653 = 1;
			x4654 = lrn_rate;
			x4651 = x4653 * x4654;
			x4652 = momentum;
			JCudaTensor x4655, x4656;
			x4655 = x4619;
			x4656 = x118;
			x71.backward_filter(x4655, x4656, x4650, x4651, x4652);

			// Dealloc(X4323)
			JCudaTensor x4657;
			x4657 = x4619;
			x4657.free();

			// 2b_b_bn_scale <~~ V_2b_b_bn_scale
			float x4658, x4659;
			x4658 = 1;
			float x4660;
			float x4661;
			x4660 = 1;
			float x4662;
			float x4663;
			float x4664;
			float x4665;
			x4664 = 1;
			x4665 = decay;
			x4662 = x4664 * x4665;
			float x4666;
			float x4667;
			x4666 = 1;
			x4667 = lrn_rate;
			x4663 = x4666 * x4667;
			x4661 = x4662 * x4663;
			x4659 = x4660 + x4661;
			JCudaTensor x4668;
			x4668 = x4633;
			x130.update(x4668, x4658, x4659);

			// 2b_b_bn_bias <~~ V_2b_b_bn_bias
			float x4669, x4670;
			x4669 = 1;
			float x4671;
			float x4672;
			x4671 = 1;
			float x4673;
			float x4674;
			float x4675;
			float x4676;
			x4675 = 1;
			x4676 = decay;
			x4673 = x4675 * x4676;
			float x4677;
			float x4678;
			x4677 = 1;
			x4678 = lrn_rate;
			x4674 = x4677 * x4678;
			x4672 = x4673 * x4674;
			x4670 = x4671 + x4672;
			JCudaTensor x4679;
			x4679 = x4640;
			x131.update(x4679, x4669, x4670);

			// 2b_b_cv_W <~~ V_2b_b_cv_W
			float x4680, x4681;
			x4680 = 1;
			float x4682;
			float x4683;
			x4682 = 1;
			float x4684;
			float x4685;
			float x4686;
			float x4687;
			x4686 = 1;
			x4687 = decay;
			x4684 = x4686 * x4687;
			float x4688;
			float x4689;
			x4688 = 1;
			x4689 = lrn_rate;
			x4685 = x4688 * x4689;
			x4683 = x4684 * x4685;
			x4681 = x4682 + x4683;
			JCudaTensor x4690;
			x4690 = x4650;
			x124.update(x4690, x4680, x4681);

			// val X4328 = X4324 * d_ReLU()(X8635)/d_X8634
			JCudaTensor x4691;
			JCudaTensor x4692, x4693;
			x4692 = x4647;
			x4693 = x118;
			x4691 = x64.backward(x4692, x4693);

			// Dealloc(X8635)
			JCudaTensor x4694;
			x4694 = x118;
			x4694.free();

			// val X4329 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_X8633
			JCudaTensor x4695;
			JCudaTensor x4696, x4697, x4698;
			x4696 = x4691;
			x4697 = x104;
			x4698 = x115;
			JCudaTensor[] x4699 = x117.backward(x4696,x4697,x4698);
			x4695 = x4699[0];

			// val X8469 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_2b_a_bn_scale
			JCudaTensor x4700;
			x4700 = x4699[1];

			// val X8468 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_2b_a_bn_bias
			JCudaTensor x4704;
			x4704 = x4699[2];

			// Dealloc(X8633)
			JCudaTensor x4708;
			x4708 = x104;
			x4708.free();

			// val X4330 = X4329 * d_Convolv(1,0)(2b_a_cv_W)/d_X8632
			JCudaTensor x4709;
			JCudaTensor x4710, x4711;
			x4710 = x4695;
			x4711 = x108;
			x4709 = x110.backward_data(x4710, x4711);

			// V_2b_a_cv_W <~~ X4329 * d_Convolv(1,0)(X8632)/d_2b_a_cv_W
			float x4713, x4714;
			float x4715;
			float x4716;
			x4715 = 1;
			x4716 = lrn_rate;
			x4713 = x4715 * x4716;
			x4714 = momentum;
			JCudaTensor x4717, x4718;
			x4717 = x4695;
			x4718 = x102;
			x110.backward_filter(x4717, x4718, x4712, x4713, x4714);

			// Dealloc(X4329)
			JCudaTensor x4719;
			x4719 = x4695;
			x4719.free();

			// V_2b_a_bn_scale <~~ X8469
			float x4721, x4722;
			float x4723;
			float x4724;
			x4723 = 1;
			x4724 = lrn_rate;
			x4721 = x4723 * x4724;
			x4722 = momentum;
			JCudaTensor x4725;
			x4725 = x4700;
			x4720.update(x4725, x4721, x4722);

			// Dealloc(X8469)
			JCudaTensor x4726;
			x4726 = x4700;
			x4726.free();

			// V_2b_a_bn_bias <~~ X8468
			float x4728, x4729;
			float x4730;
			float x4731;
			x4730 = 1;
			x4731 = lrn_rate;
			x4728 = x4730 * x4731;
			x4729 = momentum;
			JCudaTensor x4732;
			x4732 = x4704;
			x4727.update(x4732, x4728, x4729);

			// Dealloc(X8468)
			JCudaTensor x4733;
			x4733 = x4704;
			x4733.free();

			// 2b_a_cv_W <~~ V_2b_a_cv_W
			float x4734, x4735;
			x4734 = 1;
			float x4736;
			float x4737;
			x4736 = 1;
			float x4738;
			float x4739;
			float x4740;
			float x4741;
			x4740 = 1;
			x4741 = decay;
			x4738 = x4740 * x4741;
			float x4742;
			float x4743;
			x4742 = 1;
			x4743 = lrn_rate;
			x4739 = x4742 * x4743;
			x4737 = x4738 * x4739;
			x4735 = x4736 + x4737;
			JCudaTensor x4744;
			x4744 = x4712;
			x108.update(x4744, x4734, x4735);

			// 2b_a_bn_scale <~~ V_2b_a_bn_scale
			float x4745, x4746;
			x4745 = 1;
			float x4747;
			float x4748;
			x4747 = 1;
			float x4749;
			float x4750;
			float x4751;
			float x4752;
			x4751 = 1;
			x4752 = decay;
			x4749 = x4751 * x4752;
			float x4753;
			float x4754;
			x4753 = 1;
			x4754 = lrn_rate;
			x4750 = x4753 * x4754;
			x4748 = x4749 * x4750;
			x4746 = x4747 + x4748;
			JCudaTensor x4755;
			x4755 = x4720;
			x115.update(x4755, x4745, x4746);

			// 2b_a_bn_bias <~~ V_2b_a_bn_bias
			float x4756, x4757;
			x4756 = 1;
			float x4758;
			float x4759;
			x4758 = 1;
			float x4760;
			float x4761;
			float x4762;
			float x4763;
			x4762 = 1;
			x4763 = decay;
			x4760 = x4762 * x4763;
			float x4764;
			float x4765;
			x4764 = 1;
			x4765 = lrn_rate;
			x4761 = x4764 * x4765;
			x4759 = x4760 * x4761;
			x4757 = x4758 + x4759;
			JCudaTensor x4766;
			x4766 = x4727;
			x116.update(x4766, x4756, x4757);

			// val X4331 = (X4330 + X4306)
			JCudaTensor x4767;
			JCudaTensor x4768, x4769;
			x4768 = x4709;
			x4769 = x4535;
			x4767 = x4768.plus_i(x4769);

			// Dealloc(X4306)
			JCudaTensor x4770;
			x4770 = x4535;
			x4770.free();

			// val X4346 = X4331 * d_ReLU()(X8632)/d_X8631
			JCudaTensor x4771;
			JCudaTensor x4772, x4773;
			x4772 = x4767;
			x4773 = x102;
			x4771 = x96.backward(x4772, x4773);

			// Dealloc(X8632)
			JCudaTensor x4774;
			x4774 = x102;
			x4774.free();

			// val X4362 = X4346.copy * d_ReLU()(X8630)/d_X8629
			JCudaTensor x4775;
			JCudaTensor x4776, x4777;
			x4776 = x4771;
			x4776 = x4776.clone();
			x4777 = x97;
			x4775 = x96.backward(x4776, x4777);

			// Dealloc(X8630)
			JCudaTensor x4778;
			x4778 = x97;
			x4778.free();

			// val X4350 = X4346.copy * d_ReLU()(X8621)/d_X8620
			JCudaTensor x4779;
			JCudaTensor x4780, x4781;
			x4780 = x4771;
			x4780 = x4780.clone();
			x4781 = x94;
			x4779 = x96.backward(x4780, x4781);

			// Dealloc(X4346)
			JCudaTensor x4782;
			x4782 = x4771;
			x4782.free();

			// Dealloc(X8621)
			JCudaTensor x4783;
			x4783 = x94;
			x4783.free();

			// val X8465 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_2a2_c_bn_bias
			JCudaTensor x4784;
			JCudaTensor x4785, x4786, x4787;
			x4785 = x4775;
			x4786 = x81;
			x4787 = x91;
			JCudaTensor[] x4788 = x93.backward(x4785,x4786,x4787);
			x4784 = x4788[2];

			// val X4351 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_X8619
			JCudaTensor x4789;
			JCudaTensor x4790, x4791, x4792;
			x4790 = x4779;
			x4791 = x41;
			x4792 = x59;
			JCudaTensor[] x4793 = x61.backward(x4790,x4791,x4792);
			x4789 = x4793[0];

			// val X4363 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_X8628
			JCudaTensor x4794;
			x4794 = x4788[0];

			// val X8456 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_2a1_bn_bias
			JCudaTensor x4798;
			x4798 = x4793[2];

			// val X8466 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_2a2_c_bn_scale
			JCudaTensor x4802;
			x4802 = x4788[1];

			// Dealloc(X8628)
			JCudaTensor x4806;
			x4806 = x81;
			x4806.free();

			// val X8457 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_2a1_bn_scale
			JCudaTensor x4807;
			x4807 = x4793[1];

			// Dealloc(X8619)
			JCudaTensor x4811;
			x4811 = x41;
			x4811.free();

			// val X4364 = X4363 * d_Convolv(1,0)(2a2_c_cv_W)/d_X8627
			JCudaTensor x4812;
			JCudaTensor x4813, x4814;
			x4813 = x4794;
			x4814 = x85;
			x4812 = x47.backward_data(x4813, x4814);

			// V_2a1_bn_scale <~~ X8457
			float x4816, x4817;
			float x4818;
			float x4819;
			x4818 = 1;
			x4819 = lrn_rate;
			x4816 = x4818 * x4819;
			x4817 = momentum;
			JCudaTensor x4820;
			x4820 = x4807;
			x4815.update(x4820, x4816, x4817);

			// Dealloc(X8457)
			JCudaTensor x4821;
			x4821 = x4807;
			x4821.free();

			// V_2a1_bn_bias <~~ X8456
			float x4823, x4824;
			float x4825;
			float x4826;
			x4825 = 1;
			x4826 = lrn_rate;
			x4823 = x4825 * x4826;
			x4824 = momentum;
			JCudaTensor x4827;
			x4827 = x4798;
			x4822.update(x4827, x4823, x4824);

			// Dealloc(X8456)
			JCudaTensor x4828;
			x4828 = x4798;
			x4828.free();

			// V_2a1_cv_W <~~ X4351 * d_Convolv(1,0)(X8618)/d_2a1_cv_W
			float x4830, x4831;
			float x4832;
			float x4833;
			x4832 = 1;
			x4833 = lrn_rate;
			x4830 = x4832 * x4833;
			x4831 = momentum;
			JCudaTensor x4834, x4835;
			x4834 = x4789;
			x4835 = x31;
			x47.backward_filter(x4834, x4835, x4829, x4830, x4831);

			// val X4352 = X4351 * d_Convolv(1,0)(2a1_cv_W)/d_X8618
			JCudaTensor x4836;
			JCudaTensor x4837, x4838;
			x4837 = x4789;
			x4838 = x45;
			x4836 = x47.backward_data(x4837, x4838);

			// Dealloc(X4351)
			JCudaTensor x4839;
			x4839 = x4789;
			x4839.free();

			// V_2a2_c_bn_scale <~~ X8466
			float x4841, x4842;
			float x4843;
			float x4844;
			x4843 = 1;
			x4844 = lrn_rate;
			x4841 = x4843 * x4844;
			x4842 = momentum;
			JCudaTensor x4845;
			x4845 = x4802;
			x4840.update(x4845, x4841, x4842);

			// Dealloc(X8466)
			JCudaTensor x4846;
			x4846 = x4802;
			x4846.free();

			// V_2a2_c_bn_bias <~~ X8465
			float x4848, x4849;
			float x4850;
			float x4851;
			x4850 = 1;
			x4851 = lrn_rate;
			x4848 = x4850 * x4851;
			x4849 = momentum;
			JCudaTensor x4852;
			x4852 = x4784;
			x4847.update(x4852, x4848, x4849);

			// Dealloc(X8465)
			JCudaTensor x4853;
			x4853 = x4784;
			x4853.free();

			// V_2a2_c_cv_W <~~ X4363 * d_Convolv(1,0)(X8627)/d_2a2_c_cv_W
			float x4855, x4856;
			float x4857;
			float x4858;
			x4857 = 1;
			x4858 = lrn_rate;
			x4855 = x4857 * x4858;
			x4856 = momentum;
			JCudaTensor x4859, x4860;
			x4859 = x4794;
			x4860 = x79;
			x47.backward_filter(x4859, x4860, x4854, x4855, x4856);

			// Dealloc(X4363)
			JCudaTensor x4861;
			x4861 = x4794;
			x4861.free();

			// 2a1_bn_bias <~~ V_2a1_bn_bias
			float x4862, x4863;
			x4862 = 1;
			float x4864;
			float x4865;
			x4864 = 1;
			float x4866;
			float x4867;
			float x4868;
			float x4869;
			x4868 = 1;
			x4869 = decay;
			x4866 = x4868 * x4869;
			float x4870;
			float x4871;
			x4870 = 1;
			x4871 = lrn_rate;
			x4867 = x4870 * x4871;
			x4865 = x4866 * x4867;
			x4863 = x4864 + x4865;
			JCudaTensor x4872;
			x4872 = x4822;
			x60.update(x4872, x4862, x4863);

			// 2a2_c_bn_scale <~~ V_2a2_c_bn_scale
			float x4873, x4874;
			x4873 = 1;
			float x4875;
			float x4876;
			x4875 = 1;
			float x4877;
			float x4878;
			float x4879;
			float x4880;
			x4879 = 1;
			x4880 = decay;
			x4877 = x4879 * x4880;
			float x4881;
			float x4882;
			x4881 = 1;
			x4882 = lrn_rate;
			x4878 = x4881 * x4882;
			x4876 = x4877 * x4878;
			x4874 = x4875 + x4876;
			JCudaTensor x4883;
			x4883 = x4840;
			x91.update(x4883, x4873, x4874);

			// 2a2_c_cv_W <~~ V_2a2_c_cv_W
			float x4884, x4885;
			x4884 = 1;
			float x4886;
			float x4887;
			x4886 = 1;
			float x4888;
			float x4889;
			float x4890;
			float x4891;
			x4890 = 1;
			x4891 = decay;
			x4888 = x4890 * x4891;
			float x4892;
			float x4893;
			x4892 = 1;
			x4893 = lrn_rate;
			x4889 = x4892 * x4893;
			x4887 = x4888 * x4889;
			x4885 = x4886 + x4887;
			JCudaTensor x4894;
			x4894 = x4854;
			x85.update(x4894, x4884, x4885);

			// 2a1_cv_W <~~ V_2a1_cv_W
			float x4895, x4896;
			x4895 = 1;
			float x4897;
			float x4898;
			x4897 = 1;
			float x4899;
			float x4900;
			float x4901;
			float x4902;
			x4901 = 1;
			x4902 = decay;
			x4899 = x4901 * x4902;
			float x4903;
			float x4904;
			x4903 = 1;
			x4904 = lrn_rate;
			x4900 = x4903 * x4904;
			x4898 = x4899 * x4900;
			x4896 = x4897 + x4898;
			JCudaTensor x4905;
			x4905 = x4829;
			x45.update(x4905, x4895, x4896);

			// 2a1_bn_scale <~~ V_2a1_bn_scale
			float x4906, x4907;
			x4906 = 1;
			float x4908;
			float x4909;
			x4908 = 1;
			float x4910;
			float x4911;
			float x4912;
			float x4913;
			x4912 = 1;
			x4913 = decay;
			x4910 = x4912 * x4913;
			float x4914;
			float x4915;
			x4914 = 1;
			x4915 = lrn_rate;
			x4911 = x4914 * x4915;
			x4909 = x4910 * x4911;
			x4907 = x4908 + x4909;
			JCudaTensor x4916;
			x4916 = x4815;
			x59.update(x4916, x4906, x4907);

			// 2a2_c_bn_bias <~~ V_2a2_c_bn_bias
			float x4917, x4918;
			x4917 = 1;
			float x4919;
			float x4920;
			x4919 = 1;
			float x4921;
			float x4922;
			float x4923;
			float x4924;
			x4923 = 1;
			x4924 = decay;
			x4921 = x4923 * x4924;
			float x4925;
			float x4926;
			x4925 = 1;
			x4926 = lrn_rate;
			x4922 = x4925 * x4926;
			x4920 = x4921 * x4922;
			x4918 = x4919 + x4920;
			JCudaTensor x4927;
			x4927 = x4847;
			x92.update(x4927, x4917, x4918);

			// val X4368 = X4364 * d_ReLU()(X8627)/d_X8626
			JCudaTensor x4928;
			JCudaTensor x4929, x4930;
			x4929 = x4812;
			x4930 = x79;
			x4928 = x64.backward(x4929, x4930);

			// Dealloc(X8627)
			JCudaTensor x4931;
			x4931 = x79;
			x4931.free();

			// val X8462 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_2a2_b_bn_bias
			JCudaTensor x4932;
			JCudaTensor x4933, x4934, x4935;
			x4933 = x4928;
			x4934 = x65;
			x4935 = x76;
			JCudaTensor[] x4936 = x78.backward(x4933,x4934,x4935);
			x4932 = x4936[2];

			// val X4369 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_X8625
			JCudaTensor x4937;
			x4937 = x4936[0];

			// val X8463 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_2a2_b_bn_scale
			JCudaTensor x4941;
			x4941 = x4936[1];

			// Dealloc(X8625)
			JCudaTensor x4945;
			x4945 = x65;
			x4945.free();

			// V_2a2_b_bn_scale <~~ X8463
			float x4947, x4948;
			float x4949;
			float x4950;
			x4949 = 1;
			x4950 = lrn_rate;
			x4947 = x4949 * x4950;
			x4948 = momentum;
			JCudaTensor x4951;
			x4951 = x4941;
			x4946.update(x4951, x4947, x4948);

			// Dealloc(X8463)
			JCudaTensor x4952;
			x4952 = x4941;
			x4952.free();

			// val X4370 = X4369 * d_Convolv(1,1)(2a2_b_cv_W)/d_X8624
			JCudaTensor x4953;
			JCudaTensor x4954, x4955;
			x4954 = x4937;
			x4955 = x69;
			x4953 = x71.backward_data(x4954, x4955);

			// V_2a2_b_cv_W <~~ X4369 * d_Convolv(1,1)(X8624)/d_2a2_b_cv_W
			float x4957, x4958;
			float x4959;
			float x4960;
			x4959 = 1;
			x4960 = lrn_rate;
			x4957 = x4959 * x4960;
			x4958 = momentum;
			JCudaTensor x4961, x4962;
			x4961 = x4937;
			x4962 = x62;
			x71.backward_filter(x4961, x4962, x4956, x4957, x4958);

			// Dealloc(X4369)
			JCudaTensor x4963;
			x4963 = x4937;
			x4963.free();

			// V_2a2_b_bn_bias <~~ X8462
			float x4965, x4966;
			float x4967;
			float x4968;
			x4967 = 1;
			x4968 = lrn_rate;
			x4965 = x4967 * x4968;
			x4966 = momentum;
			JCudaTensor x4969;
			x4969 = x4932;
			x4964.update(x4969, x4965, x4966);

			// Dealloc(X8462)
			JCudaTensor x4970;
			x4970 = x4932;
			x4970.free();

			// 2a2_b_bn_scale <~~ V_2a2_b_bn_scale
			float x4971, x4972;
			x4971 = 1;
			float x4973;
			float x4974;
			x4973 = 1;
			float x4975;
			float x4976;
			float x4977;
			float x4978;
			x4977 = 1;
			x4978 = decay;
			x4975 = x4977 * x4978;
			float x4979;
			float x4980;
			x4979 = 1;
			x4980 = lrn_rate;
			x4976 = x4979 * x4980;
			x4974 = x4975 * x4976;
			x4972 = x4973 + x4974;
			JCudaTensor x4981;
			x4981 = x4946;
			x76.update(x4981, x4971, x4972);

			// 2a2_b_cv_W <~~ V_2a2_b_cv_W
			float x4982, x4983;
			x4982 = 1;
			float x4984;
			float x4985;
			x4984 = 1;
			float x4986;
			float x4987;
			float x4988;
			float x4989;
			x4988 = 1;
			x4989 = decay;
			x4986 = x4988 * x4989;
			float x4990;
			float x4991;
			x4990 = 1;
			x4991 = lrn_rate;
			x4987 = x4990 * x4991;
			x4985 = x4986 * x4987;
			x4983 = x4984 + x4985;
			JCudaTensor x4992;
			x4992 = x4956;
			x69.update(x4992, x4982, x4983);

			// 2a2_b_bn_bias <~~ V_2a2_b_bn_bias
			float x4993, x4994;
			x4993 = 1;
			float x4995;
			float x4996;
			x4995 = 1;
			float x4997;
			float x4998;
			float x4999;
			float x5000;
			x4999 = 1;
			x5000 = decay;
			x4997 = x4999 * x5000;
			float x5001;
			float x5002;
			x5001 = 1;
			x5002 = lrn_rate;
			x4998 = x5001 * x5002;
			x4996 = x4997 * x4998;
			x4994 = x4995 + x4996;
			JCudaTensor x5003;
			x5003 = x4964;
			x77.update(x5003, x4993, x4994);

			// val X4374 = X4370 * d_ReLU()(X8624)/d_X8623
			JCudaTensor x5004;
			JCudaTensor x5005, x5006;
			x5005 = x4953;
			x5006 = x62;
			x5004 = x64.backward(x5005, x5006);

			// Dealloc(X8624)
			JCudaTensor x5007;
			x5007 = x62;
			x5007.free();

			// val X4375 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_X8622
			JCudaTensor x5008;
			JCudaTensor x5009, x5010, x5011;
			x5009 = x5004;
			x5010 = x34;
			x5011 = x52;
			JCudaTensor[] x5012 = x54.backward(x5009,x5010,x5011);
			x5008 = x5012[0];

			// val X8460 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_2a2_a_bn_scale
			JCudaTensor x5013;
			x5013 = x5012[1];

			// val X8459 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_2a2_a_bn_bias
			JCudaTensor x5017;
			x5017 = x5012[2];

			// Dealloc(X8622)
			JCudaTensor x5021;
			x5021 = x34;
			x5021.free();

			// V_2a2_a_bn_scale <~~ X8460
			float x5023, x5024;
			float x5025;
			float x5026;
			x5025 = 1;
			x5026 = lrn_rate;
			x5023 = x5025 * x5026;
			x5024 = momentum;
			JCudaTensor x5027;
			x5027 = x5013;
			x5022.update(x5027, x5023, x5024);

			// Dealloc(X8460)
			JCudaTensor x5028;
			x5028 = x5013;
			x5028.free();

			// V_2a2_a_bn_bias <~~ X8459
			float x5030, x5031;
			float x5032;
			float x5033;
			x5032 = 1;
			x5033 = lrn_rate;
			x5030 = x5032 * x5033;
			x5031 = momentum;
			JCudaTensor x5034;
			x5034 = x5017;
			x5029.update(x5034, x5030, x5031);

			// Dealloc(X8459)
			JCudaTensor x5035;
			x5035 = x5017;
			x5035.free();

			// val X4377 = (X4352 + X4375 * d_Convolv(1,0)(2a2_a_cv_W)/d_X8618)
			JCudaTensor x5036;
			JCudaTensor x5037;
			x5037 = x4836;
			JCudaTensor x5038, x5039;
			x5038 = x5008;
			x5039 = x38;
			x5036 = x40.backward_data(x5038,x5039, x5037);

			// V_2a2_a_cv_W <~~ X4375 * d_Convolv(1,0)(X8618)/d_2a2_a_cv_W
			float x5041, x5042;
			float x5043;
			float x5044;
			x5043 = 1;
			x5044 = lrn_rate;
			x5041 = x5043 * x5044;
			x5042 = momentum;
			JCudaTensor x5045, x5046;
			x5045 = x5008;
			x5046 = x31;
			x40.backward_filter(x5045, x5046, x5040, x5041, x5042);

			// Dealloc(X4375)
			JCudaTensor x5047;
			x5047 = x5008;
			x5047.free();

			// 2a2_a_bn_scale <~~ V_2a2_a_bn_scale
			float x5048, x5049;
			x5048 = 1;
			float x5050;
			float x5051;
			x5050 = 1;
			float x5052;
			float x5053;
			float x5054;
			float x5055;
			x5054 = 1;
			x5055 = decay;
			x5052 = x5054 * x5055;
			float x5056;
			float x5057;
			x5056 = 1;
			x5057 = lrn_rate;
			x5053 = x5056 * x5057;
			x5051 = x5052 * x5053;
			x5049 = x5050 + x5051;
			JCudaTensor x5058;
			x5058 = x5022;
			x52.update(x5058, x5048, x5049);

			// 2a2_a_bn_bias <~~ V_2a2_a_bn_bias
			float x5059, x5060;
			x5059 = 1;
			float x5061;
			float x5062;
			x5061 = 1;
			float x5063;
			float x5064;
			float x5065;
			float x5066;
			x5065 = 1;
			x5066 = decay;
			x5063 = x5065 * x5066;
			float x5067;
			float x5068;
			x5067 = 1;
			x5068 = lrn_rate;
			x5064 = x5067 * x5068;
			x5062 = x5063 * x5064;
			x5060 = x5061 + x5062;
			JCudaTensor x5069;
			x5069 = x5029;
			x53.update(x5069, x5059, x5060);

			// 2a2_a_cv_W <~~ V_2a2_a_cv_W
			float x5070, x5071;
			x5070 = 1;
			float x5072;
			float x5073;
			x5072 = 1;
			float x5074;
			float x5075;
			float x5076;
			float x5077;
			x5076 = 1;
			x5077 = decay;
			x5074 = x5076 * x5077;
			float x5078;
			float x5079;
			x5078 = 1;
			x5079 = lrn_rate;
			x5075 = x5078 * x5079;
			x5073 = x5074 * x5075;
			x5071 = x5072 + x5073;
			JCudaTensor x5080;
			x5080 = x5040;
			x38.update(x5080, x5070, x5071);

			// val X4379 = X4377 * d_Pooling(3,2,0,true)(X8618,X8617)/d_X8617
			JCudaTensor x5081;
			JCudaTensor x5082, x5083, x5084;
			x5082 = x5036;
			x5083 = x31;
			x5084 = x28;
			x5081 = x33.backward(x5082, x5083, x5084);

			// Dealloc(X4377)
			JCudaTensor x5085;
			x5085 = x5036;
			x5085.free();

			// Dealloc(X8618)
			JCudaTensor x5086;
			x5086 = x31;
			x5086.free();

			// val X8448 = X4379 * d_ReLU()(X8617)/d_X8616
			JCudaTensor x5087;
			JCudaTensor x5088, x5089;
			x5088 = x5081;
			x5089 = x28;
			x5087 = x30.backward(x5088, x5089);

			// Dealloc(X8617)
			JCudaTensor x5090;
			x5090 = x28;
			x5090.free();

			// val X8454 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_1_bn_scale
			JCudaTensor x5091;
			JCudaTensor x5092, x5093, x5094;
			x5092 = x5087;
			x5093 = x11;
			x5094 = x25;
			JCudaTensor[] x5095 = x27.backward(x5092,x5093,x5094);
			x5091 = x5095[1];

			// val X8449 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_X8615
			JCudaTensor x5096;
			x5096 = x5095[0];

			// val X8453 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_1_bn_bias
			JCudaTensor x5100;
			x5100 = x5095[2];

			// Dealloc(X8615)
			JCudaTensor x5104;
			x5104 = x11;
			x5104.free();

			// V_1_bn_scale <~~ X8454
			float x5106, x5107;
			float x5108;
			float x5109;
			x5108 = 1;
			x5109 = lrn_rate;
			x5106 = x5108 * x5109;
			x5107 = momentum;
			JCudaTensor x5110;
			x5110 = x5091;
			x5105.update(x5110, x5106, x5107);

			// Dealloc(X8454)
			JCudaTensor x5111;
			x5111 = x5091;
			x5111.free();

			// V_1_bn_bias <~~ X8453
			float x5113, x5114;
			float x5115;
			float x5116;
			x5115 = 1;
			x5116 = lrn_rate;
			x5113 = x5115 * x5116;
			x5114 = momentum;
			JCudaTensor x5117;
			x5117 = x5100;
			x5112.update(x5117, x5113, x5114);

			// Dealloc(X8453)
			JCudaTensor x5118;
			x5118 = x5100;
			x5118.free();

			// V_1_cv_W <~~ X8449 * d_Convolv(2,3)(X8614)/d_1_cv_W
			float x5120, x5121;
			float x5122;
			float x5123;
			x5122 = 1;
			x5123 = lrn_rate;
			x5120 = x5122 * x5123;
			x5121 = momentum;
			JCudaTensor x5124, x5125;
			x5124 = x5096;
			x5125 = x7;
			x17.backward_filter(x5124, x5125, x5119, x5120, x5121);

			// Dealloc(X8449)
			JCudaTensor x5126;
			x5126 = x5096;
			x5126.free();

			// Dealloc(X8614)
			JCudaTensor x5127;
			x5127 = x7;
			x5127.free();

			// 1_bn_scale <~~ V_1_bn_scale
			float x5128, x5129;
			x5128 = 1;
			float x5130;
			float x5131;
			x5130 = 1;
			float x5132;
			float x5133;
			float x5134;
			float x5135;
			x5134 = 1;
			x5135 = decay;
			x5132 = x5134 * x5135;
			float x5136;
			float x5137;
			x5136 = 1;
			x5137 = lrn_rate;
			x5133 = x5136 * x5137;
			x5131 = x5132 * x5133;
			x5129 = x5130 + x5131;
			JCudaTensor x5138;
			x5138 = x5105;
			x25.update(x5138, x5128, x5129);

			// 1_bn_bias <~~ V_1_bn_bias
			float x5139, x5140;
			x5139 = 1;
			float x5141;
			float x5142;
			x5141 = 1;
			float x5143;
			float x5144;
			float x5145;
			float x5146;
			x5145 = 1;
			x5146 = decay;
			x5143 = x5145 * x5146;
			float x5147;
			float x5148;
			x5147 = 1;
			x5148 = lrn_rate;
			x5144 = x5147 * x5148;
			x5142 = x5143 * x5144;
			x5140 = x5141 + x5142;
			JCudaTensor x5149;
			x5149 = x5112;
			x26.update(x5149, x5139, x5140);

			// 1_cv_W <~~ V_1_cv_W
			float x5150, x5151;
			x5150 = 1;
			float x5152;
			float x5153;
			x5152 = 1;
			float x5154;
			float x5155;
			float x5156;
			float x5157;
			x5156 = 1;
			x5157 = decay;
			x5154 = x5156 * x5157;
			float x5158;
			float x5159;
			x5158 = 1;
			x5159 = lrn_rate;
			x5155 = x5158 * x5159;
			x5153 = x5154 * x5155;
			x5151 = x5152 + x5153;
			JCudaTensor x5160;
			x5160 = x5119;
			x15.update(x5160, x5150, x5151);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X8813 = Cuda(X)
			JCudaTensor x5161;
			JTensorFloat x5162;
			x5162 = x3;
			x5161 = x5162.asJCudaTensor();

			// val X8814 = Convolv(2,3)(X8813,1_cv_W,1_cv_B)
			JCudaTensor x5163;
			JCudaTensor x5164, x5165, x5166;
			x5164 = x5161;
			x5165 = x15;
			x5166 = x16;
			x5163 = x17.forward(x5164, x5165, x5166);

			// Dealloc(X8813)
			JCudaTensor x5167;
			x5167 = x5161;
			x5167.free();

			// val X8815 = BatchNorm(1_bn)(X8814,1_bn_scale,1_bn_bias)
			JCudaTensor x5168;
			JCudaTensor x5169, x5170, x5171;
			x5169 = x5163;
			x5170 = x25;
			x5171 = x26;
			x5168 = x27.forward_inference(x5169, x5170, x5171);

			// Dealloc(X8814)
			JCudaTensor x5172;
			x5172 = x5163;
			x5172.free();

			// val X8816 = ReLU()(X8815)
			JCudaTensor x5173;
			JCudaTensor x5174;
			x5174 = x5168;
			x5173 = x30.forward(x5174);

			// val X8817 = Pooling(3,2,0,true)(X8816)
			JCudaTensor x5175;
			JCudaTensor x5176;
			x5176 = x5173;
			x5175 = x33.forward(x5176);

			// Dealloc(X8816)
			JCudaTensor x5177;
			x5177 = x5173;
			x5177.free();

			// val X8821 = Convolv(1,0)(X8817,2a2_a_cv_W,2a2_a_cv_B)
			JCudaTensor x5178;
			JCudaTensor x5179, x5180, x5181;
			x5179 = x5175;
			x5180 = x38;
			x5181 = x39;
			x5178 = x40.forward(x5179, x5180, x5181);

			// val X8818 = Convolv(1,0)(X8817,2a1_cv_W,2a1_cv_B)
			JCudaTensor x5182;
			JCudaTensor x5183, x5184, x5185;
			x5183 = x5175;
			x5184 = x45;
			x5185 = x46;
			x5182 = x47.forward(x5183, x5184, x5185);

			// Dealloc(X8817)
			JCudaTensor x5186;
			x5186 = x5175;
			x5186.free();

			// val X8822 = BatchNorm(2a2_a_bn)(X8821,2a2_a_bn_scale,2a2_a_bn_bias)
			JCudaTensor x5187;
			JCudaTensor x5188, x5189, x5190;
			x5188 = x5178;
			x5189 = x52;
			x5190 = x53;
			x5187 = x54.forward_inference(x5188, x5189, x5190);

			// Dealloc(X8821)
			JCudaTensor x5191;
			x5191 = x5178;
			x5191.free();

			// val X8819 = BatchNorm(2a1_bn)(X8818,2a1_bn_scale,2a1_bn_bias)
			JCudaTensor x5192;
			JCudaTensor x5193, x5194, x5195;
			x5193 = x5182;
			x5194 = x59;
			x5195 = x60;
			x5192 = x61.forward_inference(x5193, x5194, x5195);

			// Dealloc(X8818)
			JCudaTensor x5196;
			x5196 = x5182;
			x5196.free();

			// val X8823 = ReLU()(X8822)
			JCudaTensor x5197;
			JCudaTensor x5198;
			x5198 = x5187;
			x5197 = x64.forward(x5198);

			// val X8824 = Convolv(1,1)(X8823,2a2_b_cv_W,2a2_b_cv_B)
			JCudaTensor x5199;
			JCudaTensor x5200, x5201, x5202;
			x5200 = x5197;
			x5201 = x69;
			x5202 = x70;
			x5199 = x71.forward(x5200, x5201, x5202);

			// Dealloc(X8823)
			JCudaTensor x5203;
			x5203 = x5197;
			x5203.free();

			// val X8825 = BatchNorm(2a2_b_bn)(X8824,2a2_b_bn_scale,2a2_b_bn_bias)
			JCudaTensor x5204;
			JCudaTensor x5205, x5206, x5207;
			x5205 = x5199;
			x5206 = x76;
			x5207 = x77;
			x5204 = x78.forward_inference(x5205, x5206, x5207);

			// Dealloc(X8824)
			JCudaTensor x5208;
			x5208 = x5199;
			x5208.free();

			// val X8826 = ReLU()(X8825)
			JCudaTensor x5209;
			JCudaTensor x5210;
			x5210 = x5204;
			x5209 = x64.forward(x5210);

			// val X8827 = Convolv(1,0)(X8826,2a2_c_cv_W,2a2_c_cv_B)
			JCudaTensor x5211;
			JCudaTensor x5212, x5213, x5214;
			x5212 = x5209;
			x5213 = x85;
			x5214 = x86;
			x5211 = x47.forward(x5212, x5213, x5214);

			// Dealloc(X8826)
			JCudaTensor x5215;
			x5215 = x5209;
			x5215.free();

			// val X8828 = BatchNorm(2a2_c_bn)(X8827,2a2_c_bn_scale,2a2_c_bn_bias)
			JCudaTensor x5216;
			JCudaTensor x5217, x5218, x5219;
			x5217 = x5211;
			x5218 = x91;
			x5219 = x92;
			x5216 = x93.forward_inference(x5217, x5218, x5219);

			// Dealloc(X8827)
			JCudaTensor x5220;
			x5220 = x5211;
			x5220.free();

			// val X8820 = ReLU()(X8819)
			JCudaTensor x5221;
			JCudaTensor x5222;
			x5222 = x5192;
			x5221 = x96.forward(x5222);

			// val X8829 = ReLU()(X8828)
			JCudaTensor x5223;
			JCudaTensor x5224;
			x5224 = x5216;
			x5223 = x96.forward(x5224);

			// val X8830 = (X8820 + X8829)
			JCudaTensor x5225;
			JCudaTensor x5226, x5227;
			x5226 = x5221;
			x5227 = x5223;
			x5225 = x5226.plus_i(x5227);

			// Dealloc(X8829)
			JCudaTensor x5228;
			x5228 = x5223;
			x5228.free();

			// val X8831 = ReLU()(X8830)
			JCudaTensor x5229;
			JCudaTensor x5230;
			x5230 = x5225;
			x5229 = x96.forward(x5230);

			// val X8832 = Convolv(1,0)(X8831,2b_a_cv_W,2b_a_cv_B)
			JCudaTensor x5231;
			JCudaTensor x5232, x5233, x5234;
			x5232 = x5229;
			x5233 = x108;
			x5234 = x109;
			x5231 = x110.forward(x5232, x5233, x5234);

			// val X8833 = BatchNorm(2b_a_bn)(X8832,2b_a_bn_scale,2b_a_bn_bias)
			JCudaTensor x5235;
			JCudaTensor x5236, x5237, x5238;
			x5236 = x5231;
			x5237 = x115;
			x5238 = x116;
			x5235 = x117.forward_inference(x5236, x5237, x5238);

			// Dealloc(X8832)
			JCudaTensor x5239;
			x5239 = x5231;
			x5239.free();

			// val X8834 = ReLU()(X8833)
			JCudaTensor x5240;
			JCudaTensor x5241;
			x5241 = x5235;
			x5240 = x64.forward(x5241);

			// val X8835 = Convolv(1,1)(X8834,2b_b_cv_W,2b_b_cv_B)
			JCudaTensor x5242;
			JCudaTensor x5243, x5244, x5245;
			x5243 = x5240;
			x5244 = x124;
			x5245 = x125;
			x5242 = x71.forward(x5243, x5244, x5245);

			// Dealloc(X8834)
			JCudaTensor x5246;
			x5246 = x5240;
			x5246.free();

			// val X8836 = BatchNorm(2b_b_bn)(X8835,2b_b_bn_scale,2b_b_bn_bias)
			JCudaTensor x5247;
			JCudaTensor x5248, x5249, x5250;
			x5248 = x5242;
			x5249 = x130;
			x5250 = x131;
			x5247 = x132.forward_inference(x5248, x5249, x5250);

			// Dealloc(X8835)
			JCudaTensor x5251;
			x5251 = x5242;
			x5251.free();

			// val X8837 = ReLU()(X8836)
			JCudaTensor x5252;
			JCudaTensor x5253;
			x5253 = x5247;
			x5252 = x64.forward(x5253);

			// val X8838 = Convolv(1,0)(X8837,2b_c_cv_W,2b_c_cv_B)
			JCudaTensor x5254;
			JCudaTensor x5255, x5256, x5257;
			x5255 = x5252;
			x5256 = x139;
			x5257 = x140;
			x5254 = x47.forward(x5255, x5256, x5257);

			// Dealloc(X8837)
			JCudaTensor x5258;
			x5258 = x5252;
			x5258.free();

			// val X8839 = BatchNorm(2b_c_bn)(X8838,2b_c_bn_scale,2b_c_bn_bias)
			JCudaTensor x5259;
			JCudaTensor x5260, x5261, x5262;
			x5260 = x5254;
			x5261 = x145;
			x5262 = x146;
			x5259 = x147.forward_inference(x5260, x5261, x5262);

			// Dealloc(X8838)
			JCudaTensor x5263;
			x5263 = x5254;
			x5263.free();

			// val X8840 = ReLU()(X8839)
			JCudaTensor x5264;
			JCudaTensor x5265;
			x5265 = x5259;
			x5264 = x96.forward(x5265);

			// val X8841 = (X8840 + X8831)
			JCudaTensor x5266;
			JCudaTensor x5267, x5268;
			x5267 = x5264;
			x5268 = x5229;
			x5266 = x5267.plus_i(x5268);

			// Dealloc(X8831)
			JCudaTensor x5269;
			x5269 = x5229;
			x5269.free();

			// val X8842 = ReLU()(X8841)
			JCudaTensor x5270;
			JCudaTensor x5271;
			x5271 = x5266;
			x5270 = x96.forward(x5271);

			// val X8843 = Convolv(1,0)(X8842,2c_a_cv_W,2c_a_cv_B)
			JCudaTensor x5272;
			JCudaTensor x5273, x5274, x5275;
			x5273 = x5270;
			x5274 = x159;
			x5275 = x160;
			x5272 = x110.forward(x5273, x5274, x5275);

			// val X8844 = BatchNorm(2c_a_bn)(X8843,2c_a_bn_scale,2c_a_bn_bias)
			JCudaTensor x5276;
			JCudaTensor x5277, x5278, x5279;
			x5277 = x5272;
			x5278 = x165;
			x5279 = x166;
			x5276 = x167.forward_inference(x5277, x5278, x5279);

			// Dealloc(X8843)
			JCudaTensor x5280;
			x5280 = x5272;
			x5280.free();

			// val X8845 = ReLU()(X8844)
			JCudaTensor x5281;
			JCudaTensor x5282;
			x5282 = x5276;
			x5281 = x64.forward(x5282);

			// val X8846 = Convolv(1,1)(X8845,2c_b_cv_W,2c_b_cv_B)
			JCudaTensor x5283;
			JCudaTensor x5284, x5285, x5286;
			x5284 = x5281;
			x5285 = x174;
			x5286 = x175;
			x5283 = x71.forward(x5284, x5285, x5286);

			// Dealloc(X8845)
			JCudaTensor x5287;
			x5287 = x5281;
			x5287.free();

			// val X8847 = BatchNorm(2c_b_bn)(X8846,2c_b_bn_scale,2c_b_bn_bias)
			JCudaTensor x5288;
			JCudaTensor x5289, x5290, x5291;
			x5289 = x5283;
			x5290 = x180;
			x5291 = x181;
			x5288 = x182.forward_inference(x5289, x5290, x5291);

			// Dealloc(X8846)
			JCudaTensor x5292;
			x5292 = x5283;
			x5292.free();

			// val X8848 = ReLU()(X8847)
			JCudaTensor x5293;
			JCudaTensor x5294;
			x5294 = x5288;
			x5293 = x64.forward(x5294);

			// val X8849 = Convolv(1,0)(X8848,2c_c_cv_W,2c_c_cv_B)
			JCudaTensor x5295;
			JCudaTensor x5296, x5297, x5298;
			x5296 = x5293;
			x5297 = x189;
			x5298 = x190;
			x5295 = x47.forward(x5296, x5297, x5298);

			// Dealloc(X8848)
			JCudaTensor x5299;
			x5299 = x5293;
			x5299.free();

			// val X8850 = BatchNorm(2c_c_bn)(X8849,2c_c_bn_scale,2c_c_bn_bias)
			JCudaTensor x5300;
			JCudaTensor x5301, x5302, x5303;
			x5301 = x5295;
			x5302 = x195;
			x5303 = x196;
			x5300 = x197.forward_inference(x5301, x5302, x5303);

			// Dealloc(X8849)
			JCudaTensor x5304;
			x5304 = x5295;
			x5304.free();

			// val X8851 = ReLU()(X8850)
			JCudaTensor x5305;
			JCudaTensor x5306;
			x5306 = x5300;
			x5305 = x96.forward(x5306);

			// val X8852 = (X8851 + X8842)
			JCudaTensor x5307;
			JCudaTensor x5308, x5309;
			x5308 = x5305;
			x5309 = x5270;
			x5307 = x5308.plus_i(x5309);

			// Dealloc(X8842)
			JCudaTensor x5310;
			x5310 = x5270;
			x5310.free();

			// val X8853 = ReLU()(X8852)
			JCudaTensor x5311;
			JCudaTensor x5312;
			x5312 = x5307;
			x5311 = x96.forward(x5312);

			// val X8854 = Convolv(2,0)(X8853,3a1_cv_W,3a1_cv_B)
			JCudaTensor x5313;
			JCudaTensor x5314, x5315, x5316;
			x5314 = x5311;
			x5315 = x216;
			x5316 = x217;
			x5313 = x218.forward(x5314, x5315, x5316);

			// val X8857 = Convolv(2,0)(X8853,3a2_a_cv_W,3a2_a_cv_B)
			JCudaTensor x5317;
			JCudaTensor x5318, x5319, x5320;
			x5318 = x5311;
			x5319 = x209;
			x5320 = x210;
			x5317 = x211.forward(x5318, x5319, x5320);

			// Dealloc(X8853)
			JCudaTensor x5321;
			x5321 = x5311;
			x5321.free();

			// val X8858 = BatchNorm(3a2_a_bn)(X8857,3a2_a_bn_scale,3a2_a_bn_bias)
			JCudaTensor x5322;
			JCudaTensor x5323, x5324, x5325;
			x5323 = x5317;
			x5324 = x230;
			x5325 = x231;
			x5322 = x232.forward_inference(x5323, x5324, x5325);

			// Dealloc(X8857)
			JCudaTensor x5326;
			x5326 = x5317;
			x5326.free();

			// val X8855 = BatchNorm(3a1_bn)(X8854,3a1_bn_scale,3a1_bn_bias)
			JCudaTensor x5327;
			JCudaTensor x5328, x5329, x5330;
			x5328 = x5313;
			x5329 = x223;
			x5330 = x224;
			x5327 = x225.forward_inference(x5328, x5329, x5330);

			// Dealloc(X8854)
			JCudaTensor x5331;
			x5331 = x5313;
			x5331.free();

			// val X8859 = ReLU()(X8858)
			JCudaTensor x5332;
			JCudaTensor x5333;
			x5333 = x5322;
			x5332 = x235.forward(x5333);

			// val X8860 = Convolv(1,1)(X8859,3a2_b_cv_W,3a2_b_cv_B)
			JCudaTensor x5334;
			JCudaTensor x5335, x5336, x5337;
			x5335 = x5332;
			x5336 = x240;
			x5337 = x241;
			x5334 = x242.forward(x5335, x5336, x5337);

			// Dealloc(X8859)
			JCudaTensor x5338;
			x5338 = x5332;
			x5338.free();

			// val X8861 = BatchNorm(3a2_b_bn)(X8860,3a2_b_bn_scale,3a2_b_bn_bias)
			JCudaTensor x5339;
			JCudaTensor x5340, x5341, x5342;
			x5340 = x5334;
			x5341 = x247;
			x5342 = x248;
			x5339 = x249.forward_inference(x5340, x5341, x5342);

			// Dealloc(X8860)
			JCudaTensor x5343;
			x5343 = x5334;
			x5343.free();

			// val X8862 = ReLU()(X8861)
			JCudaTensor x5344;
			JCudaTensor x5345;
			x5345 = x5339;
			x5344 = x235.forward(x5345);

			// val X8863 = Convolv(1,0)(X8862,3a2_c_cv_W,3a2_c_cv_B)
			JCudaTensor x5346;
			JCudaTensor x5347, x5348, x5349;
			x5347 = x5344;
			x5348 = x256;
			x5349 = x257;
			x5346 = x258.forward(x5347, x5348, x5349);

			// Dealloc(X8862)
			JCudaTensor x5350;
			x5350 = x5344;
			x5350.free();

			// val X8864 = BatchNorm(3a2_c_bn)(X8863,3a2_c_bn_scale,3a2_c_bn_bias)
			JCudaTensor x5351;
			JCudaTensor x5352, x5353, x5354;
			x5352 = x5346;
			x5353 = x263;
			x5354 = x264;
			x5351 = x265.forward_inference(x5352, x5353, x5354);

			// Dealloc(X8863)
			JCudaTensor x5355;
			x5355 = x5346;
			x5355.free();

			// val X8856 = ReLU()(X8855)
			JCudaTensor x5356;
			JCudaTensor x5357;
			x5357 = x5327;
			x5356 = x268.forward(x5357);

			// val X8865 = ReLU()(X8864)
			JCudaTensor x5358;
			JCudaTensor x5359;
			x5359 = x5351;
			x5358 = x268.forward(x5359);

			// val X8866 = (X8856 + X8865)
			JCudaTensor x5360;
			JCudaTensor x5361, x5362;
			x5361 = x5356;
			x5362 = x5358;
			x5360 = x5361.plus_i(x5362);

			// Dealloc(X8865)
			JCudaTensor x5363;
			x5363 = x5358;
			x5363.free();

			// val X8867 = ReLU()(X8866)
			JCudaTensor x5364;
			JCudaTensor x5365;
			x5365 = x5360;
			x5364 = x268.forward(x5365);

			// val X8868 = Convolv(1,0)(X8867,3b_a_cv_W,3b_a_cv_B)
			JCudaTensor x5366;
			JCudaTensor x5367, x5368, x5369;
			x5367 = x5364;
			x5368 = x280;
			x5369 = x281;
			x5366 = x282.forward(x5367, x5368, x5369);

			// val X8869 = BatchNorm(3b_a_bn)(X8868,3b_a_bn_scale,3b_a_bn_bias)
			JCudaTensor x5370;
			JCudaTensor x5371, x5372, x5373;
			x5371 = x5366;
			x5372 = x287;
			x5373 = x288;
			x5370 = x289.forward_inference(x5371, x5372, x5373);

			// Dealloc(X8868)
			JCudaTensor x5374;
			x5374 = x5366;
			x5374.free();

			// val X8870 = ReLU()(X8869)
			JCudaTensor x5375;
			JCudaTensor x5376;
			x5376 = x5370;
			x5375 = x235.forward(x5376);

			// val X8871 = Convolv(1,1)(X8870,3b_b_cv_W,3b_b_cv_B)
			JCudaTensor x5377;
			JCudaTensor x5378, x5379, x5380;
			x5378 = x5375;
			x5379 = x296;
			x5380 = x297;
			x5377 = x242.forward(x5378, x5379, x5380);

			// Dealloc(X8870)
			JCudaTensor x5381;
			x5381 = x5375;
			x5381.free();

			// val X8872 = BatchNorm(3b_b_bn)(X8871,3b_b_bn_scale,3b_b_bn_bias)
			JCudaTensor x5382;
			JCudaTensor x5383, x5384, x5385;
			x5383 = x5377;
			x5384 = x302;
			x5385 = x303;
			x5382 = x304.forward_inference(x5383, x5384, x5385);

			// Dealloc(X8871)
			JCudaTensor x5386;
			x5386 = x5377;
			x5386.free();

			// val X8873 = ReLU()(X8872)
			JCudaTensor x5387;
			JCudaTensor x5388;
			x5388 = x5382;
			x5387 = x235.forward(x5388);

			// val X8874 = Convolv(1,0)(X8873,3b_c_cv_W,3b_c_cv_B)
			JCudaTensor x5389;
			JCudaTensor x5390, x5391, x5392;
			x5390 = x5387;
			x5391 = x311;
			x5392 = x312;
			x5389 = x258.forward(x5390, x5391, x5392);

			// Dealloc(X8873)
			JCudaTensor x5393;
			x5393 = x5387;
			x5393.free();

			// val X8875 = BatchNorm(3b_c_bn)(X8874,3b_c_bn_scale,3b_c_bn_bias)
			JCudaTensor x5394;
			JCudaTensor x5395, x5396, x5397;
			x5395 = x5389;
			x5396 = x317;
			x5397 = x318;
			x5394 = x319.forward_inference(x5395, x5396, x5397);

			// Dealloc(X8874)
			JCudaTensor x5398;
			x5398 = x5389;
			x5398.free();

			// val X8876 = ReLU()(X8875)
			JCudaTensor x5399;
			JCudaTensor x5400;
			x5400 = x5394;
			x5399 = x268.forward(x5400);

			// val X8877 = (X8876 + X8867)
			JCudaTensor x5401;
			JCudaTensor x5402, x5403;
			x5402 = x5399;
			x5403 = x5364;
			x5401 = x5402.plus_i(x5403);

			// Dealloc(X8867)
			JCudaTensor x5404;
			x5404 = x5364;
			x5404.free();

			// val X8878 = ReLU()(X8877)
			JCudaTensor x5405;
			JCudaTensor x5406;
			x5406 = x5401;
			x5405 = x268.forward(x5406);

			// val X8879 = Convolv(1,0)(X8878,3c_a_cv_W,3c_a_cv_B)
			JCudaTensor x5407;
			JCudaTensor x5408, x5409, x5410;
			x5408 = x5405;
			x5409 = x331;
			x5410 = x332;
			x5407 = x282.forward(x5408, x5409, x5410);

			// val X8880 = BatchNorm(3c_a_bn)(X8879,3c_a_bn_scale,3c_a_bn_bias)
			JCudaTensor x5411;
			JCudaTensor x5412, x5413, x5414;
			x5412 = x5407;
			x5413 = x337;
			x5414 = x338;
			x5411 = x339.forward_inference(x5412, x5413, x5414);

			// Dealloc(X8879)
			JCudaTensor x5415;
			x5415 = x5407;
			x5415.free();

			// val X8881 = ReLU()(X8880)
			JCudaTensor x5416;
			JCudaTensor x5417;
			x5417 = x5411;
			x5416 = x235.forward(x5417);

			// val X8882 = Convolv(1,1)(X8881,3c_b_cv_W,3c_b_cv_B)
			JCudaTensor x5418;
			JCudaTensor x5419, x5420, x5421;
			x5419 = x5416;
			x5420 = x346;
			x5421 = x347;
			x5418 = x242.forward(x5419, x5420, x5421);

			// Dealloc(X8881)
			JCudaTensor x5422;
			x5422 = x5416;
			x5422.free();

			// val X8883 = BatchNorm(3c_b_bn)(X8882,3c_b_bn_scale,3c_b_bn_bias)
			JCudaTensor x5423;
			JCudaTensor x5424, x5425, x5426;
			x5424 = x5418;
			x5425 = x352;
			x5426 = x353;
			x5423 = x354.forward_inference(x5424, x5425, x5426);

			// Dealloc(X8882)
			JCudaTensor x5427;
			x5427 = x5418;
			x5427.free();

			// val X8884 = ReLU()(X8883)
			JCudaTensor x5428;
			JCudaTensor x5429;
			x5429 = x5423;
			x5428 = x235.forward(x5429);

			// val X8885 = Convolv(1,0)(X8884,3c_c_cv_W,3c_c_cv_B)
			JCudaTensor x5430;
			JCudaTensor x5431, x5432, x5433;
			x5431 = x5428;
			x5432 = x361;
			x5433 = x362;
			x5430 = x258.forward(x5431, x5432, x5433);

			// Dealloc(X8884)
			JCudaTensor x5434;
			x5434 = x5428;
			x5434.free();

			// val X8886 = BatchNorm(3c_c_bn)(X8885,3c_c_bn_scale,3c_c_bn_bias)
			JCudaTensor x5435;
			JCudaTensor x5436, x5437, x5438;
			x5436 = x5430;
			x5437 = x367;
			x5438 = x368;
			x5435 = x369.forward_inference(x5436, x5437, x5438);

			// Dealloc(X8885)
			JCudaTensor x5439;
			x5439 = x5430;
			x5439.free();

			// val X8887 = ReLU()(X8886)
			JCudaTensor x5440;
			JCudaTensor x5441;
			x5441 = x5435;
			x5440 = x268.forward(x5441);

			// val X8888 = (X8887 + X8878)
			JCudaTensor x5442;
			JCudaTensor x5443, x5444;
			x5443 = x5440;
			x5444 = x5405;
			x5442 = x5443.plus_i(x5444);

			// Dealloc(X8878)
			JCudaTensor x5445;
			x5445 = x5405;
			x5445.free();

			// val X8889 = ReLU()(X8888)
			JCudaTensor x5446;
			JCudaTensor x5447;
			x5447 = x5442;
			x5446 = x268.forward(x5447);

			// val X8890 = Convolv(1,0)(X8889,3d_a_cv_W,3d_a_cv_B)
			JCudaTensor x5448;
			JCudaTensor x5449, x5450, x5451;
			x5449 = x5446;
			x5450 = x381;
			x5451 = x382;
			x5448 = x282.forward(x5449, x5450, x5451);

			// val X8891 = BatchNorm(3d_a_bn)(X8890,3d_a_bn_scale,3d_a_bn_bias)
			JCudaTensor x5452;
			JCudaTensor x5453, x5454, x5455;
			x5453 = x5448;
			x5454 = x387;
			x5455 = x388;
			x5452 = x389.forward_inference(x5453, x5454, x5455);

			// Dealloc(X8890)
			JCudaTensor x5456;
			x5456 = x5448;
			x5456.free();

			// val X8892 = ReLU()(X8891)
			JCudaTensor x5457;
			JCudaTensor x5458;
			x5458 = x5452;
			x5457 = x235.forward(x5458);

			// val X8893 = Convolv(1,1)(X8892,3d_b_cv_W,3d_b_cv_B)
			JCudaTensor x5459;
			JCudaTensor x5460, x5461, x5462;
			x5460 = x5457;
			x5461 = x396;
			x5462 = x397;
			x5459 = x242.forward(x5460, x5461, x5462);

			// Dealloc(X8892)
			JCudaTensor x5463;
			x5463 = x5457;
			x5463.free();

			// val X8894 = BatchNorm(3d_b_bn)(X8893,3d_b_bn_scale,3d_b_bn_bias)
			JCudaTensor x5464;
			JCudaTensor x5465, x5466, x5467;
			x5465 = x5459;
			x5466 = x402;
			x5467 = x403;
			x5464 = x404.forward_inference(x5465, x5466, x5467);

			// Dealloc(X8893)
			JCudaTensor x5468;
			x5468 = x5459;
			x5468.free();

			// val X8895 = ReLU()(X8894)
			JCudaTensor x5469;
			JCudaTensor x5470;
			x5470 = x5464;
			x5469 = x235.forward(x5470);

			// val X8896 = Convolv(1,0)(X8895,3d_c_cv_W,3d_c_cv_B)
			JCudaTensor x5471;
			JCudaTensor x5472, x5473, x5474;
			x5472 = x5469;
			x5473 = x411;
			x5474 = x412;
			x5471 = x258.forward(x5472, x5473, x5474);

			// Dealloc(X8895)
			JCudaTensor x5475;
			x5475 = x5469;
			x5475.free();

			// val X8897 = BatchNorm(3d_c_bn)(X8896,3d_c_bn_scale,3d_c_bn_bias)
			JCudaTensor x5476;
			JCudaTensor x5477, x5478, x5479;
			x5477 = x5471;
			x5478 = x417;
			x5479 = x418;
			x5476 = x419.forward_inference(x5477, x5478, x5479);

			// Dealloc(X8896)
			JCudaTensor x5480;
			x5480 = x5471;
			x5480.free();

			// val X8898 = ReLU()(X8897)
			JCudaTensor x5481;
			JCudaTensor x5482;
			x5482 = x5476;
			x5481 = x268.forward(x5482);

			// val X8899 = (X8898 + X8889)
			JCudaTensor x5483;
			JCudaTensor x5484, x5485;
			x5484 = x5481;
			x5485 = x5446;
			x5483 = x5484.plus_i(x5485);

			// Dealloc(X8889)
			JCudaTensor x5486;
			x5486 = x5446;
			x5486.free();

			// val X8900 = ReLU()(X8899)
			JCudaTensor x5487;
			JCudaTensor x5488;
			x5488 = x5483;
			x5487 = x268.forward(x5488);

			// val X8904 = Convolv(2,0)(X8900,4a2_a_cv_W,4a2_a_cv_B)
			JCudaTensor x5489;
			JCudaTensor x5490, x5491, x5492;
			x5490 = x5487;
			x5491 = x438;
			x5492 = x439;
			x5489 = x440.forward(x5490, x5491, x5492);

			// val X8901 = Convolv(2,0)(X8900,4a1_cv_W,4a1_cv_B)
			JCudaTensor x5493;
			JCudaTensor x5494, x5495, x5496;
			x5494 = x5487;
			x5495 = x431;
			x5496 = x432;
			x5493 = x433.forward(x5494, x5495, x5496);

			// Dealloc(X8900)
			JCudaTensor x5497;
			x5497 = x5487;
			x5497.free();

			// val X8905 = BatchNorm(4a2_a_bn)(X8904,4a2_a_bn_scale,4a2_a_bn_bias)
			JCudaTensor x5498;
			JCudaTensor x5499, x5500, x5501;
			x5499 = x5489;
			x5500 = x452;
			x5501 = x453;
			x5498 = x454.forward_inference(x5499, x5500, x5501);

			// Dealloc(X8904)
			JCudaTensor x5502;
			x5502 = x5489;
			x5502.free();

			// val X8902 = BatchNorm(4a1_bn)(X8901,4a1_bn_scale,4a1_bn_bias)
			JCudaTensor x5503;
			JCudaTensor x5504, x5505, x5506;
			x5504 = x5493;
			x5505 = x445;
			x5506 = x446;
			x5503 = x447.forward_inference(x5504, x5505, x5506);

			// Dealloc(X8901)
			JCudaTensor x5507;
			x5507 = x5493;
			x5507.free();

			// val X8906 = ReLU()(X8905)
			JCudaTensor x5508;
			JCudaTensor x5509;
			x5509 = x5498;
			x5508 = x457.forward(x5509);

			// val X8907 = Convolv(1,1)(X8906,4a2_b_cv_W,4a2_b_cv_B)
			JCudaTensor x5510;
			JCudaTensor x5511, x5512, x5513;
			x5511 = x5508;
			x5512 = x462;
			x5513 = x463;
			x5510 = x464.forward(x5511, x5512, x5513);

			// Dealloc(X8906)
			JCudaTensor x5514;
			x5514 = x5508;
			x5514.free();

			// val X8908 = BatchNorm(4a2_b_bn)(X8907,4a2_b_bn_scale,4a2_b_bn_bias)
			JCudaTensor x5515;
			JCudaTensor x5516, x5517, x5518;
			x5516 = x5510;
			x5517 = x469;
			x5518 = x470;
			x5515 = x471.forward_inference(x5516, x5517, x5518);

			// Dealloc(X8907)
			JCudaTensor x5519;
			x5519 = x5510;
			x5519.free();

			// val X8909 = ReLU()(X8908)
			JCudaTensor x5520;
			JCudaTensor x5521;
			x5521 = x5515;
			x5520 = x457.forward(x5521);

			// val X8910 = Convolv(1,0)(X8909,4a2_c_cv_W,4a2_c_cv_B)
			JCudaTensor x5522;
			JCudaTensor x5523, x5524, x5525;
			x5523 = x5520;
			x5524 = x478;
			x5525 = x479;
			x5522 = x480.forward(x5523, x5524, x5525);

			// Dealloc(X8909)
			JCudaTensor x5526;
			x5526 = x5520;
			x5526.free();

			// val X8911 = BatchNorm(4a2_c_bn)(X8910,4a2_c_bn_scale,4a2_c_bn_bias)
			JCudaTensor x5527;
			JCudaTensor x5528, x5529, x5530;
			x5528 = x5522;
			x5529 = x485;
			x5530 = x486;
			x5527 = x487.forward_inference(x5528, x5529, x5530);

			// Dealloc(X8910)
			JCudaTensor x5531;
			x5531 = x5522;
			x5531.free();

			// val X8903 = ReLU()(X8902)
			JCudaTensor x5532;
			JCudaTensor x5533;
			x5533 = x5503;
			x5532 = x490.forward(x5533);

			// val X8912 = ReLU()(X8911)
			JCudaTensor x5534;
			JCudaTensor x5535;
			x5535 = x5527;
			x5534 = x490.forward(x5535);

			// val X8913 = (X8903 + X8912)
			JCudaTensor x5536;
			JCudaTensor x5537, x5538;
			x5537 = x5532;
			x5538 = x5534;
			x5536 = x5537.plus_i(x5538);

			// Dealloc(X8912)
			JCudaTensor x5539;
			x5539 = x5534;
			x5539.free();

			// val X8914 = ReLU()(X8913)
			JCudaTensor x5540;
			JCudaTensor x5541;
			x5541 = x5536;
			x5540 = x490.forward(x5541);

			// val X8915 = Convolv(1,0)(X8914,4b_a_cv_W,4b_a_cv_B)
			JCudaTensor x5542;
			JCudaTensor x5543, x5544, x5545;
			x5543 = x5540;
			x5544 = x502;
			x5545 = x503;
			x5542 = x504.forward(x5543, x5544, x5545);

			// val X8916 = BatchNorm(4b_a_bn)(X8915,4b_a_bn_scale,4b_a_bn_bias)
			JCudaTensor x5546;
			JCudaTensor x5547, x5548, x5549;
			x5547 = x5542;
			x5548 = x509;
			x5549 = x510;
			x5546 = x511.forward_inference(x5547, x5548, x5549);

			// Dealloc(X8915)
			JCudaTensor x5550;
			x5550 = x5542;
			x5550.free();

			// val X8917 = ReLU()(X8916)
			JCudaTensor x5551;
			JCudaTensor x5552;
			x5552 = x5546;
			x5551 = x457.forward(x5552);

			// val X8918 = Convolv(1,1)(X8917,4b_b_cv_W,4b_b_cv_B)
			JCudaTensor x5553;
			JCudaTensor x5554, x5555, x5556;
			x5554 = x5551;
			x5555 = x518;
			x5556 = x519;
			x5553 = x464.forward(x5554, x5555, x5556);

			// Dealloc(X8917)
			JCudaTensor x5557;
			x5557 = x5551;
			x5557.free();

			// val X8919 = BatchNorm(4b_b_bn)(X8918,4b_b_bn_scale,4b_b_bn_bias)
			JCudaTensor x5558;
			JCudaTensor x5559, x5560, x5561;
			x5559 = x5553;
			x5560 = x524;
			x5561 = x525;
			x5558 = x526.forward_inference(x5559, x5560, x5561);

			// Dealloc(X8918)
			JCudaTensor x5562;
			x5562 = x5553;
			x5562.free();

			// val X8920 = ReLU()(X8919)
			JCudaTensor x5563;
			JCudaTensor x5564;
			x5564 = x5558;
			x5563 = x457.forward(x5564);

			// val X8921 = Convolv(1,0)(X8920,4b_c_cv_W,4b_c_cv_B)
			JCudaTensor x5565;
			JCudaTensor x5566, x5567, x5568;
			x5566 = x5563;
			x5567 = x533;
			x5568 = x534;
			x5565 = x480.forward(x5566, x5567, x5568);

			// Dealloc(X8920)
			JCudaTensor x5569;
			x5569 = x5563;
			x5569.free();

			// val X8922 = BatchNorm(4b_c_bn)(X8921,4b_c_bn_scale,4b_c_bn_bias)
			JCudaTensor x5570;
			JCudaTensor x5571, x5572, x5573;
			x5571 = x5565;
			x5572 = x539;
			x5573 = x540;
			x5570 = x541.forward_inference(x5571, x5572, x5573);

			// Dealloc(X8921)
			JCudaTensor x5574;
			x5574 = x5565;
			x5574.free();

			// val X8923 = ReLU()(X8922)
			JCudaTensor x5575;
			JCudaTensor x5576;
			x5576 = x5570;
			x5575 = x490.forward(x5576);

			// val X8924 = (X8923 + X8914)
			JCudaTensor x5577;
			JCudaTensor x5578, x5579;
			x5578 = x5575;
			x5579 = x5540;
			x5577 = x5578.plus_i(x5579);

			// Dealloc(X8914)
			JCudaTensor x5580;
			x5580 = x5540;
			x5580.free();

			// val X8925 = ReLU()(X8924)
			JCudaTensor x5581;
			JCudaTensor x5582;
			x5582 = x5577;
			x5581 = x490.forward(x5582);

			// val X8926 = Convolv(1,0)(X8925,4c_a_cv_W,4c_a_cv_B)
			JCudaTensor x5583;
			JCudaTensor x5584, x5585, x5586;
			x5584 = x5581;
			x5585 = x553;
			x5586 = x554;
			x5583 = x504.forward(x5584, x5585, x5586);

			// val X8927 = BatchNorm(4c_a_bn)(X8926,4c_a_bn_scale,4c_a_bn_bias)
			JCudaTensor x5587;
			JCudaTensor x5588, x5589, x5590;
			x5588 = x5583;
			x5589 = x559;
			x5590 = x560;
			x5587 = x561.forward_inference(x5588, x5589, x5590);

			// Dealloc(X8926)
			JCudaTensor x5591;
			x5591 = x5583;
			x5591.free();

			// val X8928 = ReLU()(X8927)
			JCudaTensor x5592;
			JCudaTensor x5593;
			x5593 = x5587;
			x5592 = x457.forward(x5593);

			// val X8929 = Convolv(1,1)(X8928,4c_b_cv_W,4c_b_cv_B)
			JCudaTensor x5594;
			JCudaTensor x5595, x5596, x5597;
			x5595 = x5592;
			x5596 = x568;
			x5597 = x569;
			x5594 = x464.forward(x5595, x5596, x5597);

			// Dealloc(X8928)
			JCudaTensor x5598;
			x5598 = x5592;
			x5598.free();

			// val X8930 = BatchNorm(4c_b_bn)(X8929,4c_b_bn_scale,4c_b_bn_bias)
			JCudaTensor x5599;
			JCudaTensor x5600, x5601, x5602;
			x5600 = x5594;
			x5601 = x574;
			x5602 = x575;
			x5599 = x576.forward_inference(x5600, x5601, x5602);

			// Dealloc(X8929)
			JCudaTensor x5603;
			x5603 = x5594;
			x5603.free();

			// val X8931 = ReLU()(X8930)
			JCudaTensor x5604;
			JCudaTensor x5605;
			x5605 = x5599;
			x5604 = x457.forward(x5605);

			// val X8932 = Convolv(1,0)(X8931,4c_c_cv_W,4c_c_cv_B)
			JCudaTensor x5606;
			JCudaTensor x5607, x5608, x5609;
			x5607 = x5604;
			x5608 = x583;
			x5609 = x584;
			x5606 = x480.forward(x5607, x5608, x5609);

			// Dealloc(X8931)
			JCudaTensor x5610;
			x5610 = x5604;
			x5610.free();

			// val X8933 = BatchNorm(4c_c_bn)(X8932,4c_c_bn_scale,4c_c_bn_bias)
			JCudaTensor x5611;
			JCudaTensor x5612, x5613, x5614;
			x5612 = x5606;
			x5613 = x589;
			x5614 = x590;
			x5611 = x591.forward_inference(x5612, x5613, x5614);

			// Dealloc(X8932)
			JCudaTensor x5615;
			x5615 = x5606;
			x5615.free();

			// val X8934 = ReLU()(X8933)
			JCudaTensor x5616;
			JCudaTensor x5617;
			x5617 = x5611;
			x5616 = x490.forward(x5617);

			// val X8935 = (X8934 + X8925)
			JCudaTensor x5618;
			JCudaTensor x5619, x5620;
			x5619 = x5616;
			x5620 = x5581;
			x5618 = x5619.plus_i(x5620);

			// Dealloc(X8925)
			JCudaTensor x5621;
			x5621 = x5581;
			x5621.free();

			// val X8936 = ReLU()(X8935)
			JCudaTensor x5622;
			JCudaTensor x5623;
			x5623 = x5618;
			x5622 = x490.forward(x5623);

			// val X8937 = Convolv(1,0)(X8936,4d_a_cv_W,4d_a_cv_B)
			JCudaTensor x5624;
			JCudaTensor x5625, x5626, x5627;
			x5625 = x5622;
			x5626 = x603;
			x5627 = x604;
			x5624 = x504.forward(x5625, x5626, x5627);

			// val X8938 = BatchNorm(4d_a_bn)(X8937,4d_a_bn_scale,4d_a_bn_bias)
			JCudaTensor x5628;
			JCudaTensor x5629, x5630, x5631;
			x5629 = x5624;
			x5630 = x609;
			x5631 = x610;
			x5628 = x611.forward_inference(x5629, x5630, x5631);

			// Dealloc(X8937)
			JCudaTensor x5632;
			x5632 = x5624;
			x5632.free();

			// val X8939 = ReLU()(X8938)
			JCudaTensor x5633;
			JCudaTensor x5634;
			x5634 = x5628;
			x5633 = x457.forward(x5634);

			// val X8940 = Convolv(1,1)(X8939,4d_b_cv_W,4d_b_cv_B)
			JCudaTensor x5635;
			JCudaTensor x5636, x5637, x5638;
			x5636 = x5633;
			x5637 = x618;
			x5638 = x619;
			x5635 = x464.forward(x5636, x5637, x5638);

			// Dealloc(X8939)
			JCudaTensor x5639;
			x5639 = x5633;
			x5639.free();

			// val X8941 = BatchNorm(4d_b_bn)(X8940,4d_b_bn_scale,4d_b_bn_bias)
			JCudaTensor x5640;
			JCudaTensor x5641, x5642, x5643;
			x5641 = x5635;
			x5642 = x624;
			x5643 = x625;
			x5640 = x626.forward_inference(x5641, x5642, x5643);

			// Dealloc(X8940)
			JCudaTensor x5644;
			x5644 = x5635;
			x5644.free();

			// val X8942 = ReLU()(X8941)
			JCudaTensor x5645;
			JCudaTensor x5646;
			x5646 = x5640;
			x5645 = x457.forward(x5646);

			// val X8943 = Convolv(1,0)(X8942,4d_c_cv_W,4d_c_cv_B)
			JCudaTensor x5647;
			JCudaTensor x5648, x5649, x5650;
			x5648 = x5645;
			x5649 = x633;
			x5650 = x634;
			x5647 = x480.forward(x5648, x5649, x5650);

			// Dealloc(X8942)
			JCudaTensor x5651;
			x5651 = x5645;
			x5651.free();

			// val X8944 = BatchNorm(4d_c_bn)(X8943,4d_c_bn_scale,4d_c_bn_bias)
			JCudaTensor x5652;
			JCudaTensor x5653, x5654, x5655;
			x5653 = x5647;
			x5654 = x639;
			x5655 = x640;
			x5652 = x641.forward_inference(x5653, x5654, x5655);

			// Dealloc(X8943)
			JCudaTensor x5656;
			x5656 = x5647;
			x5656.free();

			// val X8945 = ReLU()(X8944)
			JCudaTensor x5657;
			JCudaTensor x5658;
			x5658 = x5652;
			x5657 = x490.forward(x5658);

			// val X8946 = (X8945 + X8936)
			JCudaTensor x5659;
			JCudaTensor x5660, x5661;
			x5660 = x5657;
			x5661 = x5622;
			x5659 = x5660.plus_i(x5661);

			// Dealloc(X8936)
			JCudaTensor x5662;
			x5662 = x5622;
			x5662.free();

			// val X8947 = ReLU()(X8946)
			JCudaTensor x5663;
			JCudaTensor x5664;
			x5664 = x5659;
			x5663 = x490.forward(x5664);

			// val X8948 = Convolv(1,0)(X8947,4e_a_cv_W,4e_a_cv_B)
			JCudaTensor x5665;
			JCudaTensor x5666, x5667, x5668;
			x5666 = x5663;
			x5667 = x653;
			x5668 = x654;
			x5665 = x504.forward(x5666, x5667, x5668);

			// val X8949 = BatchNorm(4e_a_bn)(X8948,4e_a_bn_scale,4e_a_bn_bias)
			JCudaTensor x5669;
			JCudaTensor x5670, x5671, x5672;
			x5670 = x5665;
			x5671 = x659;
			x5672 = x660;
			x5669 = x661.forward_inference(x5670, x5671, x5672);

			// Dealloc(X8948)
			JCudaTensor x5673;
			x5673 = x5665;
			x5673.free();

			// val X8950 = ReLU()(X8949)
			JCudaTensor x5674;
			JCudaTensor x5675;
			x5675 = x5669;
			x5674 = x457.forward(x5675);

			// val X8951 = Convolv(1,1)(X8950,4e_b_cv_W,4e_b_cv_B)
			JCudaTensor x5676;
			JCudaTensor x5677, x5678, x5679;
			x5677 = x5674;
			x5678 = x668;
			x5679 = x669;
			x5676 = x464.forward(x5677, x5678, x5679);

			// Dealloc(X8950)
			JCudaTensor x5680;
			x5680 = x5674;
			x5680.free();

			// val X8952 = BatchNorm(4e_b_bn)(X8951,4e_b_bn_scale,4e_b_bn_bias)
			JCudaTensor x5681;
			JCudaTensor x5682, x5683, x5684;
			x5682 = x5676;
			x5683 = x674;
			x5684 = x675;
			x5681 = x676.forward_inference(x5682, x5683, x5684);

			// Dealloc(X8951)
			JCudaTensor x5685;
			x5685 = x5676;
			x5685.free();

			// val X8953 = ReLU()(X8952)
			JCudaTensor x5686;
			JCudaTensor x5687;
			x5687 = x5681;
			x5686 = x457.forward(x5687);

			// val X8954 = Convolv(1,0)(X8953,4e_c_cv_W,4e_c_cv_B)
			JCudaTensor x5688;
			JCudaTensor x5689, x5690, x5691;
			x5689 = x5686;
			x5690 = x683;
			x5691 = x684;
			x5688 = x480.forward(x5689, x5690, x5691);

			// Dealloc(X8953)
			JCudaTensor x5692;
			x5692 = x5686;
			x5692.free();

			// val X8955 = BatchNorm(4e_c_bn)(X8954,4e_c_bn_scale,4e_c_bn_bias)
			JCudaTensor x5693;
			JCudaTensor x5694, x5695, x5696;
			x5694 = x5688;
			x5695 = x689;
			x5696 = x690;
			x5693 = x691.forward_inference(x5694, x5695, x5696);

			// Dealloc(X8954)
			JCudaTensor x5697;
			x5697 = x5688;
			x5697.free();

			// val X8956 = ReLU()(X8955)
			JCudaTensor x5698;
			JCudaTensor x5699;
			x5699 = x5693;
			x5698 = x490.forward(x5699);

			// val X8957 = (X8956 + X8947)
			JCudaTensor x5700;
			JCudaTensor x5701, x5702;
			x5701 = x5698;
			x5702 = x5663;
			x5700 = x5701.plus_i(x5702);

			// Dealloc(X8947)
			JCudaTensor x5703;
			x5703 = x5663;
			x5703.free();

			// val X8958 = ReLU()(X8957)
			JCudaTensor x5704;
			JCudaTensor x5705;
			x5705 = x5700;
			x5704 = x490.forward(x5705);

			// val X8959 = Convolv(1,0)(X8958,4f_a_cv_W,4f_a_cv_B)
			JCudaTensor x5706;
			JCudaTensor x5707, x5708, x5709;
			x5707 = x5704;
			x5708 = x703;
			x5709 = x704;
			x5706 = x504.forward(x5707, x5708, x5709);

			// val X8960 = BatchNorm(4f_a_bn)(X8959,4f_a_bn_scale,4f_a_bn_bias)
			JCudaTensor x5710;
			JCudaTensor x5711, x5712, x5713;
			x5711 = x5706;
			x5712 = x709;
			x5713 = x710;
			x5710 = x711.forward_inference(x5711, x5712, x5713);

			// Dealloc(X8959)
			JCudaTensor x5714;
			x5714 = x5706;
			x5714.free();

			// val X8961 = ReLU()(X8960)
			JCudaTensor x5715;
			JCudaTensor x5716;
			x5716 = x5710;
			x5715 = x457.forward(x5716);

			// val X8962 = Convolv(1,1)(X8961,4f_b_cv_W,4f_b_cv_B)
			JCudaTensor x5717;
			JCudaTensor x5718, x5719, x5720;
			x5718 = x5715;
			x5719 = x718;
			x5720 = x719;
			x5717 = x464.forward(x5718, x5719, x5720);

			// Dealloc(X8961)
			JCudaTensor x5721;
			x5721 = x5715;
			x5721.free();

			// val X8963 = BatchNorm(4f_b_bn)(X8962,4f_b_bn_scale,4f_b_bn_bias)
			JCudaTensor x5722;
			JCudaTensor x5723, x5724, x5725;
			x5723 = x5717;
			x5724 = x724;
			x5725 = x725;
			x5722 = x726.forward_inference(x5723, x5724, x5725);

			// Dealloc(X8962)
			JCudaTensor x5726;
			x5726 = x5717;
			x5726.free();

			// val X8964 = ReLU()(X8963)
			JCudaTensor x5727;
			JCudaTensor x5728;
			x5728 = x5722;
			x5727 = x457.forward(x5728);

			// val X8965 = Convolv(1,0)(X8964,4f_c_cv_W,4f_c_cv_B)
			JCudaTensor x5729;
			JCudaTensor x5730, x5731, x5732;
			x5730 = x5727;
			x5731 = x733;
			x5732 = x734;
			x5729 = x480.forward(x5730, x5731, x5732);

			// Dealloc(X8964)
			JCudaTensor x5733;
			x5733 = x5727;
			x5733.free();

			// val X8966 = BatchNorm(4f_c_bn)(X8965,4f_c_bn_scale,4f_c_bn_bias)
			JCudaTensor x5734;
			JCudaTensor x5735, x5736, x5737;
			x5735 = x5729;
			x5736 = x739;
			x5737 = x740;
			x5734 = x741.forward_inference(x5735, x5736, x5737);

			// Dealloc(X8965)
			JCudaTensor x5738;
			x5738 = x5729;
			x5738.free();

			// val X8967 = ReLU()(X8966)
			JCudaTensor x5739;
			JCudaTensor x5740;
			x5740 = x5734;
			x5739 = x490.forward(x5740);

			// val X8968 = (X8967 + X8958)
			JCudaTensor x5741;
			JCudaTensor x5742, x5743;
			x5742 = x5739;
			x5743 = x5704;
			x5741 = x5742.plus_i(x5743);

			// Dealloc(X8958)
			JCudaTensor x5744;
			x5744 = x5704;
			x5744.free();

			// val X8969 = ReLU()(X8968)
			JCudaTensor x5745;
			JCudaTensor x5746;
			x5746 = x5741;
			x5745 = x490.forward(x5746);

			// val X8970 = Convolv(2,0)(X8969,5a1_cv_W,5a1_cv_B)
			JCudaTensor x5747;
			JCudaTensor x5748, x5749, x5750;
			x5748 = x5745;
			x5749 = x760;
			x5750 = x761;
			x5747 = x762.forward(x5748, x5749, x5750);

			// val X8973 = Convolv(2,0)(X8969,5a2_a_cv_W,5a2_a_cv_B)
			JCudaTensor x5751;
			JCudaTensor x5752, x5753, x5754;
			x5752 = x5745;
			x5753 = x753;
			x5754 = x754;
			x5751 = x755.forward(x5752, x5753, x5754);

			// Dealloc(X8969)
			JCudaTensor x5755;
			x5755 = x5745;
			x5755.free();

			// val X8974 = BatchNorm(5a2_a_bn)(X8973,5a2_a_bn_scale,5a2_a_bn_bias)
			JCudaTensor x5756;
			JCudaTensor x5757, x5758, x5759;
			x5757 = x5751;
			x5758 = x774;
			x5759 = x775;
			x5756 = x776.forward_inference(x5757, x5758, x5759);

			// Dealloc(X8973)
			JCudaTensor x5760;
			x5760 = x5751;
			x5760.free();

			// val X8971 = BatchNorm(5a1_bn)(X8970,5a1_bn_scale,5a1_bn_bias)
			JCudaTensor x5761;
			JCudaTensor x5762, x5763, x5764;
			x5762 = x5747;
			x5763 = x767;
			x5764 = x768;
			x5761 = x769.forward_inference(x5762, x5763, x5764);

			// Dealloc(X8970)
			JCudaTensor x5765;
			x5765 = x5747;
			x5765.free();

			// val X8975 = ReLU()(X8974)
			JCudaTensor x5766;
			JCudaTensor x5767;
			x5767 = x5756;
			x5766 = x779.forward(x5767);

			// val X8976 = Convolv(1,1)(X8975,5a2_b_cv_W,5a2_b_cv_B)
			JCudaTensor x5768;
			JCudaTensor x5769, x5770, x5771;
			x5769 = x5766;
			x5770 = x784;
			x5771 = x785;
			x5768 = x786.forward(x5769, x5770, x5771);

			// Dealloc(X8975)
			JCudaTensor x5772;
			x5772 = x5766;
			x5772.free();

			// val X8977 = BatchNorm(5a2_b_bn)(X8976,5a2_b_bn_scale,5a2_b_bn_bias)
			JCudaTensor x5773;
			JCudaTensor x5774, x5775, x5776;
			x5774 = x5768;
			x5775 = x791;
			x5776 = x792;
			x5773 = x793.forward_inference(x5774, x5775, x5776);

			// Dealloc(X8976)
			JCudaTensor x5777;
			x5777 = x5768;
			x5777.free();

			// val X8978 = ReLU()(X8977)
			JCudaTensor x5778;
			JCudaTensor x5779;
			x5779 = x5773;
			x5778 = x779.forward(x5779);

			// val X8979 = Convolv(1,0)(X8978,5a2_c_cv_W,5a2_c_cv_B)
			JCudaTensor x5780;
			JCudaTensor x5781, x5782, x5783;
			x5781 = x5778;
			x5782 = x800;
			x5783 = x801;
			x5780 = x802.forward(x5781, x5782, x5783);

			// Dealloc(X8978)
			JCudaTensor x5784;
			x5784 = x5778;
			x5784.free();

			// val X8980 = BatchNorm(5a2_c_bn)(X8979,5a2_c_bn_scale,5a2_c_bn_bias)
			JCudaTensor x5785;
			JCudaTensor x5786, x5787, x5788;
			x5786 = x5780;
			x5787 = x807;
			x5788 = x808;
			x5785 = x809.forward_inference(x5786, x5787, x5788);

			// Dealloc(X8979)
			JCudaTensor x5789;
			x5789 = x5780;
			x5789.free();

			// val X8972 = ReLU()(X8971)
			JCudaTensor x5790;
			JCudaTensor x5791;
			x5791 = x5761;
			x5790 = x812.forward(x5791);

			// val X8981 = ReLU()(X8980)
			JCudaTensor x5792;
			JCudaTensor x5793;
			x5793 = x5785;
			x5792 = x812.forward(x5793);

			// val X8982 = (X8972 + X8981)
			JCudaTensor x5794;
			JCudaTensor x5795, x5796;
			x5795 = x5790;
			x5796 = x5792;
			x5794 = x5795.plus_i(x5796);

			// Dealloc(X8981)
			JCudaTensor x5797;
			x5797 = x5792;
			x5797.free();

			// val X8983 = ReLU()(X8982)
			JCudaTensor x5798;
			JCudaTensor x5799;
			x5799 = x5794;
			x5798 = x812.forward(x5799);

			// val X8984 = Convolv(1,0)(X8983,5b_a_cv_W,5b_a_cv_B)
			JCudaTensor x5800;
			JCudaTensor x5801, x5802, x5803;
			x5801 = x5798;
			x5802 = x824;
			x5803 = x825;
			x5800 = x826.forward(x5801, x5802, x5803);

			// val X8985 = BatchNorm(5b_a_bn)(X8984,5b_a_bn_scale,5b_a_bn_bias)
			JCudaTensor x5804;
			JCudaTensor x5805, x5806, x5807;
			x5805 = x5800;
			x5806 = x831;
			x5807 = x832;
			x5804 = x833.forward_inference(x5805, x5806, x5807);

			// Dealloc(X8984)
			JCudaTensor x5808;
			x5808 = x5800;
			x5808.free();

			// val X8986 = ReLU()(X8985)
			JCudaTensor x5809;
			JCudaTensor x5810;
			x5810 = x5804;
			x5809 = x779.forward(x5810);

			// val X8987 = Convolv(1,1)(X8986,5b_b_cv_W,5b_b_cv_B)
			JCudaTensor x5811;
			JCudaTensor x5812, x5813, x5814;
			x5812 = x5809;
			x5813 = x840;
			x5814 = x841;
			x5811 = x786.forward(x5812, x5813, x5814);

			// Dealloc(X8986)
			JCudaTensor x5815;
			x5815 = x5809;
			x5815.free();

			// val X8988 = BatchNorm(5b_b_bn)(X8987,5b_b_bn_scale,5b_b_bn_bias)
			JCudaTensor x5816;
			JCudaTensor x5817, x5818, x5819;
			x5817 = x5811;
			x5818 = x846;
			x5819 = x847;
			x5816 = x848.forward_inference(x5817, x5818, x5819);

			// Dealloc(X8987)
			JCudaTensor x5820;
			x5820 = x5811;
			x5820.free();

			// val X8989 = ReLU()(X8988)
			JCudaTensor x5821;
			JCudaTensor x5822;
			x5822 = x5816;
			x5821 = x779.forward(x5822);

			// val X8990 = Convolv(1,0)(X8989,5b_c_cv_W,5b_c_cv_B)
			JCudaTensor x5823;
			JCudaTensor x5824, x5825, x5826;
			x5824 = x5821;
			x5825 = x855;
			x5826 = x856;
			x5823 = x802.forward(x5824, x5825, x5826);

			// Dealloc(X8989)
			JCudaTensor x5827;
			x5827 = x5821;
			x5827.free();

			// val X8991 = BatchNorm(5b_c_bn)(X8990,5b_c_bn_scale,5b_c_bn_bias)
			JCudaTensor x5828;
			JCudaTensor x5829, x5830, x5831;
			x5829 = x5823;
			x5830 = x861;
			x5831 = x862;
			x5828 = x863.forward_inference(x5829, x5830, x5831);

			// Dealloc(X8990)
			JCudaTensor x5832;
			x5832 = x5823;
			x5832.free();

			// val X8992 = ReLU()(X8991)
			JCudaTensor x5833;
			JCudaTensor x5834;
			x5834 = x5828;
			x5833 = x812.forward(x5834);

			// val X8993 = (X8992 + X8983)
			JCudaTensor x5835;
			JCudaTensor x5836, x5837;
			x5836 = x5833;
			x5837 = x5798;
			x5835 = x5836.plus_i(x5837);

			// Dealloc(X8983)
			JCudaTensor x5838;
			x5838 = x5798;
			x5838.free();

			// val X8994 = ReLU()(X8993)
			JCudaTensor x5839;
			JCudaTensor x5840;
			x5840 = x5835;
			x5839 = x812.forward(x5840);

			// val X8995 = Convolv(1,0)(X8994,5c_a_cv_W,5c_a_cv_B)
			JCudaTensor x5841;
			JCudaTensor x5842, x5843, x5844;
			x5842 = x5839;
			x5843 = x875;
			x5844 = x876;
			x5841 = x826.forward(x5842, x5843, x5844);

			// val X8996 = BatchNorm(5c_a_bn)(X8995,5c_a_bn_scale,5c_a_bn_bias)
			JCudaTensor x5845;
			JCudaTensor x5846, x5847, x5848;
			x5846 = x5841;
			x5847 = x881;
			x5848 = x882;
			x5845 = x883.forward_inference(x5846, x5847, x5848);

			// Dealloc(X8995)
			JCudaTensor x5849;
			x5849 = x5841;
			x5849.free();

			// val X8997 = ReLU()(X8996)
			JCudaTensor x5850;
			JCudaTensor x5851;
			x5851 = x5845;
			x5850 = x779.forward(x5851);

			// val X8998 = Convolv(1,1)(X8997,5c_b_cv_W,5c_b_cv_B)
			JCudaTensor x5852;
			JCudaTensor x5853, x5854, x5855;
			x5853 = x5850;
			x5854 = x890;
			x5855 = x891;
			x5852 = x786.forward(x5853, x5854, x5855);

			// Dealloc(X8997)
			JCudaTensor x5856;
			x5856 = x5850;
			x5856.free();

			// val X8999 = BatchNorm(5c_b_bn)(X8998,5c_b_bn_scale,5c_b_bn_bias)
			JCudaTensor x5857;
			JCudaTensor x5858, x5859, x5860;
			x5858 = x5852;
			x5859 = x896;
			x5860 = x897;
			x5857 = x898.forward_inference(x5858, x5859, x5860);

			// Dealloc(X8998)
			JCudaTensor x5861;
			x5861 = x5852;
			x5861.free();

			// val X9000 = ReLU()(X8999)
			JCudaTensor x5862;
			JCudaTensor x5863;
			x5863 = x5857;
			x5862 = x779.forward(x5863);

			// val X9001 = Convolv(1,0)(X9000,5c_c_cv_W,5c_c_cv_B)
			JCudaTensor x5864;
			JCudaTensor x5865, x5866, x5867;
			x5865 = x5862;
			x5866 = x905;
			x5867 = x906;
			x5864 = x802.forward(x5865, x5866, x5867);

			// Dealloc(X9000)
			JCudaTensor x5868;
			x5868 = x5862;
			x5868.free();

			// val X9002 = BatchNorm(5c_c_bn)(X9001,5c_c_bn_scale,5c_c_bn_bias)
			JCudaTensor x5869;
			JCudaTensor x5870, x5871, x5872;
			x5870 = x5864;
			x5871 = x911;
			x5872 = x912;
			x5869 = x913.forward_inference(x5870, x5871, x5872);

			// Dealloc(X9001)
			JCudaTensor x5873;
			x5873 = x5864;
			x5873.free();

			// val X9003 = ReLU()(X9002)
			JCudaTensor x5874;
			JCudaTensor x5875;
			x5875 = x5869;
			x5874 = x812.forward(x5875);

			// val X9004 = (X9003 + X8994)
			JCudaTensor x5876;
			JCudaTensor x5877, x5878;
			x5877 = x5874;
			x5878 = x5839;
			x5876 = x5877.plus_i(x5878);

			// Dealloc(X8994)
			JCudaTensor x5879;
			x5879 = x5839;
			x5879.free();

			// val X9005 = ReLU()(X9004)
			JCudaTensor x5880;
			JCudaTensor x5881;
			x5881 = x5876;
			x5880 = x812.forward(x5881);

			// val X9006 = Pooling(7,1,0,false)(X9005)
			JCudaTensor x5882;
			JCudaTensor x5883;
			x5883 = x5880;
			x5882 = x923.forward(x5883);

			// Dealloc(X9005)
			JCudaTensor x5884;
			x5884 = x5880;
			x5884.free();

			// val X9007 = (X9006[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x5885;
			JCudaMatrix x5886;
			JCudaMatrix x5887;
			JCudaTensor x5888;
			JCudaTensor x5889;
			x5889 = x5882;
			x5888 = x5889.flatten(1, new int[]{2048, 1, 1});
			x5886 = x5888.asMatrix(1, true);
			JCudaTensor x5890;
			x5890 = x930;
			x5887 = x5890.asMatrix(1, true);
			x5885 = x5886.times(x5887);

			// Dealloc(X9006)
			JCudaTensor x5891;
			x5891 = x5882;
			x5891.free();

			// val X9009 = (X9007 + (i) => fc_B)
			JCudaTensor x5892;
			JCudaTensor x5893, x5894;
			x5893 = x5885;
			x5894 = x934;
			x5892 = x5894.copy(64, x5893);

			// Precision(Accuracy(X9009, Y, 1))
			float x5896;
			JCudaTensor x5897;
			JTensorFloat x5898;
			x5897 = x5892;
			x5898 = x4;
			x5896 = x5897.accuracy(x5898, 1);
			System.out.println(x5 + " test precision "  + x5896);
			x5895 += x5896;

			// Dealloc(X9009)
			JCudaTensor x5899;
			x5899 = x5892;
			x5899.free();

		}
		System.out.println();
		System.out.println("average precision: " + x5895/10);
		System.out.println(); 
	}

}