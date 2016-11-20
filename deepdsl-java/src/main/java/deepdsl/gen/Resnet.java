package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Resnet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay_1
	static float decay_1 = 0.999995f;
	// lrn_rate_1
	static float lrn_rate_1 = -0.01f;
	// momentum
	static float momentum = 0.9f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/resnet";
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
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, platform, 1000, true);
	// (Lmdb(1000000,10000,Win32,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{64, 3, 224, 224}, platform, 1000, false);
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
	// Precision(Accuracy(1))
	static float x4285;
	// V_1_bn_bias
	static JCudaTensor x3525 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_1_bn_scale
	static JCudaTensor x3530 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_1_cv_W
	static JCudaTensor x3535 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
	// V_2a1_bn_bias
	static JCudaTensor x3385 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a1_bn_scale
	static JCudaTensor x3355 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a1_cv_W
	static JCudaTensor x3360 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_bn_bias
	static JCudaTensor x3487 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_bn_scale
	static JCudaTensor x3482 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_a_cv_W
	static JCudaTensor x3476 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_bn_bias
	static JCudaTensor x3426 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_bn_scale
	static JCudaTensor x3440 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2a2_b_cv_W
	static JCudaTensor x3434 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2a2_c_bn_bias
	static JCudaTensor x3365 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a2_c_bn_scale
	static JCudaTensor x3380 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2a2_c_cv_W
	static JCudaTensor x3374 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2b_a_bn_bias
	static JCudaTensor x3293 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_a_bn_scale
	static JCudaTensor x3279 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_a_cv_W
	static JCudaTensor x3287 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_2b_b_bn_bias
	static JCudaTensor x3241 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_b_bn_scale
	static JCudaTensor x3233 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2b_b_cv_W
	static JCudaTensor x3246 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2b_c_bn_bias
	static JCudaTensor x3195 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2b_c_bn_scale
	static JCudaTensor x3187 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2b_c_cv_W
	static JCudaTensor x3200 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_2c_a_bn_bias
	static JCudaTensor x3147 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_a_bn_scale
	static JCudaTensor x3136 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_a_cv_W
	static JCudaTensor x3141 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_2c_b_bn_bias
	static JCudaTensor x3096 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_b_bn_scale
	static JCudaTensor x3101 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
	// V_2c_b_cv_W
	static JCudaTensor x3090 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
	// V_2c_c_bn_bias
	static JCudaTensor x3044 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2c_c_bn_scale
	static JCudaTensor x3049 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_2c_c_cv_W
	static JCudaTensor x3054 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
	// V_3a1_bn_bias
	static JCudaTensor x2870 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a1_bn_scale
	static JCudaTensor x2903 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a1_cv_W
	static JCudaTensor x2897 = JTensor.constFloat(0.0f, 512, 256, 1, 1).asJCudaTensor();
	// V_3a2_a_bn_bias
	static JCudaTensor x3005 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_a_bn_scale
	static JCudaTensor x3000 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_a_cv_W
	static JCudaTensor x2994 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_3a2_b_bn_bias
	static JCudaTensor x2944 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_b_bn_scale
	static JCudaTensor x2949 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3a2_b_cv_W
	static JCudaTensor x2957 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3a2_c_bn_bias
	static JCudaTensor x2878 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a2_c_bn_scale
	static JCudaTensor x2883 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3a2_c_cv_W
	static JCudaTensor x2891 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3b_a_bn_bias
	static JCudaTensor x2802 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_a_bn_scale
	static JCudaTensor x2797 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_a_cv_W
	static JCudaTensor x2810 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3b_b_bn_bias
	static JCudaTensor x2756 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_b_bn_scale
	static JCudaTensor x2751 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3b_b_cv_W
	static JCudaTensor x2764 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3b_c_bn_bias
	static JCudaTensor x2708 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3b_c_bn_scale
	static JCudaTensor x2713 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3b_c_cv_W
	static JCudaTensor x2718 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3c_a_bn_bias
	static JCudaTensor x2660 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_a_bn_scale
	static JCudaTensor x2665 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_a_cv_W
	static JCudaTensor x2654 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3c_b_bn_bias
	static JCudaTensor x2619 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_b_bn_scale
	static JCudaTensor x2608 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3c_b_cv_W
	static JCudaTensor x2613 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3c_c_bn_bias
	static JCudaTensor x2559 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3c_c_bn_scale
	static JCudaTensor x2573 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3c_c_cv_W
	static JCudaTensor x2567 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_3d_a_bn_bias
	static JCudaTensor x2510 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_a_bn_scale
	static JCudaTensor x2505 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_a_cv_W
	static JCudaTensor x2518 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
	// V_3d_b_bn_bias
	static JCudaTensor x2459 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_b_bn_scale
	static JCudaTensor x2464 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
	// V_3d_b_cv_W
	static JCudaTensor x2472 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
	// V_3d_c_bn_bias
	static JCudaTensor x2422 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3d_c_bn_scale
	static JCudaTensor x2427 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_3d_c_cv_W
	static JCudaTensor x2416 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
	// V_4a1_bn_bias
	static JCudaTensor x2270 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a1_bn_scale
	static JCudaTensor x2275 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a1_cv_W
	static JCudaTensor x2247 = JTensor.constFloat(0.0f, 1024, 512, 1, 1).asJCudaTensor();
	// V_4a2_a_bn_bias
	static JCudaTensor x2372 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_a_bn_scale
	static JCudaTensor x2377 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_a_cv_W
	static JCudaTensor x2366 = JTensor.constFloat(0.0f, 256, 512, 1, 1).asJCudaTensor();
	// V_4a2_b_bn_bias
	static JCudaTensor x2330 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_b_bn_scale
	static JCudaTensor x2325 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4a2_b_cv_W
	static JCudaTensor x2319 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4a2_c_bn_bias
	static JCudaTensor x2252 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a2_c_bn_scale
	static JCudaTensor x2242 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4a2_c_cv_W
	static JCudaTensor x2257 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4b_a_bn_bias
	static JCudaTensor x2178 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_a_bn_scale
	static JCudaTensor x2183 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_a_cv_W
	static JCudaTensor x2172 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4b_b_bn_bias
	static JCudaTensor x2137 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_b_bn_scale
	static JCudaTensor x2132 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4b_b_cv_W
	static JCudaTensor x2126 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4b_c_bn_bias
	static JCudaTensor x2080 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4b_c_bn_scale
	static JCudaTensor x2091 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4b_c_cv_W
	static JCudaTensor x2085 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4c_a_bn_bias
	static JCudaTensor x2026 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_a_bn_scale
	static JCudaTensor x2031 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_a_cv_W
	static JCudaTensor x2036 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4c_b_bn_bias
	static JCudaTensor x1986 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_b_bn_scale
	static JCudaTensor x1991 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4c_b_cv_W
	static JCudaTensor x1980 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4c_c_bn_bias
	static JCudaTensor x1931 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4c_c_bn_scale
	static JCudaTensor x1945 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4c_c_cv_W
	static JCudaTensor x1939 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4d_a_bn_bias
	static JCudaTensor x1877 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_a_bn_scale
	static JCudaTensor x1891 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_a_cv_W
	static JCudaTensor x1885 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4d_b_bn_bias
	static JCudaTensor x1845 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_b_bn_scale
	static JCudaTensor x1840 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4d_b_cv_W
	static JCudaTensor x1834 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4d_c_bn_bias
	static JCudaTensor x1790 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4d_c_bn_scale
	static JCudaTensor x1785 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4d_c_cv_W
	static JCudaTensor x1798 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4e_a_bn_bias
	static JCudaTensor x1745 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_a_bn_scale
	static JCudaTensor x1740 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_a_cv_W
	static JCudaTensor x1734 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4e_b_bn_bias
	static JCudaTensor x1688 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_b_bn_scale
	static JCudaTensor x1693 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4e_b_cv_W
	static JCudaTensor x1698 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4e_c_bn_bias
	static JCudaTensor x1653 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4e_c_bn_scale
	static JCudaTensor x1648 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4e_c_cv_W
	static JCudaTensor x1642 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_4f_a_bn_bias
	static JCudaTensor x1594 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_a_bn_scale
	static JCudaTensor x1599 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_a_cv_W
	static JCudaTensor x1588 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
	// V_4f_b_bn_bias
	static JCudaTensor x1539 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_b_bn_scale
	static JCudaTensor x1544 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
	// V_4f_b_cv_W
	static JCudaTensor x1552 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
	// V_4f_c_bn_bias
	static JCudaTensor x1493 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4f_c_bn_scale
	static JCudaTensor x1507 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
	// V_4f_c_cv_W
	static JCudaTensor x1501 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
	// V_5a1_bn_bias
	static JCudaTensor x1332 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a1_bn_scale
	static JCudaTensor x1346 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a1_cv_W
	static JCudaTensor x1327 = JTensor.constFloat(0.0f, 2048, 1024, 1, 1).asJCudaTensor();
	// V_5a2_a_bn_bias
	static JCudaTensor x1442 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_a_bn_scale
	static JCudaTensor x1451 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_a_cv_W
	static JCudaTensor x1456 = JTensor.constFloat(0.0f, 512, 1024, 1, 1).asJCudaTensor();
	// V_5a2_b_bn_bias
	static JCudaTensor x1410 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_b_bn_scale
	static JCudaTensor x1405 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5a2_b_cv_W
	static JCudaTensor x1399 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5a2_c_bn_bias
	static JCudaTensor x1341 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a2_c_bn_scale
	static JCudaTensor x1355 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5a2_c_cv_W
	static JCudaTensor x1322 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_5b_a_bn_bias
	static JCudaTensor x1263 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_a_bn_scale
	static JCudaTensor x1249 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_a_cv_W
	static JCudaTensor x1257 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
	// V_5b_b_bn_bias
	static JCudaTensor x1203 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_b_bn_scale
	static JCudaTensor x1208 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5b_b_cv_W
	static JCudaTensor x1216 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5b_c_bn_bias
	static JCudaTensor x1171 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5b_c_bn_scale
	static JCudaTensor x1166 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5b_c_cv_W
	static JCudaTensor x1160 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_5c_a_bn_bias
	static JCudaTensor x1103 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_a_bn_scale
	static JCudaTensor x1111 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_a_cv_W
	static JCudaTensor x1116 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
	// V_5c_b_bn_bias
	static JCudaTensor x1057 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_b_bn_scale
	static JCudaTensor x1071 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
	// V_5c_b_cv_W
	static JCudaTensor x1065 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
	// V_5c_c_bn_bias
	static JCudaTensor x1025 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5c_c_bn_scale
	static JCudaTensor x1020 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
	// V_5c_c_cv_W
	static JCudaTensor x1014 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
	// V_fc_B
	static JCudaTensor x978 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc_W
	static JCudaTensor x972 = JTensor.constFloat(0.0f, 1000, 2048).asJCudaTensor();
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
		x3530.free();
		x934.free();
		x753.free();
		x469.free();
		x53.free();
		x733.free();
		x1642.free();
		x131.free();
		x2091.free();
		x15.free();
		x317.free();
		x3090.free();
		x2377.free();
		x1501.free();
		x2810.free();
		x1945.free();
		x116.free();
		x683.free();
		x767.free();
		x2031.free();
		x825.free();
		x2330.free();
		x855.free();
		x774.free();
		x3241.free();
		x2957.free();
		x337.free();
		x453.free();
		x140.free();
		x906.free();
		x2764.free();
		x3147.free();
		x3044.free();
		x890.free();
		x241.free();
		x703.free();
		x3476.free();
		x1991.free();
		x2567.free();
		x847.free();
		x831.free();
		x534.free();
		x180.free();
		x1355.free();
		x856.free();
		x709.free();
		x1539.free();
		x1166.free();
		x39.free();
		x2472.free();
		x841.free();
		x1552.free();
		x1057.free();
		x784.free();
		x3054.free();
		x1986.free();
		x1931.free();
		x1346.free();
		x3365.free();
		x3246.free();
		x397.free();
		x1790.free();
		x247.free();
		x224.free();
		x2608.free();
		x718.free();
		x1493.free();
		x1257.free();
		x331.free();
		x775.free();
		x2275.free();
		x175.free();
		x124.free();
		x396.free();
		x740.free();
		x891.free();
		x2270.free();
		x2080.free();
		x1594.free();
		x1160.free();
		x674.free();
		x2708.free();
		x3482.free();
		x1014.free();
		x734.free();
		x739.free();
		x125.free();
		x1171.free();
		x689.free();
		x2247.free();
		x130.free();
		x3355.free();
		x2870.free();
		x3200.free();
		x2613.free();
		x159.free();
		x280.free();
		x59.free();
		x896.free();
		x2459.free();
		x710.free();
		x3136.free();
		x2132.free();
		x875.free();
		x486.free();
		x352.free();
		x3374.free();
		x368.free();
		x231.free();
		x230.free();
		x554.free();
		x181.free();
		x518.free();
		x2242.free();
		x2319.free();
		x296.free();
		x584.free();
		x332.free();
		x2619.free();
		x353.free();
		x311.free();
		x1980.free();
		x624.free();
		x704.free();
		x2172.free();
		x2891.free();
		x1939.free();
		x668.free();
		x634.free();
		x445.free();
		x2751.free();
		x2325.free();
		x297.free();
		x2903.free();
		x1891.free();
		x2994.free();
		x1688.free();
		x2126.free();
		x145.free();
		x1203.free();
		x2372.free();
		x618.free();
		x801.free();
		x824.free();
		x1025.free();
		x609.free();
		x619.free();
		x2026.free();
		x2654.free();
		x590.free();
		x2137.free();
		x70.free();
		x403.free();
		x675.free();
		x77.free();
		x263.free();
		x248.free();
		x633.free();
		x2756.free();
		x25.free();
		x196.free();
		x3233.free();
		x911.free();
		x1785.free();
		x574.free();
		x1103.free();
		x2366.free();
		x195.free();
		x1798.free();
		x2505.free();
		x719.free();
		x3525.free();
		x690.free();
		x432.free();
		x519.free();
		x165.free();
		x1327.free();
		x26.free();
		x1065.free();
		x256.free();
		x510.free();
		x2660.free();
		x660.free();
		x257.free();
		x761.free();
		x1020.free();
		x1648.free();
		x338.free();
		x2427.free();
		x470.free();
		x318.free();
		x85.free();
		x2510.free();
		x724.free();
		x553.free();
		x367.free();
		x3101.free();
		x1399.free();
		x881.free();
		x2718.free();
		x388.free();
		x223.free();
		x3279.free();
		x2878.free();
		x166.free();
		x45.free();
		x288.free();
		x478.free();
		x60.free();
		x832.free();
		x768.free();
		x1834.free();
		x791.free();
		x479.free();
		x2518.free();
		x2183.free();
		x1599.free();
		x1877.free();
		x86.free();
		x684.free();
		x808.free();
		x978.free();
		x3000.free();
		x792.free();
		x2949.free();
		x217.free();
		x625.free();
		x603.free();
		x930.free();
		x3141.free();
		x92.free();
		x639.free();
		x108.free();
		x1341.free();
		x91.free();
		x362.free();
		x1693.free();
		x402.free();
		x1116.free();
		x52.free();
		x382.free();
		x640.free();
		x1410.free();
		x1249.free();
		x1405.free();
		x3287.free();
		x2797.free();
		x303.free();
		x46.free();
		x725.free();
		x2802.free();
		x3049.free();
		x2883.free();
		x846.free();
		x139.free();
		x439.free();
		x3440.free();
		x2257.free();
		x861.free();
		x2036.free();
		x2178.free();
		x2573.free();
		x1698.free();
		x38.free();
		x264.free();
		x446.free();
		x312.free();
		x972.free();
		x69.free();
		x1745.free();
		x115.free();
		x575.free();
		x2665.free();
		x569.free();
		x503.free();
		x438.free();
		x1208.free();
		x160.free();
		x146.free();
		x1216.free();
		x800.free();
		x109.free();
		x568.free();
		x346.free();
		x540.free();
		x1442.free();
		x3005.free();
		x3434.free();
		x2944.free();
		x418.free();
		x1071.free();
		x862.free();
		x1885.free();
		x3380.free();
		x287.free();
		x2897.free();
		x2713.free();
		x281.free();
		x1588.free();
		x840.free();
		x16.free();
		x524.free();
		x897.free();
		x502.free();
		x589.free();
		x785.free();
		x653.free();
		x905.free();
		x3195.free();
		x2252.free();
		x3187.free();
		x1845.free();
		x3535.free();
		x533.free();
		x216.free();
		x209.free();
		x604.free();
		x417.free();
		x3293.free();
		x882.free();
		x659.free();
		x463.free();
		x411.free();
		x559.free();
		x1544.free();
		x1734.free();
		x1456.free();
		x347.free();
		x1653.free();
		x412.free();
		x583.free();
		x2559.free();
		x76.free();
		x912.free();
		x1263.free();
		x190.free();
		x876.free();
		x3426.free();
		x1451.free();
		x485.free();
		x1740.free();
		x1507.free();
		x525.free();
		x560.free();
		x2464.free();
		x760.free();
		x2085.free();
		x654.free();
		x2422.free();
		x807.free();
		x1322.free();
		x3385.free();
		x509.free();
		x174.free();
		x1840.free();
		x189.free();
		x452.free();
		x302.free();
		x3360.free();
		x1332.free();
		x210.free();
		x754.free();
		x539.free();
		x3096.free();
		x3487.free();
		x669.free();
		x240.free();
		x610.free();
		x431.free();
		x1111.free();
		x2416.free();
		x462.free();
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

			// val X3605 = X3603[1<>3] * d_Pooling(7,1,0,false)(X8807,X8806)/d_X8806
			JCudaTensor x966;
			JCudaTensor x967, x968, x969;
			JCudaTensor x970;
			x970 = x962;
			x967 = x970.unflatten(1, new int[]{2048, 1, 1});
			x968 = x921;
			x969 = x919;
			x966 = x923.backward(x967, x968, x969);

			// Dealloc(X3603)
			JCudaTensor x971;
			x971 = x962;
			x971.free();

			// V_fc_W <~~ m6 * m8
			float x973, x974;
			x973 = lrn_rate_1;
			x974 = momentum;
			JCudaMatrix x975;
			JCudaMatrix x976;
			x975 = x957;
			x976 = x959;
			x975.times(x976, x972, x973, x974);

			// Dealloc(X8807)
			JCudaTensor x977;
			x977 = x921;
			x977.free();

			// V_fc_B <~~ Sum(m6)
			float x979, x980;
			x979 = lrn_rate_1;
			x980 = momentum;
			JCudaMatrix x981;
			x981 = x957;
			x981.sum(x978, x979, x980);

			// Dealloc(X3409)
			JCudaTensor x982;
			x982 = x950;
			x982.free();

			// fc_W <~~ V_fc_W
			float x983, x984;
			x983 = 1;
			x984 = decay_1;
			JCudaTensor x985;
			x985 = x972;
			x930.update(x985, x983, x984);

			// fc_B <~~ V_fc_B
			float x986, x987;
			x986 = 1;
			x987 = decay_1;
			JCudaTensor x988;
			x988 = x978;
			x934.update(x988, x986, x987);

			// val X3642 = X3605 * d_ReLU()(X8806)/d_X8805
			JCudaTensor x989;
			JCudaTensor x990, x991;
			x990 = x966;
			x991 = x919;
			x989 = x812.backward(x990, x991);

			// Dealloc(X8806)
			JCudaTensor x992;
			x992 = x919;
			x992.free();

			// val X3652 = X3642.copy * d_ReLU()(X8804)/d_X8803
			JCudaTensor x993;
			JCudaTensor x994, x995;
			x994 = x989;
			x994 = x994.clone();
			x995 = x914;
			x993 = x812.backward(x994, x995);

			// Dealloc(X8804)
			JCudaTensor x996;
			x996 = x914;
			x996.free();

			// val X8610 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_5c_c_bn_scale
			JCudaTensor x997;
			JCudaTensor x998, x999, x1000;
			x998 = x993;
			x999 = x901;
			x1000 = x911;
			JCudaTensor[] x1001 = x913.backward(x998,x999,x1000);
			x997 = x1001[1];

			// val X8609 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_5c_c_bn_bias
			JCudaTensor x1002;
			x1002 = x1001[2];

			// val X3653 = X3652 * d_BatchNorm(5c_c_bn)(X8802,5c_c_bn_scale)/d_X8802
			JCudaTensor x1006;
			x1006 = x1001[0];

			// Dealloc(X8802)
			JCudaTensor x1010;
			x1010 = x901;
			x1010.free();

			// val X3654 = X3653 * d_Convolv(1,0)(5c_c_cv_W)/d_X8801
			JCudaTensor x1011;
			JCudaTensor x1012, x1013;
			x1012 = x1006;
			x1013 = x905;
			x1011 = x802.backward_data(x1012, x1013);

			// V_5c_c_cv_W <~~ X3653 * d_Convolv(1,0)(X8801)/d_5c_c_cv_W
			float x1015, x1016;
			x1015 = lrn_rate_1;
			x1016 = momentum;
			JCudaTensor x1017, x1018;
			x1017 = x1006;
			x1018 = x899;
			x802.backward_filter(x1017, x1018, x1014, x1015, x1016);

			// Dealloc(X3653)
			JCudaTensor x1019;
			x1019 = x1006;
			x1019.free();

			// V_5c_c_bn_scale <~~ X8610
			float x1021, x1022;
			x1021 = lrn_rate_1;
			x1022 = momentum;
			JCudaTensor x1023;
			x1023 = x997;
			x1020.update(x1023, x1021, x1022);

			// Dealloc(X8610)
			JCudaTensor x1024;
			x1024 = x997;
			x1024.free();

			// V_5c_c_bn_bias <~~ X8609
			float x1026, x1027;
			x1026 = lrn_rate_1;
			x1027 = momentum;
			JCudaTensor x1028;
			x1028 = x1002;
			x1025.update(x1028, x1026, x1027);

			// Dealloc(X8609)
			JCudaTensor x1029;
			x1029 = x1002;
			x1029.free();

			// 5c_c_cv_W <~~ V_5c_c_cv_W
			float x1030, x1031;
			x1030 = 1;
			x1031 = decay_1;
			JCudaTensor x1032;
			x1032 = x1014;
			x905.update(x1032, x1030, x1031);

			// 5c_c_bn_scale <~~ V_5c_c_bn_scale
			float x1033, x1034;
			x1033 = 1;
			x1034 = decay_1;
			JCudaTensor x1035;
			x1035 = x1020;
			x911.update(x1035, x1033, x1034);

			// 5c_c_bn_bias <~~ V_5c_c_bn_bias
			float x1036, x1037;
			x1036 = 1;
			x1037 = decay_1;
			JCudaTensor x1038;
			x1038 = x1025;
			x912.update(x1038, x1036, x1037);

			// val X3658 = X3654 * d_ReLU()(X8801)/d_X8800
			JCudaTensor x1039;
			JCudaTensor x1040, x1041;
			x1040 = x1011;
			x1041 = x899;
			x1039 = x779.backward(x1040, x1041);

			// Dealloc(X8801)
			JCudaTensor x1042;
			x1042 = x899;
			x1042.free();

			// val X3659 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_X8799
			JCudaTensor x1043;
			JCudaTensor x1044, x1045, x1046;
			x1044 = x1039;
			x1045 = x886;
			x1046 = x896;
			JCudaTensor[] x1047 = x898.backward(x1044,x1045,x1046);
			x1043 = x1047[0];

			// val X8606 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_5c_b_bn_bias
			JCudaTensor x1048;
			x1048 = x1047[2];

			// val X8607 = X3658 * d_BatchNorm(5c_b_bn)(X8799,5c_b_bn_scale)/d_5c_b_bn_scale
			JCudaTensor x1052;
			x1052 = x1047[1];

			// Dealloc(X8799)
			JCudaTensor x1056;
			x1056 = x886;
			x1056.free();

			// V_5c_b_bn_bias <~~ X8606
			float x1058, x1059;
			x1058 = lrn_rate_1;
			x1059 = momentum;
			JCudaTensor x1060;
			x1060 = x1048;
			x1057.update(x1060, x1058, x1059);

			// Dealloc(X8606)
			JCudaTensor x1061;
			x1061 = x1048;
			x1061.free();

			// val X3660 = X3659 * d_Convolv(1,1)(5c_b_cv_W)/d_X8798
			JCudaTensor x1062;
			JCudaTensor x1063, x1064;
			x1063 = x1043;
			x1064 = x890;
			x1062 = x786.backward_data(x1063, x1064);

			// V_5c_b_cv_W <~~ X3659 * d_Convolv(1,1)(X8798)/d_5c_b_cv_W
			float x1066, x1067;
			x1066 = lrn_rate_1;
			x1067 = momentum;
			JCudaTensor x1068, x1069;
			x1068 = x1043;
			x1069 = x884;
			x786.backward_filter(x1068, x1069, x1065, x1066, x1067);

			// Dealloc(X3659)
			JCudaTensor x1070;
			x1070 = x1043;
			x1070.free();

			// V_5c_b_bn_scale <~~ X8607
			float x1072, x1073;
			x1072 = lrn_rate_1;
			x1073 = momentum;
			JCudaTensor x1074;
			x1074 = x1052;
			x1071.update(x1074, x1072, x1073);

			// Dealloc(X8607)
			JCudaTensor x1075;
			x1075 = x1052;
			x1075.free();

			// 5c_b_bn_bias <~~ V_5c_b_bn_bias
			float x1076, x1077;
			x1076 = 1;
			x1077 = decay_1;
			JCudaTensor x1078;
			x1078 = x1057;
			x897.update(x1078, x1076, x1077);

			// 5c_b_cv_W <~~ V_5c_b_cv_W
			float x1079, x1080;
			x1079 = 1;
			x1080 = decay_1;
			JCudaTensor x1081;
			x1081 = x1065;
			x890.update(x1081, x1079, x1080);

			// 5c_b_bn_scale <~~ V_5c_b_bn_scale
			float x1082, x1083;
			x1082 = 1;
			x1083 = decay_1;
			JCudaTensor x1084;
			x1084 = x1071;
			x896.update(x1084, x1082, x1083);

			// val X3664 = X3660 * d_ReLU()(X8798)/d_X8797
			JCudaTensor x1085;
			JCudaTensor x1086, x1087;
			x1086 = x1062;
			x1087 = x884;
			x1085 = x779.backward(x1086, x1087);

			// Dealloc(X8798)
			JCudaTensor x1088;
			x1088 = x884;
			x1088.free();

			// val X3665 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_X8796
			JCudaTensor x1089;
			JCudaTensor x1090, x1091, x1092;
			x1090 = x1085;
			x1091 = x871;
			x1092 = x881;
			JCudaTensor[] x1093 = x883.backward(x1090,x1091,x1092);
			x1089 = x1093[0];

			// val X8603 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_5c_a_bn_bias
			JCudaTensor x1094;
			x1094 = x1093[2];

			// val X8604 = X3664 * d_BatchNorm(5c_a_bn)(X8796,5c_a_bn_scale)/d_5c_a_bn_scale
			JCudaTensor x1098;
			x1098 = x1093[1];

			// Dealloc(X8796)
			JCudaTensor x1102;
			x1102 = x871;
			x1102.free();

			// V_5c_a_bn_bias <~~ X8603
			float x1104, x1105;
			x1104 = lrn_rate_1;
			x1105 = momentum;
			JCudaTensor x1106;
			x1106 = x1094;
			x1103.update(x1106, x1104, x1105);

			// Dealloc(X8603)
			JCudaTensor x1107;
			x1107 = x1094;
			x1107.free();

			// val X3666 = X3665 * d_Convolv(1,0)(5c_a_cv_W)/d_X8795
			JCudaTensor x1108;
			JCudaTensor x1109, x1110;
			x1109 = x1089;
			x1110 = x875;
			x1108 = x826.backward_data(x1109, x1110);

			// V_5c_a_bn_scale <~~ X8604
			float x1112, x1113;
			x1112 = lrn_rate_1;
			x1113 = momentum;
			JCudaTensor x1114;
			x1114 = x1098;
			x1111.update(x1114, x1112, x1113);

			// Dealloc(X8604)
			JCudaTensor x1115;
			x1115 = x1098;
			x1115.free();

			// V_5c_a_cv_W <~~ X3665 * d_Convolv(1,0)(X8795)/d_5c_a_cv_W
			float x1117, x1118;
			x1117 = lrn_rate_1;
			x1118 = momentum;
			JCudaTensor x1119, x1120;
			x1119 = x1089;
			x1120 = x869;
			x826.backward_filter(x1119, x1120, x1116, x1117, x1118);

			// Dealloc(X3665)
			JCudaTensor x1121;
			x1121 = x1089;
			x1121.free();

			// 5c_a_bn_bias <~~ V_5c_a_bn_bias
			float x1122, x1123;
			x1122 = 1;
			x1123 = decay_1;
			JCudaTensor x1124;
			x1124 = x1103;
			x882.update(x1124, x1122, x1123);

			// 5c_a_bn_scale <~~ V_5c_a_bn_scale
			float x1125, x1126;
			x1125 = 1;
			x1126 = decay_1;
			JCudaTensor x1127;
			x1127 = x1111;
			x881.update(x1127, x1125, x1126);

			// 5c_a_cv_W <~~ V_5c_a_cv_W
			float x1128, x1129;
			x1128 = 1;
			x1129 = decay_1;
			JCudaTensor x1130;
			x1130 = x1116;
			x875.update(x1130, x1128, x1129);

			// val X3667 = (X3666 + X3642)
			JCudaTensor x1131;
			JCudaTensor x1132, x1133;
			x1132 = x1108;
			x1133 = x989;
			x1131 = x1132.plus_i(x1133);

			// Dealloc(X3642)
			JCudaTensor x1134;
			x1134 = x989;
			x1134.free();

			// val X3679 = X3667 * d_ReLU()(X8795)/d_X8794
			JCudaTensor x1135;
			JCudaTensor x1136, x1137;
			x1136 = x1131;
			x1137 = x869;
			x1135 = x812.backward(x1136, x1137);

			// Dealloc(X8795)
			JCudaTensor x1138;
			x1138 = x869;
			x1138.free();

			// val X3689 = X3679.copy * d_ReLU()(X8793)/d_X8792
			JCudaTensor x1139;
			JCudaTensor x1140, x1141;
			x1140 = x1135;
			x1140 = x1140.clone();
			x1141 = x864;
			x1139 = x812.backward(x1140, x1141);

			// Dealloc(X8793)
			JCudaTensor x1142;
			x1142 = x864;
			x1142.free();

			// val X8601 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_5b_c_bn_scale
			JCudaTensor x1143;
			JCudaTensor x1144, x1145, x1146;
			x1144 = x1139;
			x1145 = x851;
			x1146 = x861;
			JCudaTensor[] x1147 = x863.backward(x1144,x1145,x1146);
			x1143 = x1147[1];

			// val X8600 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_5b_c_bn_bias
			JCudaTensor x1148;
			x1148 = x1147[2];

			// val X3690 = X3689 * d_BatchNorm(5b_c_bn)(X8791,5b_c_bn_scale)/d_X8791
			JCudaTensor x1152;
			x1152 = x1147[0];

			// Dealloc(X8791)
			JCudaTensor x1156;
			x1156 = x851;
			x1156.free();

			// val X3691 = X3690 * d_Convolv(1,0)(5b_c_cv_W)/d_X8790
			JCudaTensor x1157;
			JCudaTensor x1158, x1159;
			x1158 = x1152;
			x1159 = x855;
			x1157 = x802.backward_data(x1158, x1159);

			// V_5b_c_cv_W <~~ X3690 * d_Convolv(1,0)(X8790)/d_5b_c_cv_W
			float x1161, x1162;
			x1161 = lrn_rate_1;
			x1162 = momentum;
			JCudaTensor x1163, x1164;
			x1163 = x1152;
			x1164 = x849;
			x802.backward_filter(x1163, x1164, x1160, x1161, x1162);

			// Dealloc(X3690)
			JCudaTensor x1165;
			x1165 = x1152;
			x1165.free();

			// V_5b_c_bn_scale <~~ X8601
			float x1167, x1168;
			x1167 = lrn_rate_1;
			x1168 = momentum;
			JCudaTensor x1169;
			x1169 = x1143;
			x1166.update(x1169, x1167, x1168);

			// Dealloc(X8601)
			JCudaTensor x1170;
			x1170 = x1143;
			x1170.free();

			// V_5b_c_bn_bias <~~ X8600
			float x1172, x1173;
			x1172 = lrn_rate_1;
			x1173 = momentum;
			JCudaTensor x1174;
			x1174 = x1148;
			x1171.update(x1174, x1172, x1173);

			// Dealloc(X8600)
			JCudaTensor x1175;
			x1175 = x1148;
			x1175.free();

			// 5b_c_cv_W <~~ V_5b_c_cv_W
			float x1176, x1177;
			x1176 = 1;
			x1177 = decay_1;
			JCudaTensor x1178;
			x1178 = x1160;
			x855.update(x1178, x1176, x1177);

			// 5b_c_bn_scale <~~ V_5b_c_bn_scale
			float x1179, x1180;
			x1179 = 1;
			x1180 = decay_1;
			JCudaTensor x1181;
			x1181 = x1166;
			x861.update(x1181, x1179, x1180);

			// 5b_c_bn_bias <~~ V_5b_c_bn_bias
			float x1182, x1183;
			x1182 = 1;
			x1183 = decay_1;
			JCudaTensor x1184;
			x1184 = x1171;
			x862.update(x1184, x1182, x1183);

			// val X3695 = X3691 * d_ReLU()(X8790)/d_X8789
			JCudaTensor x1185;
			JCudaTensor x1186, x1187;
			x1186 = x1157;
			x1187 = x849;
			x1185 = x779.backward(x1186, x1187);

			// Dealloc(X8790)
			JCudaTensor x1188;
			x1188 = x849;
			x1188.free();

			// val X8598 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_5b_b_bn_scale
			JCudaTensor x1189;
			JCudaTensor x1190, x1191, x1192;
			x1190 = x1185;
			x1191 = x836;
			x1192 = x846;
			JCudaTensor[] x1193 = x848.backward(x1190,x1191,x1192);
			x1189 = x1193[1];

			// val X8597 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_5b_b_bn_bias
			JCudaTensor x1194;
			x1194 = x1193[2];

			// val X3696 = X3695 * d_BatchNorm(5b_b_bn)(X8788,5b_b_bn_scale)/d_X8788
			JCudaTensor x1198;
			x1198 = x1193[0];

			// Dealloc(X8788)
			JCudaTensor x1202;
			x1202 = x836;
			x1202.free();

			// V_5b_b_bn_bias <~~ X8597
			float x1204, x1205;
			x1204 = lrn_rate_1;
			x1205 = momentum;
			JCudaTensor x1206;
			x1206 = x1194;
			x1203.update(x1206, x1204, x1205);

			// Dealloc(X8597)
			JCudaTensor x1207;
			x1207 = x1194;
			x1207.free();

			// V_5b_b_bn_scale <~~ X8598
			float x1209, x1210;
			x1209 = lrn_rate_1;
			x1210 = momentum;
			JCudaTensor x1211;
			x1211 = x1189;
			x1208.update(x1211, x1209, x1210);

			// Dealloc(X8598)
			JCudaTensor x1212;
			x1212 = x1189;
			x1212.free();

			// val X3697 = X3696 * d_Convolv(1,1)(5b_b_cv_W)/d_X8787
			JCudaTensor x1213;
			JCudaTensor x1214, x1215;
			x1214 = x1198;
			x1215 = x840;
			x1213 = x786.backward_data(x1214, x1215);

			// V_5b_b_cv_W <~~ X3696 * d_Convolv(1,1)(X8787)/d_5b_b_cv_W
			float x1217, x1218;
			x1217 = lrn_rate_1;
			x1218 = momentum;
			JCudaTensor x1219, x1220;
			x1219 = x1198;
			x1220 = x834;
			x786.backward_filter(x1219, x1220, x1216, x1217, x1218);

			// Dealloc(X3696)
			JCudaTensor x1221;
			x1221 = x1198;
			x1221.free();

			// 5b_b_bn_bias <~~ V_5b_b_bn_bias
			float x1222, x1223;
			x1222 = 1;
			x1223 = decay_1;
			JCudaTensor x1224;
			x1224 = x1203;
			x847.update(x1224, x1222, x1223);

			// 5b_b_bn_scale <~~ V_5b_b_bn_scale
			float x1225, x1226;
			x1225 = 1;
			x1226 = decay_1;
			JCudaTensor x1227;
			x1227 = x1208;
			x846.update(x1227, x1225, x1226);

			// 5b_b_cv_W <~~ V_5b_b_cv_W
			float x1228, x1229;
			x1228 = 1;
			x1229 = decay_1;
			JCudaTensor x1230;
			x1230 = x1216;
			x840.update(x1230, x1228, x1229);

			// val X3701 = X3697 * d_ReLU()(X8787)/d_X8786
			JCudaTensor x1231;
			JCudaTensor x1232, x1233;
			x1232 = x1213;
			x1233 = x834;
			x1231 = x779.backward(x1232, x1233);

			// Dealloc(X8787)
			JCudaTensor x1234;
			x1234 = x834;
			x1234.free();

			// val X8594 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_5b_a_bn_bias
			JCudaTensor x1235;
			JCudaTensor x1236, x1237, x1238;
			x1236 = x1231;
			x1237 = x820;
			x1238 = x831;
			JCudaTensor[] x1239 = x833.backward(x1236,x1237,x1238);
			x1235 = x1239[2];

			// val X3702 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_X8785
			JCudaTensor x1240;
			x1240 = x1239[0];

			// val X8595 = X3701 * d_BatchNorm(5b_a_bn)(X8785,5b_a_bn_scale)/d_5b_a_bn_scale
			JCudaTensor x1244;
			x1244 = x1239[1];

			// Dealloc(X8785)
			JCudaTensor x1248;
			x1248 = x820;
			x1248.free();

			// V_5b_a_bn_scale <~~ X8595
			float x1250, x1251;
			x1250 = lrn_rate_1;
			x1251 = momentum;
			JCudaTensor x1252;
			x1252 = x1244;
			x1249.update(x1252, x1250, x1251);

			// Dealloc(X8595)
			JCudaTensor x1253;
			x1253 = x1244;
			x1253.free();

			// val X3703 = X3702 * d_Convolv(1,0)(5b_a_cv_W)/d_X8784
			JCudaTensor x1254;
			JCudaTensor x1255, x1256;
			x1255 = x1240;
			x1256 = x824;
			x1254 = x826.backward_data(x1255, x1256);

			// V_5b_a_cv_W <~~ X3702 * d_Convolv(1,0)(X8784)/d_5b_a_cv_W
			float x1258, x1259;
			x1258 = lrn_rate_1;
			x1259 = momentum;
			JCudaTensor x1260, x1261;
			x1260 = x1240;
			x1261 = x818;
			x826.backward_filter(x1260, x1261, x1257, x1258, x1259);

			// Dealloc(X3702)
			JCudaTensor x1262;
			x1262 = x1240;
			x1262.free();

			// V_5b_a_bn_bias <~~ X8594
			float x1264, x1265;
			x1264 = lrn_rate_1;
			x1265 = momentum;
			JCudaTensor x1266;
			x1266 = x1235;
			x1263.update(x1266, x1264, x1265);

			// Dealloc(X8594)
			JCudaTensor x1267;
			x1267 = x1235;
			x1267.free();

			// 5b_a_bn_scale <~~ V_5b_a_bn_scale
			float x1268, x1269;
			x1268 = 1;
			x1269 = decay_1;
			JCudaTensor x1270;
			x1270 = x1249;
			x831.update(x1270, x1268, x1269);

			// 5b_a_cv_W <~~ V_5b_a_cv_W
			float x1271, x1272;
			x1271 = 1;
			x1272 = decay_1;
			JCudaTensor x1273;
			x1273 = x1257;
			x824.update(x1273, x1271, x1272);

			// 5b_a_bn_bias <~~ V_5b_a_bn_bias
			float x1274, x1275;
			x1274 = 1;
			x1275 = decay_1;
			JCudaTensor x1276;
			x1276 = x1263;
			x832.update(x1276, x1274, x1275);

			// val X3704 = (X3703 + X3679)
			JCudaTensor x1277;
			JCudaTensor x1278, x1279;
			x1278 = x1254;
			x1279 = x1135;
			x1277 = x1278.plus_i(x1279);

			// Dealloc(X3679)
			JCudaTensor x1280;
			x1280 = x1135;
			x1280.free();

			// val X3719 = X3704 * d_ReLU()(X8784)/d_X8783
			JCudaTensor x1281;
			JCudaTensor x1282, x1283;
			x1282 = x1277;
			x1283 = x818;
			x1281 = x812.backward(x1282, x1283);

			// Dealloc(X8784)
			JCudaTensor x1284;
			x1284 = x818;
			x1284.free();

			// val X3723 = X3719.copy * d_ReLU()(X8773)/d_X8772
			JCudaTensor x1285;
			JCudaTensor x1286, x1287;
			x1286 = x1281;
			x1286 = x1286.clone();
			x1287 = x810;
			x1285 = x812.backward(x1286, x1287);

			// Dealloc(X8773)
			JCudaTensor x1288;
			x1288 = x810;
			x1288.free();

			// val X3735 = X3719.copy * d_ReLU()(X8782)/d_X8781
			JCudaTensor x1289;
			JCudaTensor x1290, x1291;
			x1290 = x1281;
			x1290 = x1290.clone();
			x1291 = x813;
			x1289 = x812.backward(x1290, x1291);

			// Dealloc(X3719)
			JCudaTensor x1292;
			x1292 = x1281;
			x1292.free();

			// Dealloc(X8782)
			JCudaTensor x1293;
			x1293 = x813;
			x1293.free();

			// val X8583 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_5a1_bn_scale
			JCudaTensor x1294;
			JCudaTensor x1295, x1296, x1297;
			x1295 = x1285;
			x1296 = x756;
			x1297 = x767;
			JCudaTensor[] x1298 = x769.backward(x1295,x1296,x1297);
			x1294 = x1298[1];

			// val X3724 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_X8771
			JCudaTensor x1299;
			x1299 = x1298[0];

			// val X8592 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_5a2_c_bn_scale
			JCudaTensor x1303;
			JCudaTensor x1304, x1305, x1306;
			x1304 = x1289;
			x1305 = x796;
			x1306 = x807;
			JCudaTensor[] x1307 = x809.backward(x1304,x1305,x1306);
			x1303 = x1307[1];

			// val X8582 = X3723 * d_BatchNorm(5a1_bn)(X8771,5a1_bn_scale)/d_5a1_bn_bias
			JCudaTensor x1308;
			x1308 = x1298[2];

			// Dealloc(X8771)
			JCudaTensor x1312;
			x1312 = x756;
			x1312.free();

			// val X3736 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_X8780
			JCudaTensor x1313;
			x1313 = x1307[0];

			// val X8591 = X3735 * d_BatchNorm(5a2_c_bn)(X8780,5a2_c_bn_scale)/d_5a2_c_bn_bias
			JCudaTensor x1317;
			x1317 = x1307[2];

			// Dealloc(X8780)
			JCudaTensor x1321;
			x1321 = x796;
			x1321.free();

			// V_5a2_c_cv_W <~~ X3736 * d_Convolv(1,0)(X8779)/d_5a2_c_cv_W
			float x1323, x1324;
			x1323 = lrn_rate_1;
			x1324 = momentum;
			JCudaTensor x1325, x1326;
			x1325 = x1313;
			x1326 = x794;
			x802.backward_filter(x1325, x1326, x1322, x1323, x1324);

			// V_5a1_cv_W <~~ X3724 * d_Convolv(2,0)(X8770)/d_5a1_cv_W
			float x1328, x1329;
			x1328 = lrn_rate_1;
			x1329 = momentum;
			JCudaTensor x1330, x1331;
			x1330 = x1299;
			x1331 = x747;
			x762.backward_filter(x1330, x1331, x1327, x1328, x1329);

			// V_5a1_bn_bias <~~ X8582
			float x1333, x1334;
			x1333 = lrn_rate_1;
			x1334 = momentum;
			JCudaTensor x1335;
			x1335 = x1308;
			x1332.update(x1335, x1333, x1334);

			// Dealloc(X8582)
			JCudaTensor x1336;
			x1336 = x1308;
			x1336.free();

			// val X3737 = X3736 * d_Convolv(1,0)(5a2_c_cv_W)/d_X8779
			JCudaTensor x1337;
			JCudaTensor x1338, x1339;
			x1338 = x1313;
			x1339 = x800;
			x1337 = x802.backward_data(x1338, x1339);

			// Dealloc(X3736)
			JCudaTensor x1340;
			x1340 = x1313;
			x1340.free();

			// V_5a2_c_bn_bias <~~ X8591
			float x1342, x1343;
			x1342 = lrn_rate_1;
			x1343 = momentum;
			JCudaTensor x1344;
			x1344 = x1317;
			x1341.update(x1344, x1342, x1343);

			// Dealloc(X8591)
			JCudaTensor x1345;
			x1345 = x1317;
			x1345.free();

			// V_5a1_bn_scale <~~ X8583
			float x1347, x1348;
			x1347 = lrn_rate_1;
			x1348 = momentum;
			JCudaTensor x1349;
			x1349 = x1294;
			x1346.update(x1349, x1347, x1348);

			// Dealloc(X8583)
			JCudaTensor x1350;
			x1350 = x1294;
			x1350.free();

			// val X3725 = X3724 * d_Convolv(2,0)(5a1_cv_W)/d_X8770
			JCudaTensor x1351;
			JCudaTensor x1352, x1353;
			x1352 = x1299;
			x1353 = x760;
			x1351 = x762.backward_data(x1352, x1353);

			// Dealloc(X3724)
			JCudaTensor x1354;
			x1354 = x1299;
			x1354.free();

			// V_5a2_c_bn_scale <~~ X8592
			float x1356, x1357;
			x1356 = lrn_rate_1;
			x1357 = momentum;
			JCudaTensor x1358;
			x1358 = x1303;
			x1355.update(x1358, x1356, x1357);

			// Dealloc(X8592)
			JCudaTensor x1359;
			x1359 = x1303;
			x1359.free();

			// 5a1_bn_scale <~~ V_5a1_bn_scale
			float x1360, x1361;
			x1360 = 1;
			x1361 = decay_1;
			JCudaTensor x1362;
			x1362 = x1346;
			x767.update(x1362, x1360, x1361);

			// 5a2_c_bn_scale <~~ V_5a2_c_bn_scale
			float x1363, x1364;
			x1363 = 1;
			x1364 = decay_1;
			JCudaTensor x1365;
			x1365 = x1355;
			x807.update(x1365, x1363, x1364);

			// 5a2_c_bn_bias <~~ V_5a2_c_bn_bias
			float x1366, x1367;
			x1366 = 1;
			x1367 = decay_1;
			JCudaTensor x1368;
			x1368 = x1341;
			x808.update(x1368, x1366, x1367);

			// 5a2_c_cv_W <~~ V_5a2_c_cv_W
			float x1369, x1370;
			x1369 = 1;
			x1370 = decay_1;
			JCudaTensor x1371;
			x1371 = x1322;
			x800.update(x1371, x1369, x1370);

			// 5a1_bn_bias <~~ V_5a1_bn_bias
			float x1372, x1373;
			x1372 = 1;
			x1373 = decay_1;
			JCudaTensor x1374;
			x1374 = x1332;
			x768.update(x1374, x1372, x1373);

			// 5a1_cv_W <~~ V_5a1_cv_W
			float x1375, x1376;
			x1375 = 1;
			x1376 = decay_1;
			JCudaTensor x1377;
			x1377 = x1327;
			x760.update(x1377, x1375, x1376);

			// val X3741 = X3737 * d_ReLU()(X8779)/d_X8778
			JCudaTensor x1378;
			JCudaTensor x1379, x1380;
			x1379 = x1337;
			x1380 = x794;
			x1378 = x779.backward(x1379, x1380);

			// Dealloc(X8779)
			JCudaTensor x1381;
			x1381 = x794;
			x1381.free();

			// val X8588 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_5a2_b_bn_bias
			JCudaTensor x1382;
			JCudaTensor x1383, x1384, x1385;
			x1383 = x1378;
			x1384 = x780;
			x1385 = x791;
			JCudaTensor[] x1386 = x793.backward(x1383,x1384,x1385);
			x1382 = x1386[2];

			// val X3742 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_X8777
			JCudaTensor x1387;
			x1387 = x1386[0];

			// val X8589 = X3741 * d_BatchNorm(5a2_b_bn)(X8777,5a2_b_bn_scale)/d_5a2_b_bn_scale
			JCudaTensor x1391;
			x1391 = x1386[1];

			// Dealloc(X8777)
			JCudaTensor x1395;
			x1395 = x780;
			x1395.free();

			// val X3743 = X3742 * d_Convolv(1,1)(5a2_b_cv_W)/d_X8776
			JCudaTensor x1396;
			JCudaTensor x1397, x1398;
			x1397 = x1387;
			x1398 = x784;
			x1396 = x786.backward_data(x1397, x1398);

			// V_5a2_b_cv_W <~~ X3742 * d_Convolv(1,1)(X8776)/d_5a2_b_cv_W
			float x1400, x1401;
			x1400 = lrn_rate_1;
			x1401 = momentum;
			JCudaTensor x1402, x1403;
			x1402 = x1387;
			x1403 = x777;
			x786.backward_filter(x1402, x1403, x1399, x1400, x1401);

			// Dealloc(X3742)
			JCudaTensor x1404;
			x1404 = x1387;
			x1404.free();

			// V_5a2_b_bn_scale <~~ X8589
			float x1406, x1407;
			x1406 = lrn_rate_1;
			x1407 = momentum;
			JCudaTensor x1408;
			x1408 = x1391;
			x1405.update(x1408, x1406, x1407);

			// Dealloc(X8589)
			JCudaTensor x1409;
			x1409 = x1391;
			x1409.free();

			// V_5a2_b_bn_bias <~~ X8588
			float x1411, x1412;
			x1411 = lrn_rate_1;
			x1412 = momentum;
			JCudaTensor x1413;
			x1413 = x1382;
			x1410.update(x1413, x1411, x1412);

			// Dealloc(X8588)
			JCudaTensor x1414;
			x1414 = x1382;
			x1414.free();

			// 5a2_b_cv_W <~~ V_5a2_b_cv_W
			float x1415, x1416;
			x1415 = 1;
			x1416 = decay_1;
			JCudaTensor x1417;
			x1417 = x1399;
			x784.update(x1417, x1415, x1416);

			// 5a2_b_bn_scale <~~ V_5a2_b_bn_scale
			float x1418, x1419;
			x1418 = 1;
			x1419 = decay_1;
			JCudaTensor x1420;
			x1420 = x1405;
			x791.update(x1420, x1418, x1419);

			// 5a2_b_bn_bias <~~ V_5a2_b_bn_bias
			float x1421, x1422;
			x1421 = 1;
			x1422 = decay_1;
			JCudaTensor x1423;
			x1423 = x1410;
			x792.update(x1423, x1421, x1422);

			// val X3747 = X3743 * d_ReLU()(X8776)/d_X8775
			JCudaTensor x1424;
			JCudaTensor x1425, x1426;
			x1425 = x1396;
			x1426 = x777;
			x1424 = x779.backward(x1425, x1426);

			// Dealloc(X8776)
			JCudaTensor x1427;
			x1427 = x777;
			x1427.free();

			// val X3748 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_X8774
			JCudaTensor x1428;
			JCudaTensor x1429, x1430, x1431;
			x1429 = x1424;
			x1430 = x749;
			x1431 = x774;
			JCudaTensor[] x1432 = x776.backward(x1429,x1430,x1431);
			x1428 = x1432[0];

			// val X8585 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_5a2_a_bn_bias
			JCudaTensor x1433;
			x1433 = x1432[2];

			// val X8586 = X3747 * d_BatchNorm(5a2_a_bn)(X8774,5a2_a_bn_scale)/d_5a2_a_bn_scale
			JCudaTensor x1437;
			x1437 = x1432[1];

			// Dealloc(X8774)
			JCudaTensor x1441;
			x1441 = x749;
			x1441.free();

			// V_5a2_a_bn_bias <~~ X8585
			float x1443, x1444;
			x1443 = lrn_rate_1;
			x1444 = momentum;
			JCudaTensor x1445;
			x1445 = x1433;
			x1442.update(x1445, x1443, x1444);

			// Dealloc(X8585)
			JCudaTensor x1446;
			x1446 = x1433;
			x1446.free();

			// val X3750 = (X3725 + X3748 * d_Convolv(2,0)(5a2_a_cv_W)/d_X8770)
			JCudaTensor x1447;
			JCudaTensor x1448;
			x1448 = x1351;
			JCudaTensor x1449, x1450;
			x1449 = x1428;
			x1450 = x753;
			x1447 = x755.backward_data(x1449,x1450, x1448);

			// V_5a2_a_bn_scale <~~ X8586
			float x1452, x1453;
			x1452 = lrn_rate_1;
			x1453 = momentum;
			JCudaTensor x1454;
			x1454 = x1437;
			x1451.update(x1454, x1452, x1453);

			// Dealloc(X8586)
			JCudaTensor x1455;
			x1455 = x1437;
			x1455.free();

			// V_5a2_a_cv_W <~~ X3748 * d_Convolv(2,0)(X8770)/d_5a2_a_cv_W
			float x1457, x1458;
			x1457 = lrn_rate_1;
			x1458 = momentum;
			JCudaTensor x1459, x1460;
			x1459 = x1428;
			x1460 = x747;
			x755.backward_filter(x1459, x1460, x1456, x1457, x1458);

			// Dealloc(X3748)
			JCudaTensor x1461;
			x1461 = x1428;
			x1461.free();

			// 5a2_a_bn_bias <~~ V_5a2_a_bn_bias
			float x1462, x1463;
			x1462 = 1;
			x1463 = decay_1;
			JCudaTensor x1464;
			x1464 = x1442;
			x775.update(x1464, x1462, x1463);

			// 5a2_a_bn_scale <~~ V_5a2_a_bn_scale
			float x1465, x1466;
			x1465 = 1;
			x1466 = decay_1;
			JCudaTensor x1467;
			x1467 = x1451;
			x774.update(x1467, x1465, x1466);

			// 5a2_a_cv_W <~~ V_5a2_a_cv_W
			float x1468, x1469;
			x1468 = 1;
			x1469 = decay_1;
			JCudaTensor x1470;
			x1470 = x1456;
			x753.update(x1470, x1468, x1469);

			// val X3820 = X3750 * d_ReLU()(X8770)/d_X8769
			JCudaTensor x1471;
			JCudaTensor x1472, x1473;
			x1472 = x1447;
			x1473 = x747;
			x1471 = x490.backward(x1472, x1473);

			// Dealloc(X8770)
			JCudaTensor x1474;
			x1474 = x747;
			x1474.free();

			// val X3830 = X3820.copy * d_ReLU()(X8768)/d_X8767
			JCudaTensor x1475;
			JCudaTensor x1476, x1477;
			x1476 = x1471;
			x1476 = x1476.clone();
			x1477 = x742;
			x1475 = x490.backward(x1476, x1477);

			// Dealloc(X8768)
			JCudaTensor x1478;
			x1478 = x742;
			x1478.free();

			// val X8580 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_4f_c_bn_scale
			JCudaTensor x1479;
			JCudaTensor x1480, x1481, x1482;
			x1480 = x1475;
			x1481 = x729;
			x1482 = x739;
			JCudaTensor[] x1483 = x741.backward(x1480,x1481,x1482);
			x1479 = x1483[1];

			// val X3831 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_X8766
			JCudaTensor x1484;
			x1484 = x1483[0];

			// val X8579 = X3830 * d_BatchNorm(4f_c_bn)(X8766,4f_c_bn_scale)/d_4f_c_bn_bias
			JCudaTensor x1488;
			x1488 = x1483[2];

			// Dealloc(X8766)
			JCudaTensor x1492;
			x1492 = x729;
			x1492.free();

			// V_4f_c_bn_bias <~~ X8579
			float x1494, x1495;
			x1494 = lrn_rate_1;
			x1495 = momentum;
			JCudaTensor x1496;
			x1496 = x1488;
			x1493.update(x1496, x1494, x1495);

			// Dealloc(X8579)
			JCudaTensor x1497;
			x1497 = x1488;
			x1497.free();

			// val X3832 = X3831 * d_Convolv(1,0)(4f_c_cv_W)/d_X8765
			JCudaTensor x1498;
			JCudaTensor x1499, x1500;
			x1499 = x1484;
			x1500 = x733;
			x1498 = x480.backward_data(x1499, x1500);

			// V_4f_c_cv_W <~~ X3831 * d_Convolv(1,0)(X8765)/d_4f_c_cv_W
			float x1502, x1503;
			x1502 = lrn_rate_1;
			x1503 = momentum;
			JCudaTensor x1504, x1505;
			x1504 = x1484;
			x1505 = x727;
			x480.backward_filter(x1504, x1505, x1501, x1502, x1503);

			// Dealloc(X3831)
			JCudaTensor x1506;
			x1506 = x1484;
			x1506.free();

			// V_4f_c_bn_scale <~~ X8580
			float x1508, x1509;
			x1508 = lrn_rate_1;
			x1509 = momentum;
			JCudaTensor x1510;
			x1510 = x1479;
			x1507.update(x1510, x1508, x1509);

			// Dealloc(X8580)
			JCudaTensor x1511;
			x1511 = x1479;
			x1511.free();

			// 4f_c_bn_bias <~~ V_4f_c_bn_bias
			float x1512, x1513;
			x1512 = 1;
			x1513 = decay_1;
			JCudaTensor x1514;
			x1514 = x1493;
			x740.update(x1514, x1512, x1513);

			// 4f_c_cv_W <~~ V_4f_c_cv_W
			float x1515, x1516;
			x1515 = 1;
			x1516 = decay_1;
			JCudaTensor x1517;
			x1517 = x1501;
			x733.update(x1517, x1515, x1516);

			// 4f_c_bn_scale <~~ V_4f_c_bn_scale
			float x1518, x1519;
			x1518 = 1;
			x1519 = decay_1;
			JCudaTensor x1520;
			x1520 = x1507;
			x739.update(x1520, x1518, x1519);

			// val X3836 = X3832 * d_ReLU()(X8765)/d_X8764
			JCudaTensor x1521;
			JCudaTensor x1522, x1523;
			x1522 = x1498;
			x1523 = x727;
			x1521 = x457.backward(x1522, x1523);

			// Dealloc(X8765)
			JCudaTensor x1524;
			x1524 = x727;
			x1524.free();

			// val X3837 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_X8763
			JCudaTensor x1525;
			JCudaTensor x1526, x1527, x1528;
			x1526 = x1521;
			x1527 = x714;
			x1528 = x724;
			JCudaTensor[] x1529 = x726.backward(x1526,x1527,x1528);
			x1525 = x1529[0];

			// val X8577 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_4f_b_bn_scale
			JCudaTensor x1530;
			x1530 = x1529[1];

			// val X8576 = X3836 * d_BatchNorm(4f_b_bn)(X8763,4f_b_bn_scale)/d_4f_b_bn_bias
			JCudaTensor x1534;
			x1534 = x1529[2];

			// Dealloc(X8763)
			JCudaTensor x1538;
			x1538 = x714;
			x1538.free();

			// V_4f_b_bn_bias <~~ X8576
			float x1540, x1541;
			x1540 = lrn_rate_1;
			x1541 = momentum;
			JCudaTensor x1542;
			x1542 = x1534;
			x1539.update(x1542, x1540, x1541);

			// Dealloc(X8576)
			JCudaTensor x1543;
			x1543 = x1534;
			x1543.free();

			// V_4f_b_bn_scale <~~ X8577
			float x1545, x1546;
			x1545 = lrn_rate_1;
			x1546 = momentum;
			JCudaTensor x1547;
			x1547 = x1530;
			x1544.update(x1547, x1545, x1546);

			// Dealloc(X8577)
			JCudaTensor x1548;
			x1548 = x1530;
			x1548.free();

			// val X3838 = X3837 * d_Convolv(1,1)(4f_b_cv_W)/d_X8762
			JCudaTensor x1549;
			JCudaTensor x1550, x1551;
			x1550 = x1525;
			x1551 = x718;
			x1549 = x464.backward_data(x1550, x1551);

			// V_4f_b_cv_W <~~ X3837 * d_Convolv(1,1)(X8762)/d_4f_b_cv_W
			float x1553, x1554;
			x1553 = lrn_rate_1;
			x1554 = momentum;
			JCudaTensor x1555, x1556;
			x1555 = x1525;
			x1556 = x712;
			x464.backward_filter(x1555, x1556, x1552, x1553, x1554);

			// Dealloc(X3837)
			JCudaTensor x1557;
			x1557 = x1525;
			x1557.free();

			// 4f_b_bn_bias <~~ V_4f_b_bn_bias
			float x1558, x1559;
			x1558 = 1;
			x1559 = decay_1;
			JCudaTensor x1560;
			x1560 = x1539;
			x725.update(x1560, x1558, x1559);

			// 4f_b_bn_scale <~~ V_4f_b_bn_scale
			float x1561, x1562;
			x1561 = 1;
			x1562 = decay_1;
			JCudaTensor x1563;
			x1563 = x1544;
			x724.update(x1563, x1561, x1562);

			// 4f_b_cv_W <~~ V_4f_b_cv_W
			float x1564, x1565;
			x1564 = 1;
			x1565 = decay_1;
			JCudaTensor x1566;
			x1566 = x1552;
			x718.update(x1566, x1564, x1565);

			// val X3842 = X3838 * d_ReLU()(X8762)/d_X8761
			JCudaTensor x1567;
			JCudaTensor x1568, x1569;
			x1568 = x1549;
			x1569 = x712;
			x1567 = x457.backward(x1568, x1569);

			// Dealloc(X8762)
			JCudaTensor x1570;
			x1570 = x712;
			x1570.free();

			// val X8573 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_4f_a_bn_bias
			JCudaTensor x1571;
			JCudaTensor x1572, x1573, x1574;
			x1572 = x1567;
			x1573 = x699;
			x1574 = x709;
			JCudaTensor[] x1575 = x711.backward(x1572,x1573,x1574);
			x1571 = x1575[2];

			// val X3843 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_X8760
			JCudaTensor x1576;
			x1576 = x1575[0];

			// val X8574 = X3842 * d_BatchNorm(4f_a_bn)(X8760,4f_a_bn_scale)/d_4f_a_bn_scale
			JCudaTensor x1580;
			x1580 = x1575[1];

			// Dealloc(X8760)
			JCudaTensor x1584;
			x1584 = x699;
			x1584.free();

			// val X3844 = X3843 * d_Convolv(1,0)(4f_a_cv_W)/d_X8759
			JCudaTensor x1585;
			JCudaTensor x1586, x1587;
			x1586 = x1576;
			x1587 = x703;
			x1585 = x504.backward_data(x1586, x1587);

			// V_4f_a_cv_W <~~ X3843 * d_Convolv(1,0)(X8759)/d_4f_a_cv_W
			float x1589, x1590;
			x1589 = lrn_rate_1;
			x1590 = momentum;
			JCudaTensor x1591, x1592;
			x1591 = x1576;
			x1592 = x697;
			x504.backward_filter(x1591, x1592, x1588, x1589, x1590);

			// Dealloc(X3843)
			JCudaTensor x1593;
			x1593 = x1576;
			x1593.free();

			// V_4f_a_bn_bias <~~ X8573
			float x1595, x1596;
			x1595 = lrn_rate_1;
			x1596 = momentum;
			JCudaTensor x1597;
			x1597 = x1571;
			x1594.update(x1597, x1595, x1596);

			// Dealloc(X8573)
			JCudaTensor x1598;
			x1598 = x1571;
			x1598.free();

			// V_4f_a_bn_scale <~~ X8574
			float x1600, x1601;
			x1600 = lrn_rate_1;
			x1601 = momentum;
			JCudaTensor x1602;
			x1602 = x1580;
			x1599.update(x1602, x1600, x1601);

			// Dealloc(X8574)
			JCudaTensor x1603;
			x1603 = x1580;
			x1603.free();

			// 4f_a_cv_W <~~ V_4f_a_cv_W
			float x1604, x1605;
			x1604 = 1;
			x1605 = decay_1;
			JCudaTensor x1606;
			x1606 = x1588;
			x703.update(x1606, x1604, x1605);

			// 4f_a_bn_bias <~~ V_4f_a_bn_bias
			float x1607, x1608;
			x1607 = 1;
			x1608 = decay_1;
			JCudaTensor x1609;
			x1609 = x1594;
			x710.update(x1609, x1607, x1608);

			// 4f_a_bn_scale <~~ V_4f_a_bn_scale
			float x1610, x1611;
			x1610 = 1;
			x1611 = decay_1;
			JCudaTensor x1612;
			x1612 = x1599;
			x709.update(x1612, x1610, x1611);

			// val X3845 = (X3844 + X3820)
			JCudaTensor x1613;
			JCudaTensor x1614, x1615;
			x1614 = x1585;
			x1615 = x1471;
			x1613 = x1614.plus_i(x1615);

			// Dealloc(X3820)
			JCudaTensor x1616;
			x1616 = x1471;
			x1616.free();

			// val X3857 = X3845 * d_ReLU()(X8759)/d_X8758
			JCudaTensor x1617;
			JCudaTensor x1618, x1619;
			x1618 = x1613;
			x1619 = x697;
			x1617 = x490.backward(x1618, x1619);

			// Dealloc(X8759)
			JCudaTensor x1620;
			x1620 = x697;
			x1620.free();

			// val X3867 = X3857.copy * d_ReLU()(X8757)/d_X8756
			JCudaTensor x1621;
			JCudaTensor x1622, x1623;
			x1622 = x1617;
			x1622 = x1622.clone();
			x1623 = x692;
			x1621 = x490.backward(x1622, x1623);

			// Dealloc(X8757)
			JCudaTensor x1624;
			x1624 = x692;
			x1624.free();

			// val X3868 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_X8755
			JCudaTensor x1625;
			JCudaTensor x1626, x1627, x1628;
			x1626 = x1621;
			x1627 = x679;
			x1628 = x689;
			JCudaTensor[] x1629 = x691.backward(x1626,x1627,x1628);
			x1625 = x1629[0];

			// val X8570 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_4e_c_bn_bias
			JCudaTensor x1630;
			x1630 = x1629[2];

			// val X8571 = X3867 * d_BatchNorm(4e_c_bn)(X8755,4e_c_bn_scale)/d_4e_c_bn_scale
			JCudaTensor x1634;
			x1634 = x1629[1];

			// Dealloc(X8755)
			JCudaTensor x1638;
			x1638 = x679;
			x1638.free();

			// val X3869 = X3868 * d_Convolv(1,0)(4e_c_cv_W)/d_X8754
			JCudaTensor x1639;
			JCudaTensor x1640, x1641;
			x1640 = x1625;
			x1641 = x683;
			x1639 = x480.backward_data(x1640, x1641);

			// V_4e_c_cv_W <~~ X3868 * d_Convolv(1,0)(X8754)/d_4e_c_cv_W
			float x1643, x1644;
			x1643 = lrn_rate_1;
			x1644 = momentum;
			JCudaTensor x1645, x1646;
			x1645 = x1625;
			x1646 = x677;
			x480.backward_filter(x1645, x1646, x1642, x1643, x1644);

			// Dealloc(X3868)
			JCudaTensor x1647;
			x1647 = x1625;
			x1647.free();

			// V_4e_c_bn_scale <~~ X8571
			float x1649, x1650;
			x1649 = lrn_rate_1;
			x1650 = momentum;
			JCudaTensor x1651;
			x1651 = x1634;
			x1648.update(x1651, x1649, x1650);

			// Dealloc(X8571)
			JCudaTensor x1652;
			x1652 = x1634;
			x1652.free();

			// V_4e_c_bn_bias <~~ X8570
			float x1654, x1655;
			x1654 = lrn_rate_1;
			x1655 = momentum;
			JCudaTensor x1656;
			x1656 = x1630;
			x1653.update(x1656, x1654, x1655);

			// Dealloc(X8570)
			JCudaTensor x1657;
			x1657 = x1630;
			x1657.free();

			// 4e_c_cv_W <~~ V_4e_c_cv_W
			float x1658, x1659;
			x1658 = 1;
			x1659 = decay_1;
			JCudaTensor x1660;
			x1660 = x1642;
			x683.update(x1660, x1658, x1659);

			// 4e_c_bn_scale <~~ V_4e_c_bn_scale
			float x1661, x1662;
			x1661 = 1;
			x1662 = decay_1;
			JCudaTensor x1663;
			x1663 = x1648;
			x689.update(x1663, x1661, x1662);

			// 4e_c_bn_bias <~~ V_4e_c_bn_bias
			float x1664, x1665;
			x1664 = 1;
			x1665 = decay_1;
			JCudaTensor x1666;
			x1666 = x1653;
			x690.update(x1666, x1664, x1665);

			// val X3873 = X3869 * d_ReLU()(X8754)/d_X8753
			JCudaTensor x1667;
			JCudaTensor x1668, x1669;
			x1668 = x1639;
			x1669 = x677;
			x1667 = x457.backward(x1668, x1669);

			// Dealloc(X8754)
			JCudaTensor x1670;
			x1670 = x677;
			x1670.free();

			// val X3874 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_X8752
			JCudaTensor x1671;
			JCudaTensor x1672, x1673, x1674;
			x1672 = x1667;
			x1673 = x664;
			x1674 = x674;
			JCudaTensor[] x1675 = x676.backward(x1672,x1673,x1674);
			x1671 = x1675[0];

			// val X8568 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_4e_b_bn_scale
			JCudaTensor x1676;
			x1676 = x1675[1];

			// val X8567 = X3873 * d_BatchNorm(4e_b_bn)(X8752,4e_b_bn_scale)/d_4e_b_bn_bias
			JCudaTensor x1680;
			x1680 = x1675[2];

			// Dealloc(X8752)
			JCudaTensor x1684;
			x1684 = x664;
			x1684.free();

			// val X3875 = X3874 * d_Convolv(1,1)(4e_b_cv_W)/d_X8751
			JCudaTensor x1685;
			JCudaTensor x1686, x1687;
			x1686 = x1671;
			x1687 = x668;
			x1685 = x464.backward_data(x1686, x1687);

			// V_4e_b_bn_bias <~~ X8567
			float x1689, x1690;
			x1689 = lrn_rate_1;
			x1690 = momentum;
			JCudaTensor x1691;
			x1691 = x1680;
			x1688.update(x1691, x1689, x1690);

			// Dealloc(X8567)
			JCudaTensor x1692;
			x1692 = x1680;
			x1692.free();

			// V_4e_b_bn_scale <~~ X8568
			float x1694, x1695;
			x1694 = lrn_rate_1;
			x1695 = momentum;
			JCudaTensor x1696;
			x1696 = x1676;
			x1693.update(x1696, x1694, x1695);

			// Dealloc(X8568)
			JCudaTensor x1697;
			x1697 = x1676;
			x1697.free();

			// V_4e_b_cv_W <~~ X3874 * d_Convolv(1,1)(X8751)/d_4e_b_cv_W
			float x1699, x1700;
			x1699 = lrn_rate_1;
			x1700 = momentum;
			JCudaTensor x1701, x1702;
			x1701 = x1671;
			x1702 = x662;
			x464.backward_filter(x1701, x1702, x1698, x1699, x1700);

			// Dealloc(X3874)
			JCudaTensor x1703;
			x1703 = x1671;
			x1703.free();

			// 4e_b_bn_bias <~~ V_4e_b_bn_bias
			float x1704, x1705;
			x1704 = 1;
			x1705 = decay_1;
			JCudaTensor x1706;
			x1706 = x1688;
			x675.update(x1706, x1704, x1705);

			// 4e_b_bn_scale <~~ V_4e_b_bn_scale
			float x1707, x1708;
			x1707 = 1;
			x1708 = decay_1;
			JCudaTensor x1709;
			x1709 = x1693;
			x674.update(x1709, x1707, x1708);

			// 4e_b_cv_W <~~ V_4e_b_cv_W
			float x1710, x1711;
			x1710 = 1;
			x1711 = decay_1;
			JCudaTensor x1712;
			x1712 = x1698;
			x668.update(x1712, x1710, x1711);

			// val X3879 = X3875 * d_ReLU()(X8751)/d_X8750
			JCudaTensor x1713;
			JCudaTensor x1714, x1715;
			x1714 = x1685;
			x1715 = x662;
			x1713 = x457.backward(x1714, x1715);

			// Dealloc(X8751)
			JCudaTensor x1716;
			x1716 = x662;
			x1716.free();

			// val X3880 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_X8749
			JCudaTensor x1717;
			JCudaTensor x1718, x1719, x1720;
			x1718 = x1713;
			x1719 = x649;
			x1720 = x659;
			JCudaTensor[] x1721 = x661.backward(x1718,x1719,x1720);
			x1717 = x1721[0];

			// val X8564 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_4e_a_bn_bias
			JCudaTensor x1722;
			x1722 = x1721[2];

			// val X8565 = X3879 * d_BatchNorm(4e_a_bn)(X8749,4e_a_bn_scale)/d_4e_a_bn_scale
			JCudaTensor x1726;
			x1726 = x1721[1];

			// Dealloc(X8749)
			JCudaTensor x1730;
			x1730 = x649;
			x1730.free();

			// val X3881 = X3880 * d_Convolv(1,0)(4e_a_cv_W)/d_X8748
			JCudaTensor x1731;
			JCudaTensor x1732, x1733;
			x1732 = x1717;
			x1733 = x653;
			x1731 = x504.backward_data(x1732, x1733);

			// V_4e_a_cv_W <~~ X3880 * d_Convolv(1,0)(X8748)/d_4e_a_cv_W
			float x1735, x1736;
			x1735 = lrn_rate_1;
			x1736 = momentum;
			JCudaTensor x1737, x1738;
			x1737 = x1717;
			x1738 = x647;
			x504.backward_filter(x1737, x1738, x1734, x1735, x1736);

			// Dealloc(X3880)
			JCudaTensor x1739;
			x1739 = x1717;
			x1739.free();

			// V_4e_a_bn_scale <~~ X8565
			float x1741, x1742;
			x1741 = lrn_rate_1;
			x1742 = momentum;
			JCudaTensor x1743;
			x1743 = x1726;
			x1740.update(x1743, x1741, x1742);

			// Dealloc(X8565)
			JCudaTensor x1744;
			x1744 = x1726;
			x1744.free();

			// V_4e_a_bn_bias <~~ X8564
			float x1746, x1747;
			x1746 = lrn_rate_1;
			x1747 = momentum;
			JCudaTensor x1748;
			x1748 = x1722;
			x1745.update(x1748, x1746, x1747);

			// Dealloc(X8564)
			JCudaTensor x1749;
			x1749 = x1722;
			x1749.free();

			// 4e_a_cv_W <~~ V_4e_a_cv_W
			float x1750, x1751;
			x1750 = 1;
			x1751 = decay_1;
			JCudaTensor x1752;
			x1752 = x1734;
			x653.update(x1752, x1750, x1751);

			// 4e_a_bn_scale <~~ V_4e_a_bn_scale
			float x1753, x1754;
			x1753 = 1;
			x1754 = decay_1;
			JCudaTensor x1755;
			x1755 = x1740;
			x659.update(x1755, x1753, x1754);

			// 4e_a_bn_bias <~~ V_4e_a_bn_bias
			float x1756, x1757;
			x1756 = 1;
			x1757 = decay_1;
			JCudaTensor x1758;
			x1758 = x1745;
			x660.update(x1758, x1756, x1757);

			// val X3882 = (X3881 + X3857)
			JCudaTensor x1759;
			JCudaTensor x1760, x1761;
			x1760 = x1731;
			x1761 = x1617;
			x1759 = x1760.plus_i(x1761);

			// Dealloc(X3857)
			JCudaTensor x1762;
			x1762 = x1617;
			x1762.free();

			// val X3894 = X3882 * d_ReLU()(X8748)/d_X8747
			JCudaTensor x1763;
			JCudaTensor x1764, x1765;
			x1764 = x1759;
			x1765 = x647;
			x1763 = x490.backward(x1764, x1765);

			// Dealloc(X8748)
			JCudaTensor x1766;
			x1766 = x647;
			x1766.free();

			// val X3904 = X3894.copy * d_ReLU()(X8746)/d_X8745
			JCudaTensor x1767;
			JCudaTensor x1768, x1769;
			x1768 = x1763;
			x1768 = x1768.clone();
			x1769 = x642;
			x1767 = x490.backward(x1768, x1769);

			// Dealloc(X8746)
			JCudaTensor x1770;
			x1770 = x642;
			x1770.free();

			// val X3905 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_X8744
			JCudaTensor x1771;
			JCudaTensor x1772, x1773, x1774;
			x1772 = x1767;
			x1773 = x629;
			x1774 = x639;
			JCudaTensor[] x1775 = x641.backward(x1772,x1773,x1774);
			x1771 = x1775[0];

			// val X8562 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_4d_c_bn_scale
			JCudaTensor x1776;
			x1776 = x1775[1];

			// val X8561 = X3904 * d_BatchNorm(4d_c_bn)(X8744,4d_c_bn_scale)/d_4d_c_bn_bias
			JCudaTensor x1780;
			x1780 = x1775[2];

			// Dealloc(X8744)
			JCudaTensor x1784;
			x1784 = x629;
			x1784.free();

			// V_4d_c_bn_scale <~~ X8562
			float x1786, x1787;
			x1786 = lrn_rate_1;
			x1787 = momentum;
			JCudaTensor x1788;
			x1788 = x1776;
			x1785.update(x1788, x1786, x1787);

			// Dealloc(X8562)
			JCudaTensor x1789;
			x1789 = x1776;
			x1789.free();

			// V_4d_c_bn_bias <~~ X8561
			float x1791, x1792;
			x1791 = lrn_rate_1;
			x1792 = momentum;
			JCudaTensor x1793;
			x1793 = x1780;
			x1790.update(x1793, x1791, x1792);

			// Dealloc(X8561)
			JCudaTensor x1794;
			x1794 = x1780;
			x1794.free();

			// val X3906 = X3905 * d_Convolv(1,0)(4d_c_cv_W)/d_X8743
			JCudaTensor x1795;
			JCudaTensor x1796, x1797;
			x1796 = x1771;
			x1797 = x633;
			x1795 = x480.backward_data(x1796, x1797);

			// V_4d_c_cv_W <~~ X3905 * d_Convolv(1,0)(X8743)/d_4d_c_cv_W
			float x1799, x1800;
			x1799 = lrn_rate_1;
			x1800 = momentum;
			JCudaTensor x1801, x1802;
			x1801 = x1771;
			x1802 = x627;
			x480.backward_filter(x1801, x1802, x1798, x1799, x1800);

			// Dealloc(X3905)
			JCudaTensor x1803;
			x1803 = x1771;
			x1803.free();

			// 4d_c_bn_scale <~~ V_4d_c_bn_scale
			float x1804, x1805;
			x1804 = 1;
			x1805 = decay_1;
			JCudaTensor x1806;
			x1806 = x1785;
			x639.update(x1806, x1804, x1805);

			// 4d_c_bn_bias <~~ V_4d_c_bn_bias
			float x1807, x1808;
			x1807 = 1;
			x1808 = decay_1;
			JCudaTensor x1809;
			x1809 = x1790;
			x640.update(x1809, x1807, x1808);

			// 4d_c_cv_W <~~ V_4d_c_cv_W
			float x1810, x1811;
			x1810 = 1;
			x1811 = decay_1;
			JCudaTensor x1812;
			x1812 = x1798;
			x633.update(x1812, x1810, x1811);

			// val X3910 = X3906 * d_ReLU()(X8743)/d_X8742
			JCudaTensor x1813;
			JCudaTensor x1814, x1815;
			x1814 = x1795;
			x1815 = x627;
			x1813 = x457.backward(x1814, x1815);

			// Dealloc(X8743)
			JCudaTensor x1816;
			x1816 = x627;
			x1816.free();

			// val X8558 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_4d_b_bn_bias
			JCudaTensor x1817;
			JCudaTensor x1818, x1819, x1820;
			x1818 = x1813;
			x1819 = x614;
			x1820 = x624;
			JCudaTensor[] x1821 = x626.backward(x1818,x1819,x1820);
			x1817 = x1821[2];

			// val X3911 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_X8741
			JCudaTensor x1822;
			x1822 = x1821[0];

			// val X8559 = X3910 * d_BatchNorm(4d_b_bn)(X8741,4d_b_bn_scale)/d_4d_b_bn_scale
			JCudaTensor x1826;
			x1826 = x1821[1];

			// Dealloc(X8741)
			JCudaTensor x1830;
			x1830 = x614;
			x1830.free();

			// val X3912 = X3911 * d_Convolv(1,1)(4d_b_cv_W)/d_X8740
			JCudaTensor x1831;
			JCudaTensor x1832, x1833;
			x1832 = x1822;
			x1833 = x618;
			x1831 = x464.backward_data(x1832, x1833);

			// V_4d_b_cv_W <~~ X3911 * d_Convolv(1,1)(X8740)/d_4d_b_cv_W
			float x1835, x1836;
			x1835 = lrn_rate_1;
			x1836 = momentum;
			JCudaTensor x1837, x1838;
			x1837 = x1822;
			x1838 = x612;
			x464.backward_filter(x1837, x1838, x1834, x1835, x1836);

			// Dealloc(X3911)
			JCudaTensor x1839;
			x1839 = x1822;
			x1839.free();

			// V_4d_b_bn_scale <~~ X8559
			float x1841, x1842;
			x1841 = lrn_rate_1;
			x1842 = momentum;
			JCudaTensor x1843;
			x1843 = x1826;
			x1840.update(x1843, x1841, x1842);

			// Dealloc(X8559)
			JCudaTensor x1844;
			x1844 = x1826;
			x1844.free();

			// V_4d_b_bn_bias <~~ X8558
			float x1846, x1847;
			x1846 = lrn_rate_1;
			x1847 = momentum;
			JCudaTensor x1848;
			x1848 = x1817;
			x1845.update(x1848, x1846, x1847);

			// Dealloc(X8558)
			JCudaTensor x1849;
			x1849 = x1817;
			x1849.free();

			// 4d_b_cv_W <~~ V_4d_b_cv_W
			float x1850, x1851;
			x1850 = 1;
			x1851 = decay_1;
			JCudaTensor x1852;
			x1852 = x1834;
			x618.update(x1852, x1850, x1851);

			// 4d_b_bn_scale <~~ V_4d_b_bn_scale
			float x1853, x1854;
			x1853 = 1;
			x1854 = decay_1;
			JCudaTensor x1855;
			x1855 = x1840;
			x624.update(x1855, x1853, x1854);

			// 4d_b_bn_bias <~~ V_4d_b_bn_bias
			float x1856, x1857;
			x1856 = 1;
			x1857 = decay_1;
			JCudaTensor x1858;
			x1858 = x1845;
			x625.update(x1858, x1856, x1857);

			// val X3916 = X3912 * d_ReLU()(X8740)/d_X8739
			JCudaTensor x1859;
			JCudaTensor x1860, x1861;
			x1860 = x1831;
			x1861 = x612;
			x1859 = x457.backward(x1860, x1861);

			// Dealloc(X8740)
			JCudaTensor x1862;
			x1862 = x612;
			x1862.free();

			// val X3917 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_X8738
			JCudaTensor x1863;
			JCudaTensor x1864, x1865, x1866;
			x1864 = x1859;
			x1865 = x599;
			x1866 = x609;
			JCudaTensor[] x1867 = x611.backward(x1864,x1865,x1866);
			x1863 = x1867[0];

			// val X8556 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_4d_a_bn_scale
			JCudaTensor x1868;
			x1868 = x1867[1];

			// val X8555 = X3916 * d_BatchNorm(4d_a_bn)(X8738,4d_a_bn_scale)/d_4d_a_bn_bias
			JCudaTensor x1872;
			x1872 = x1867[2];

			// Dealloc(X8738)
			JCudaTensor x1876;
			x1876 = x599;
			x1876.free();

			// V_4d_a_bn_bias <~~ X8555
			float x1878, x1879;
			x1878 = lrn_rate_1;
			x1879 = momentum;
			JCudaTensor x1880;
			x1880 = x1872;
			x1877.update(x1880, x1878, x1879);

			// Dealloc(X8555)
			JCudaTensor x1881;
			x1881 = x1872;
			x1881.free();

			// val X3918 = X3917 * d_Convolv(1,0)(4d_a_cv_W)/d_X8737
			JCudaTensor x1882;
			JCudaTensor x1883, x1884;
			x1883 = x1863;
			x1884 = x603;
			x1882 = x504.backward_data(x1883, x1884);

			// V_4d_a_cv_W <~~ X3917 * d_Convolv(1,0)(X8737)/d_4d_a_cv_W
			float x1886, x1887;
			x1886 = lrn_rate_1;
			x1887 = momentum;
			JCudaTensor x1888, x1889;
			x1888 = x1863;
			x1889 = x597;
			x504.backward_filter(x1888, x1889, x1885, x1886, x1887);

			// Dealloc(X3917)
			JCudaTensor x1890;
			x1890 = x1863;
			x1890.free();

			// V_4d_a_bn_scale <~~ X8556
			float x1892, x1893;
			x1892 = lrn_rate_1;
			x1893 = momentum;
			JCudaTensor x1894;
			x1894 = x1868;
			x1891.update(x1894, x1892, x1893);

			// Dealloc(X8556)
			JCudaTensor x1895;
			x1895 = x1868;
			x1895.free();

			// 4d_a_bn_bias <~~ V_4d_a_bn_bias
			float x1896, x1897;
			x1896 = 1;
			x1897 = decay_1;
			JCudaTensor x1898;
			x1898 = x1877;
			x610.update(x1898, x1896, x1897);

			// 4d_a_cv_W <~~ V_4d_a_cv_W
			float x1899, x1900;
			x1899 = 1;
			x1900 = decay_1;
			JCudaTensor x1901;
			x1901 = x1885;
			x603.update(x1901, x1899, x1900);

			// 4d_a_bn_scale <~~ V_4d_a_bn_scale
			float x1902, x1903;
			x1902 = 1;
			x1903 = decay_1;
			JCudaTensor x1904;
			x1904 = x1891;
			x609.update(x1904, x1902, x1903);

			// val X3919 = (X3918 + X3894)
			JCudaTensor x1905;
			JCudaTensor x1906, x1907;
			x1906 = x1882;
			x1907 = x1763;
			x1905 = x1906.plus_i(x1907);

			// Dealloc(X3894)
			JCudaTensor x1908;
			x1908 = x1763;
			x1908.free();

			// val X3931 = X3919 * d_ReLU()(X8737)/d_X8736
			JCudaTensor x1909;
			JCudaTensor x1910, x1911;
			x1910 = x1905;
			x1911 = x597;
			x1909 = x490.backward(x1910, x1911);

			// Dealloc(X8737)
			JCudaTensor x1912;
			x1912 = x597;
			x1912.free();

			// val X3941 = X3931.copy * d_ReLU()(X8735)/d_X8734
			JCudaTensor x1913;
			JCudaTensor x1914, x1915;
			x1914 = x1909;
			x1914 = x1914.clone();
			x1915 = x592;
			x1913 = x490.backward(x1914, x1915);

			// Dealloc(X8735)
			JCudaTensor x1916;
			x1916 = x592;
			x1916.free();

			// val X3942 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_X8733
			JCudaTensor x1917;
			JCudaTensor x1918, x1919, x1920;
			x1918 = x1913;
			x1919 = x579;
			x1920 = x589;
			JCudaTensor[] x1921 = x591.backward(x1918,x1919,x1920);
			x1917 = x1921[0];

			// val X8553 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_4c_c_bn_scale
			JCudaTensor x1922;
			x1922 = x1921[1];

			// val X8552 = X3941 * d_BatchNorm(4c_c_bn)(X8733,4c_c_bn_scale)/d_4c_c_bn_bias
			JCudaTensor x1926;
			x1926 = x1921[2];

			// Dealloc(X8733)
			JCudaTensor x1930;
			x1930 = x579;
			x1930.free();

			// V_4c_c_bn_bias <~~ X8552
			float x1932, x1933;
			x1932 = lrn_rate_1;
			x1933 = momentum;
			JCudaTensor x1934;
			x1934 = x1926;
			x1931.update(x1934, x1932, x1933);

			// Dealloc(X8552)
			JCudaTensor x1935;
			x1935 = x1926;
			x1935.free();

			// val X3943 = X3942 * d_Convolv(1,0)(4c_c_cv_W)/d_X8732
			JCudaTensor x1936;
			JCudaTensor x1937, x1938;
			x1937 = x1917;
			x1938 = x583;
			x1936 = x480.backward_data(x1937, x1938);

			// V_4c_c_cv_W <~~ X3942 * d_Convolv(1,0)(X8732)/d_4c_c_cv_W
			float x1940, x1941;
			x1940 = lrn_rate_1;
			x1941 = momentum;
			JCudaTensor x1942, x1943;
			x1942 = x1917;
			x1943 = x577;
			x480.backward_filter(x1942, x1943, x1939, x1940, x1941);

			// Dealloc(X3942)
			JCudaTensor x1944;
			x1944 = x1917;
			x1944.free();

			// V_4c_c_bn_scale <~~ X8553
			float x1946, x1947;
			x1946 = lrn_rate_1;
			x1947 = momentum;
			JCudaTensor x1948;
			x1948 = x1922;
			x1945.update(x1948, x1946, x1947);

			// Dealloc(X8553)
			JCudaTensor x1949;
			x1949 = x1922;
			x1949.free();

			// 4c_c_bn_bias <~~ V_4c_c_bn_bias
			float x1950, x1951;
			x1950 = 1;
			x1951 = decay_1;
			JCudaTensor x1952;
			x1952 = x1931;
			x590.update(x1952, x1950, x1951);

			// 4c_c_cv_W <~~ V_4c_c_cv_W
			float x1953, x1954;
			x1953 = 1;
			x1954 = decay_1;
			JCudaTensor x1955;
			x1955 = x1939;
			x583.update(x1955, x1953, x1954);

			// 4c_c_bn_scale <~~ V_4c_c_bn_scale
			float x1956, x1957;
			x1956 = 1;
			x1957 = decay_1;
			JCudaTensor x1958;
			x1958 = x1945;
			x589.update(x1958, x1956, x1957);

			// val X3947 = X3943 * d_ReLU()(X8732)/d_X8731
			JCudaTensor x1959;
			JCudaTensor x1960, x1961;
			x1960 = x1936;
			x1961 = x577;
			x1959 = x457.backward(x1960, x1961);

			// Dealloc(X8732)
			JCudaTensor x1962;
			x1962 = x577;
			x1962.free();

			// val X8550 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_4c_b_bn_scale
			JCudaTensor x1963;
			JCudaTensor x1964, x1965, x1966;
			x1964 = x1959;
			x1965 = x564;
			x1966 = x574;
			JCudaTensor[] x1967 = x576.backward(x1964,x1965,x1966);
			x1963 = x1967[1];

			// val X3948 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_X8730
			JCudaTensor x1968;
			x1968 = x1967[0];

			// val X8549 = X3947 * d_BatchNorm(4c_b_bn)(X8730,4c_b_bn_scale)/d_4c_b_bn_bias
			JCudaTensor x1972;
			x1972 = x1967[2];

			// Dealloc(X8730)
			JCudaTensor x1976;
			x1976 = x564;
			x1976.free();

			// val X3949 = X3948 * d_Convolv(1,1)(4c_b_cv_W)/d_X8729
			JCudaTensor x1977;
			JCudaTensor x1978, x1979;
			x1978 = x1968;
			x1979 = x568;
			x1977 = x464.backward_data(x1978, x1979);

			// V_4c_b_cv_W <~~ X3948 * d_Convolv(1,1)(X8729)/d_4c_b_cv_W
			float x1981, x1982;
			x1981 = lrn_rate_1;
			x1982 = momentum;
			JCudaTensor x1983, x1984;
			x1983 = x1968;
			x1984 = x562;
			x464.backward_filter(x1983, x1984, x1980, x1981, x1982);

			// Dealloc(X3948)
			JCudaTensor x1985;
			x1985 = x1968;
			x1985.free();

			// V_4c_b_bn_bias <~~ X8549
			float x1987, x1988;
			x1987 = lrn_rate_1;
			x1988 = momentum;
			JCudaTensor x1989;
			x1989 = x1972;
			x1986.update(x1989, x1987, x1988);

			// Dealloc(X8549)
			JCudaTensor x1990;
			x1990 = x1972;
			x1990.free();

			// V_4c_b_bn_scale <~~ X8550
			float x1992, x1993;
			x1992 = lrn_rate_1;
			x1993 = momentum;
			JCudaTensor x1994;
			x1994 = x1963;
			x1991.update(x1994, x1992, x1993);

			// Dealloc(X8550)
			JCudaTensor x1995;
			x1995 = x1963;
			x1995.free();

			// 4c_b_cv_W <~~ V_4c_b_cv_W
			float x1996, x1997;
			x1996 = 1;
			x1997 = decay_1;
			JCudaTensor x1998;
			x1998 = x1980;
			x568.update(x1998, x1996, x1997);

			// 4c_b_bn_bias <~~ V_4c_b_bn_bias
			float x1999, x2000;
			x1999 = 1;
			x2000 = decay_1;
			JCudaTensor x2001;
			x2001 = x1986;
			x575.update(x2001, x1999, x2000);

			// 4c_b_bn_scale <~~ V_4c_b_bn_scale
			float x2002, x2003;
			x2002 = 1;
			x2003 = decay_1;
			JCudaTensor x2004;
			x2004 = x1991;
			x574.update(x2004, x2002, x2003);

			// val X3953 = X3949 * d_ReLU()(X8729)/d_X8728
			JCudaTensor x2005;
			JCudaTensor x2006, x2007;
			x2006 = x1977;
			x2007 = x562;
			x2005 = x457.backward(x2006, x2007);

			// Dealloc(X8729)
			JCudaTensor x2008;
			x2008 = x562;
			x2008.free();

			// val X3954 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_X8727
			JCudaTensor x2009;
			JCudaTensor x2010, x2011, x2012;
			x2010 = x2005;
			x2011 = x549;
			x2012 = x559;
			JCudaTensor[] x2013 = x561.backward(x2010,x2011,x2012);
			x2009 = x2013[0];

			// val X8547 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_4c_a_bn_scale
			JCudaTensor x2014;
			x2014 = x2013[1];

			// val X8546 = X3953 * d_BatchNorm(4c_a_bn)(X8727,4c_a_bn_scale)/d_4c_a_bn_bias
			JCudaTensor x2018;
			x2018 = x2013[2];

			// Dealloc(X8727)
			JCudaTensor x2022;
			x2022 = x549;
			x2022.free();

			// val X3955 = X3954 * d_Convolv(1,0)(4c_a_cv_W)/d_X8726
			JCudaTensor x2023;
			JCudaTensor x2024, x2025;
			x2024 = x2009;
			x2025 = x553;
			x2023 = x504.backward_data(x2024, x2025);

			// V_4c_a_bn_bias <~~ X8546
			float x2027, x2028;
			x2027 = lrn_rate_1;
			x2028 = momentum;
			JCudaTensor x2029;
			x2029 = x2018;
			x2026.update(x2029, x2027, x2028);

			// Dealloc(X8546)
			JCudaTensor x2030;
			x2030 = x2018;
			x2030.free();

			// V_4c_a_bn_scale <~~ X8547
			float x2032, x2033;
			x2032 = lrn_rate_1;
			x2033 = momentum;
			JCudaTensor x2034;
			x2034 = x2014;
			x2031.update(x2034, x2032, x2033);

			// Dealloc(X8547)
			JCudaTensor x2035;
			x2035 = x2014;
			x2035.free();

			// V_4c_a_cv_W <~~ X3954 * d_Convolv(1,0)(X8726)/d_4c_a_cv_W
			float x2037, x2038;
			x2037 = lrn_rate_1;
			x2038 = momentum;
			JCudaTensor x2039, x2040;
			x2039 = x2009;
			x2040 = x547;
			x504.backward_filter(x2039, x2040, x2036, x2037, x2038);

			// Dealloc(X3954)
			JCudaTensor x2041;
			x2041 = x2009;
			x2041.free();

			// 4c_a_bn_bias <~~ V_4c_a_bn_bias
			float x2042, x2043;
			x2042 = 1;
			x2043 = decay_1;
			JCudaTensor x2044;
			x2044 = x2026;
			x560.update(x2044, x2042, x2043);

			// 4c_a_bn_scale <~~ V_4c_a_bn_scale
			float x2045, x2046;
			x2045 = 1;
			x2046 = decay_1;
			JCudaTensor x2047;
			x2047 = x2031;
			x559.update(x2047, x2045, x2046);

			// 4c_a_cv_W <~~ V_4c_a_cv_W
			float x2048, x2049;
			x2048 = 1;
			x2049 = decay_1;
			JCudaTensor x2050;
			x2050 = x2036;
			x553.update(x2050, x2048, x2049);

			// val X3956 = (X3955 + X3931)
			JCudaTensor x2051;
			JCudaTensor x2052, x2053;
			x2052 = x2023;
			x2053 = x1909;
			x2051 = x2052.plus_i(x2053);

			// Dealloc(X3931)
			JCudaTensor x2054;
			x2054 = x1909;
			x2054.free();

			// val X3968 = X3956 * d_ReLU()(X8726)/d_X8725
			JCudaTensor x2055;
			JCudaTensor x2056, x2057;
			x2056 = x2051;
			x2057 = x547;
			x2055 = x490.backward(x2056, x2057);

			// Dealloc(X8726)
			JCudaTensor x2058;
			x2058 = x547;
			x2058.free();

			// val X3978 = X3968.copy * d_ReLU()(X8724)/d_X8723
			JCudaTensor x2059;
			JCudaTensor x2060, x2061;
			x2060 = x2055;
			x2060 = x2060.clone();
			x2061 = x542;
			x2059 = x490.backward(x2060, x2061);

			// Dealloc(X8724)
			JCudaTensor x2062;
			x2062 = x542;
			x2062.free();

			// val X8543 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_4b_c_bn_bias
			JCudaTensor x2063;
			JCudaTensor x2064, x2065, x2066;
			x2064 = x2059;
			x2065 = x529;
			x2066 = x539;
			JCudaTensor[] x2067 = x541.backward(x2064,x2065,x2066);
			x2063 = x2067[2];

			// val X3979 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_X8722
			JCudaTensor x2068;
			x2068 = x2067[0];

			// val X8544 = X3978 * d_BatchNorm(4b_c_bn)(X8722,4b_c_bn_scale)/d_4b_c_bn_scale
			JCudaTensor x2072;
			x2072 = x2067[1];

			// Dealloc(X8722)
			JCudaTensor x2076;
			x2076 = x529;
			x2076.free();

			// val X3980 = X3979 * d_Convolv(1,0)(4b_c_cv_W)/d_X8721
			JCudaTensor x2077;
			JCudaTensor x2078, x2079;
			x2078 = x2068;
			x2079 = x533;
			x2077 = x480.backward_data(x2078, x2079);

			// V_4b_c_bn_bias <~~ X8543
			float x2081, x2082;
			x2081 = lrn_rate_1;
			x2082 = momentum;
			JCudaTensor x2083;
			x2083 = x2063;
			x2080.update(x2083, x2081, x2082);

			// Dealloc(X8543)
			JCudaTensor x2084;
			x2084 = x2063;
			x2084.free();

			// V_4b_c_cv_W <~~ X3979 * d_Convolv(1,0)(X8721)/d_4b_c_cv_W
			float x2086, x2087;
			x2086 = lrn_rate_1;
			x2087 = momentum;
			JCudaTensor x2088, x2089;
			x2088 = x2068;
			x2089 = x527;
			x480.backward_filter(x2088, x2089, x2085, x2086, x2087);

			// Dealloc(X3979)
			JCudaTensor x2090;
			x2090 = x2068;
			x2090.free();

			// V_4b_c_bn_scale <~~ X8544
			float x2092, x2093;
			x2092 = lrn_rate_1;
			x2093 = momentum;
			JCudaTensor x2094;
			x2094 = x2072;
			x2091.update(x2094, x2092, x2093);

			// Dealloc(X8544)
			JCudaTensor x2095;
			x2095 = x2072;
			x2095.free();

			// 4b_c_bn_bias <~~ V_4b_c_bn_bias
			float x2096, x2097;
			x2096 = 1;
			x2097 = decay_1;
			JCudaTensor x2098;
			x2098 = x2080;
			x540.update(x2098, x2096, x2097);

			// 4b_c_cv_W <~~ V_4b_c_cv_W
			float x2099, x2100;
			x2099 = 1;
			x2100 = decay_1;
			JCudaTensor x2101;
			x2101 = x2085;
			x533.update(x2101, x2099, x2100);

			// 4b_c_bn_scale <~~ V_4b_c_bn_scale
			float x2102, x2103;
			x2102 = 1;
			x2103 = decay_1;
			JCudaTensor x2104;
			x2104 = x2091;
			x539.update(x2104, x2102, x2103);

			// val X3984 = X3980 * d_ReLU()(X8721)/d_X8720
			JCudaTensor x2105;
			JCudaTensor x2106, x2107;
			x2106 = x2077;
			x2107 = x527;
			x2105 = x457.backward(x2106, x2107);

			// Dealloc(X8721)
			JCudaTensor x2108;
			x2108 = x527;
			x2108.free();

			// val X3985 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_X8719
			JCudaTensor x2109;
			JCudaTensor x2110, x2111, x2112;
			x2110 = x2105;
			x2111 = x514;
			x2112 = x524;
			JCudaTensor[] x2113 = x526.backward(x2110,x2111,x2112);
			x2109 = x2113[0];

			// val X8541 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_4b_b_bn_scale
			JCudaTensor x2114;
			x2114 = x2113[1];

			// val X8540 = X3984 * d_BatchNorm(4b_b_bn)(X8719,4b_b_bn_scale)/d_4b_b_bn_bias
			JCudaTensor x2118;
			x2118 = x2113[2];

			// Dealloc(X8719)
			JCudaTensor x2122;
			x2122 = x514;
			x2122.free();

			// val X3986 = X3985 * d_Convolv(1,1)(4b_b_cv_W)/d_X8718
			JCudaTensor x2123;
			JCudaTensor x2124, x2125;
			x2124 = x2109;
			x2125 = x518;
			x2123 = x464.backward_data(x2124, x2125);

			// V_4b_b_cv_W <~~ X3985 * d_Convolv(1,1)(X8718)/d_4b_b_cv_W
			float x2127, x2128;
			x2127 = lrn_rate_1;
			x2128 = momentum;
			JCudaTensor x2129, x2130;
			x2129 = x2109;
			x2130 = x512;
			x464.backward_filter(x2129, x2130, x2126, x2127, x2128);

			// Dealloc(X3985)
			JCudaTensor x2131;
			x2131 = x2109;
			x2131.free();

			// V_4b_b_bn_scale <~~ X8541
			float x2133, x2134;
			x2133 = lrn_rate_1;
			x2134 = momentum;
			JCudaTensor x2135;
			x2135 = x2114;
			x2132.update(x2135, x2133, x2134);

			// Dealloc(X8541)
			JCudaTensor x2136;
			x2136 = x2114;
			x2136.free();

			// V_4b_b_bn_bias <~~ X8540
			float x2138, x2139;
			x2138 = lrn_rate_1;
			x2139 = momentum;
			JCudaTensor x2140;
			x2140 = x2118;
			x2137.update(x2140, x2138, x2139);

			// Dealloc(X8540)
			JCudaTensor x2141;
			x2141 = x2118;
			x2141.free();

			// 4b_b_cv_W <~~ V_4b_b_cv_W
			float x2142, x2143;
			x2142 = 1;
			x2143 = decay_1;
			JCudaTensor x2144;
			x2144 = x2126;
			x518.update(x2144, x2142, x2143);

			// 4b_b_bn_scale <~~ V_4b_b_bn_scale
			float x2145, x2146;
			x2145 = 1;
			x2146 = decay_1;
			JCudaTensor x2147;
			x2147 = x2132;
			x524.update(x2147, x2145, x2146);

			// 4b_b_bn_bias <~~ V_4b_b_bn_bias
			float x2148, x2149;
			x2148 = 1;
			x2149 = decay_1;
			JCudaTensor x2150;
			x2150 = x2137;
			x525.update(x2150, x2148, x2149);

			// val X3990 = X3986 * d_ReLU()(X8718)/d_X8717
			JCudaTensor x2151;
			JCudaTensor x2152, x2153;
			x2152 = x2123;
			x2153 = x512;
			x2151 = x457.backward(x2152, x2153);

			// Dealloc(X8718)
			JCudaTensor x2154;
			x2154 = x512;
			x2154.free();

			// val X8538 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_4b_a_bn_scale
			JCudaTensor x2155;
			JCudaTensor x2156, x2157, x2158;
			x2156 = x2151;
			x2157 = x498;
			x2158 = x509;
			JCudaTensor[] x2159 = x511.backward(x2156,x2157,x2158);
			x2155 = x2159[1];

			// val X3991 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_X8716
			JCudaTensor x2160;
			x2160 = x2159[0];

			// val X8537 = X3990 * d_BatchNorm(4b_a_bn)(X8716,4b_a_bn_scale)/d_4b_a_bn_bias
			JCudaTensor x2164;
			x2164 = x2159[2];

			// Dealloc(X8716)
			JCudaTensor x2168;
			x2168 = x498;
			x2168.free();

			// val X3992 = X3991 * d_Convolv(1,0)(4b_a_cv_W)/d_X8715
			JCudaTensor x2169;
			JCudaTensor x2170, x2171;
			x2170 = x2160;
			x2171 = x502;
			x2169 = x504.backward_data(x2170, x2171);

			// V_4b_a_cv_W <~~ X3991 * d_Convolv(1,0)(X8715)/d_4b_a_cv_W
			float x2173, x2174;
			x2173 = lrn_rate_1;
			x2174 = momentum;
			JCudaTensor x2175, x2176;
			x2175 = x2160;
			x2176 = x496;
			x504.backward_filter(x2175, x2176, x2172, x2173, x2174);

			// Dealloc(X3991)
			JCudaTensor x2177;
			x2177 = x2160;
			x2177.free();

			// V_4b_a_bn_bias <~~ X8537
			float x2179, x2180;
			x2179 = lrn_rate_1;
			x2180 = momentum;
			JCudaTensor x2181;
			x2181 = x2164;
			x2178.update(x2181, x2179, x2180);

			// Dealloc(X8537)
			JCudaTensor x2182;
			x2182 = x2164;
			x2182.free();

			// V_4b_a_bn_scale <~~ X8538
			float x2184, x2185;
			x2184 = lrn_rate_1;
			x2185 = momentum;
			JCudaTensor x2186;
			x2186 = x2155;
			x2183.update(x2186, x2184, x2185);

			// Dealloc(X8538)
			JCudaTensor x2187;
			x2187 = x2155;
			x2187.free();

			// 4b_a_cv_W <~~ V_4b_a_cv_W
			float x2188, x2189;
			x2188 = 1;
			x2189 = decay_1;
			JCudaTensor x2190;
			x2190 = x2172;
			x502.update(x2190, x2188, x2189);

			// 4b_a_bn_bias <~~ V_4b_a_bn_bias
			float x2191, x2192;
			x2191 = 1;
			x2192 = decay_1;
			JCudaTensor x2193;
			x2193 = x2178;
			x510.update(x2193, x2191, x2192);

			// 4b_a_bn_scale <~~ V_4b_a_bn_scale
			float x2194, x2195;
			x2194 = 1;
			x2195 = decay_1;
			JCudaTensor x2196;
			x2196 = x2183;
			x509.update(x2196, x2194, x2195);

			// val X3993 = (X3992 + X3968)
			JCudaTensor x2197;
			JCudaTensor x2198, x2199;
			x2198 = x2169;
			x2199 = x2055;
			x2197 = x2198.plus_i(x2199);

			// Dealloc(X3968)
			JCudaTensor x2200;
			x2200 = x2055;
			x2200.free();

			// val X4008 = X3993 * d_ReLU()(X8715)/d_X8714
			JCudaTensor x2201;
			JCudaTensor x2202, x2203;
			x2202 = x2197;
			x2203 = x496;
			x2201 = x490.backward(x2202, x2203);

			// Dealloc(X8715)
			JCudaTensor x2204;
			x2204 = x496;
			x2204.free();

			// val X4012 = X4008.copy * d_ReLU()(X8704)/d_X8703
			JCudaTensor x2205;
			JCudaTensor x2206, x2207;
			x2206 = x2201;
			x2206 = x2206.clone();
			x2207 = x488;
			x2205 = x490.backward(x2206, x2207);

			// Dealloc(X8704)
			JCudaTensor x2208;
			x2208 = x488;
			x2208.free();

			// val X4024 = X4008.copy * d_ReLU()(X8713)/d_X8712
			JCudaTensor x2209;
			JCudaTensor x2210, x2211;
			x2210 = x2201;
			x2210 = x2210.clone();
			x2211 = x491;
			x2209 = x490.backward(x2210, x2211);

			// Dealloc(X4008)
			JCudaTensor x2212;
			x2212 = x2201;
			x2212.free();

			// Dealloc(X8713)
			JCudaTensor x2213;
			x2213 = x491;
			x2213.free();

			// val X8526 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_4a1_bn_scale
			JCudaTensor x2214;
			JCudaTensor x2215, x2216, x2217;
			x2215 = x2205;
			x2216 = x427;
			x2217 = x445;
			JCudaTensor[] x2218 = x447.backward(x2215,x2216,x2217);
			x2214 = x2218[1];

			// val X8525 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_4a1_bn_bias
			JCudaTensor x2219;
			x2219 = x2218[2];

			// val X4013 = X4012 * d_BatchNorm(4a1_bn)(X8702,4a1_bn_scale)/d_X8702
			JCudaTensor x2223;
			x2223 = x2218[0];

			// Dealloc(X8702)
			JCudaTensor x2227;
			x2227 = x427;
			x2227.free();

			// val X8534 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_4a2_c_bn_bias
			JCudaTensor x2228;
			JCudaTensor x2229, x2230, x2231;
			x2229 = x2209;
			x2230 = x474;
			x2231 = x485;
			JCudaTensor[] x2232 = x487.backward(x2229,x2230,x2231);
			x2228 = x2232[2];

			// val X4025 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_X8711
			JCudaTensor x2233;
			x2233 = x2232[0];

			// val X8535 = X4024 * d_BatchNorm(4a2_c_bn)(X8711,4a2_c_bn_scale)/d_4a2_c_bn_scale
			JCudaTensor x2237;
			x2237 = x2232[1];

			// Dealloc(X8711)
			JCudaTensor x2241;
			x2241 = x474;
			x2241.free();

			// V_4a2_c_bn_scale <~~ X8535
			float x2243, x2244;
			x2243 = lrn_rate_1;
			x2244 = momentum;
			JCudaTensor x2245;
			x2245 = x2237;
			x2242.update(x2245, x2243, x2244);

			// Dealloc(X8535)
			JCudaTensor x2246;
			x2246 = x2237;
			x2246.free();

			// V_4a1_cv_W <~~ X4013 * d_Convolv(2,0)(X8701)/d_4a1_cv_W
			float x2248, x2249;
			x2248 = lrn_rate_1;
			x2249 = momentum;
			JCudaTensor x2250, x2251;
			x2250 = x2223;
			x2251 = x425;
			x433.backward_filter(x2250, x2251, x2247, x2248, x2249);

			// V_4a2_c_bn_bias <~~ X8534
			float x2253, x2254;
			x2253 = lrn_rate_1;
			x2254 = momentum;
			JCudaTensor x2255;
			x2255 = x2228;
			x2252.update(x2255, x2253, x2254);

			// Dealloc(X8534)
			JCudaTensor x2256;
			x2256 = x2228;
			x2256.free();

			// V_4a2_c_cv_W <~~ X4025 * d_Convolv(1,0)(X8710)/d_4a2_c_cv_W
			float x2258, x2259;
			x2258 = lrn_rate_1;
			x2259 = momentum;
			JCudaTensor x2260, x2261;
			x2260 = x2233;
			x2261 = x472;
			x480.backward_filter(x2260, x2261, x2257, x2258, x2259);

			// val X4014 = X4013 * d_Convolv(2,0)(4a1_cv_W)/d_X8701
			JCudaTensor x2262;
			JCudaTensor x2263, x2264;
			x2263 = x2223;
			x2264 = x431;
			x2262 = x433.backward_data(x2263, x2264);

			// Dealloc(X4013)
			JCudaTensor x2265;
			x2265 = x2223;
			x2265.free();

			// val X4026 = X4025 * d_Convolv(1,0)(4a2_c_cv_W)/d_X8710
			JCudaTensor x2266;
			JCudaTensor x2267, x2268;
			x2267 = x2233;
			x2268 = x478;
			x2266 = x480.backward_data(x2267, x2268);

			// Dealloc(X4025)
			JCudaTensor x2269;
			x2269 = x2233;
			x2269.free();

			// V_4a1_bn_bias <~~ X8525
			float x2271, x2272;
			x2271 = lrn_rate_1;
			x2272 = momentum;
			JCudaTensor x2273;
			x2273 = x2219;
			x2270.update(x2273, x2271, x2272);

			// Dealloc(X8525)
			JCudaTensor x2274;
			x2274 = x2219;
			x2274.free();

			// V_4a1_bn_scale <~~ X8526
			float x2276, x2277;
			x2276 = lrn_rate_1;
			x2277 = momentum;
			JCudaTensor x2278;
			x2278 = x2214;
			x2275.update(x2278, x2276, x2277);

			// Dealloc(X8526)
			JCudaTensor x2279;
			x2279 = x2214;
			x2279.free();

			// 4a1_bn_scale <~~ V_4a1_bn_scale
			float x2280, x2281;
			x2280 = 1;
			x2281 = decay_1;
			JCudaTensor x2282;
			x2282 = x2275;
			x445.update(x2282, x2280, x2281);

			// 4a2_c_bn_scale <~~ V_4a2_c_bn_scale
			float x2283, x2284;
			x2283 = 1;
			x2284 = decay_1;
			JCudaTensor x2285;
			x2285 = x2242;
			x485.update(x2285, x2283, x2284);

			// 4a1_bn_bias <~~ V_4a1_bn_bias
			float x2286, x2287;
			x2286 = 1;
			x2287 = decay_1;
			JCudaTensor x2288;
			x2288 = x2270;
			x446.update(x2288, x2286, x2287);

			// 4a1_cv_W <~~ V_4a1_cv_W
			float x2289, x2290;
			x2289 = 1;
			x2290 = decay_1;
			JCudaTensor x2291;
			x2291 = x2247;
			x431.update(x2291, x2289, x2290);

			// 4a2_c_bn_bias <~~ V_4a2_c_bn_bias
			float x2292, x2293;
			x2292 = 1;
			x2293 = decay_1;
			JCudaTensor x2294;
			x2294 = x2252;
			x486.update(x2294, x2292, x2293);

			// 4a2_c_cv_W <~~ V_4a2_c_cv_W
			float x2295, x2296;
			x2295 = 1;
			x2296 = decay_1;
			JCudaTensor x2297;
			x2297 = x2257;
			x478.update(x2297, x2295, x2296);

			// val X4030 = X4026 * d_ReLU()(X8710)/d_X8709
			JCudaTensor x2298;
			JCudaTensor x2299, x2300;
			x2299 = x2266;
			x2300 = x472;
			x2298 = x457.backward(x2299, x2300);

			// Dealloc(X8710)
			JCudaTensor x2301;
			x2301 = x472;
			x2301.free();

			// val X8531 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_4a2_b_bn_bias
			JCudaTensor x2302;
			JCudaTensor x2303, x2304, x2305;
			x2303 = x2298;
			x2304 = x458;
			x2305 = x469;
			JCudaTensor[] x2306 = x471.backward(x2303,x2304,x2305);
			x2302 = x2306[2];

			// val X4031 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_X8708
			JCudaTensor x2307;
			x2307 = x2306[0];

			// val X8532 = X4030 * d_BatchNorm(4a2_b_bn)(X8708,4a2_b_bn_scale)/d_4a2_b_bn_scale
			JCudaTensor x2311;
			x2311 = x2306[1];

			// Dealloc(X8708)
			JCudaTensor x2315;
			x2315 = x458;
			x2315.free();

			// val X4032 = X4031 * d_Convolv(1,1)(4a2_b_cv_W)/d_X8707
			JCudaTensor x2316;
			JCudaTensor x2317, x2318;
			x2317 = x2307;
			x2318 = x462;
			x2316 = x464.backward_data(x2317, x2318);

			// V_4a2_b_cv_W <~~ X4031 * d_Convolv(1,1)(X8707)/d_4a2_b_cv_W
			float x2320, x2321;
			x2320 = lrn_rate_1;
			x2321 = momentum;
			JCudaTensor x2322, x2323;
			x2322 = x2307;
			x2323 = x455;
			x464.backward_filter(x2322, x2323, x2319, x2320, x2321);

			// Dealloc(X4031)
			JCudaTensor x2324;
			x2324 = x2307;
			x2324.free();

			// V_4a2_b_bn_scale <~~ X8532
			float x2326, x2327;
			x2326 = lrn_rate_1;
			x2327 = momentum;
			JCudaTensor x2328;
			x2328 = x2311;
			x2325.update(x2328, x2326, x2327);

			// Dealloc(X8532)
			JCudaTensor x2329;
			x2329 = x2311;
			x2329.free();

			// V_4a2_b_bn_bias <~~ X8531
			float x2331, x2332;
			x2331 = lrn_rate_1;
			x2332 = momentum;
			JCudaTensor x2333;
			x2333 = x2302;
			x2330.update(x2333, x2331, x2332);

			// Dealloc(X8531)
			JCudaTensor x2334;
			x2334 = x2302;
			x2334.free();

			// 4a2_b_cv_W <~~ V_4a2_b_cv_W
			float x2335, x2336;
			x2335 = 1;
			x2336 = decay_1;
			JCudaTensor x2337;
			x2337 = x2319;
			x462.update(x2337, x2335, x2336);

			// 4a2_b_bn_scale <~~ V_4a2_b_bn_scale
			float x2338, x2339;
			x2338 = 1;
			x2339 = decay_1;
			JCudaTensor x2340;
			x2340 = x2325;
			x469.update(x2340, x2338, x2339);

			// 4a2_b_bn_bias <~~ V_4a2_b_bn_bias
			float x2341, x2342;
			x2341 = 1;
			x2342 = decay_1;
			JCudaTensor x2343;
			x2343 = x2330;
			x470.update(x2343, x2341, x2342);

			// val X4036 = X4032 * d_ReLU()(X8707)/d_X8706
			JCudaTensor x2344;
			JCudaTensor x2345, x2346;
			x2345 = x2316;
			x2346 = x455;
			x2344 = x457.backward(x2345, x2346);

			// Dealloc(X8707)
			JCudaTensor x2347;
			x2347 = x455;
			x2347.free();

			// val X8529 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_4a2_a_bn_scale
			JCudaTensor x2348;
			JCudaTensor x2349, x2350, x2351;
			x2349 = x2344;
			x2350 = x434;
			x2351 = x452;
			JCudaTensor[] x2352 = x454.backward(x2349,x2350,x2351);
			x2348 = x2352[1];

			// val X8528 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_4a2_a_bn_bias
			JCudaTensor x2353;
			x2353 = x2352[2];

			// val X4037 = X4036 * d_BatchNorm(4a2_a_bn)(X8705,4a2_a_bn_scale)/d_X8705
			JCudaTensor x2357;
			x2357 = x2352[0];

			// Dealloc(X8705)
			JCudaTensor x2361;
			x2361 = x434;
			x2361.free();

			// val X4039 = (X4014 + X4037 * d_Convolv(2,0)(4a2_a_cv_W)/d_X8701)
			JCudaTensor x2362;
			JCudaTensor x2363;
			x2363 = x2262;
			JCudaTensor x2364, x2365;
			x2364 = x2357;
			x2365 = x438;
			x2362 = x440.backward_data(x2364,x2365, x2363);

			// V_4a2_a_cv_W <~~ X4037 * d_Convolv(2,0)(X8701)/d_4a2_a_cv_W
			float x2367, x2368;
			x2367 = lrn_rate_1;
			x2368 = momentum;
			JCudaTensor x2369, x2370;
			x2369 = x2357;
			x2370 = x425;
			x440.backward_filter(x2369, x2370, x2366, x2367, x2368);

			// Dealloc(X4037)
			JCudaTensor x2371;
			x2371 = x2357;
			x2371.free();

			// V_4a2_a_bn_bias <~~ X8528
			float x2373, x2374;
			x2373 = lrn_rate_1;
			x2374 = momentum;
			JCudaTensor x2375;
			x2375 = x2353;
			x2372.update(x2375, x2373, x2374);

			// Dealloc(X8528)
			JCudaTensor x2376;
			x2376 = x2353;
			x2376.free();

			// V_4a2_a_bn_scale <~~ X8529
			float x2378, x2379;
			x2378 = lrn_rate_1;
			x2379 = momentum;
			JCudaTensor x2380;
			x2380 = x2348;
			x2377.update(x2380, x2378, x2379);

			// Dealloc(X8529)
			JCudaTensor x2381;
			x2381 = x2348;
			x2381.free();

			// 4a2_a_cv_W <~~ V_4a2_a_cv_W
			float x2382, x2383;
			x2382 = 1;
			x2383 = decay_1;
			JCudaTensor x2384;
			x2384 = x2366;
			x438.update(x2384, x2382, x2383);

			// 4a2_a_bn_bias <~~ V_4a2_a_bn_bias
			float x2385, x2386;
			x2385 = 1;
			x2386 = decay_1;
			JCudaTensor x2387;
			x2387 = x2372;
			x453.update(x2387, x2385, x2386);

			// 4a2_a_bn_scale <~~ V_4a2_a_bn_scale
			float x2388, x2389;
			x2388 = 1;
			x2389 = decay_1;
			JCudaTensor x2390;
			x2390 = x2377;
			x452.update(x2390, x2388, x2389);

			// val X4087 = X4039 * d_ReLU()(X8701)/d_X8700
			JCudaTensor x2391;
			JCudaTensor x2392, x2393;
			x2392 = x2362;
			x2393 = x425;
			x2391 = x268.backward(x2392, x2393);

			// Dealloc(X8701)
			JCudaTensor x2394;
			x2394 = x425;
			x2394.free();

			// val X4097 = X4087.copy * d_ReLU()(X8699)/d_X8698
			JCudaTensor x2395;
			JCudaTensor x2396, x2397;
			x2396 = x2391;
			x2396 = x2396.clone();
			x2397 = x420;
			x2395 = x268.backward(x2396, x2397);

			// Dealloc(X8699)
			JCudaTensor x2398;
			x2398 = x420;
			x2398.free();

			// val X8522 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_3d_c_bn_bias
			JCudaTensor x2399;
			JCudaTensor x2400, x2401, x2402;
			x2400 = x2395;
			x2401 = x407;
			x2402 = x417;
			JCudaTensor[] x2403 = x419.backward(x2400,x2401,x2402);
			x2399 = x2403[2];

			// val X4098 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_X8697
			JCudaTensor x2404;
			x2404 = x2403[0];

			// val X8523 = X4097 * d_BatchNorm(3d_c_bn)(X8697,3d_c_bn_scale)/d_3d_c_bn_scale
			JCudaTensor x2408;
			x2408 = x2403[1];

			// Dealloc(X8697)
			JCudaTensor x2412;
			x2412 = x407;
			x2412.free();

			// val X4099 = X4098 * d_Convolv(1,0)(3d_c_cv_W)/d_X8696
			JCudaTensor x2413;
			JCudaTensor x2414, x2415;
			x2414 = x2404;
			x2415 = x411;
			x2413 = x258.backward_data(x2414, x2415);

			// V_3d_c_cv_W <~~ X4098 * d_Convolv(1,0)(X8696)/d_3d_c_cv_W
			float x2417, x2418;
			x2417 = lrn_rate_1;
			x2418 = momentum;
			JCudaTensor x2419, x2420;
			x2419 = x2404;
			x2420 = x405;
			x258.backward_filter(x2419, x2420, x2416, x2417, x2418);

			// Dealloc(X4098)
			JCudaTensor x2421;
			x2421 = x2404;
			x2421.free();

			// V_3d_c_bn_bias <~~ X8522
			float x2423, x2424;
			x2423 = lrn_rate_1;
			x2424 = momentum;
			JCudaTensor x2425;
			x2425 = x2399;
			x2422.update(x2425, x2423, x2424);

			// Dealloc(X8522)
			JCudaTensor x2426;
			x2426 = x2399;
			x2426.free();

			// V_3d_c_bn_scale <~~ X8523
			float x2428, x2429;
			x2428 = lrn_rate_1;
			x2429 = momentum;
			JCudaTensor x2430;
			x2430 = x2408;
			x2427.update(x2430, x2428, x2429);

			// Dealloc(X8523)
			JCudaTensor x2431;
			x2431 = x2408;
			x2431.free();

			// 3d_c_cv_W <~~ V_3d_c_cv_W
			float x2432, x2433;
			x2432 = 1;
			x2433 = decay_1;
			JCudaTensor x2434;
			x2434 = x2416;
			x411.update(x2434, x2432, x2433);

			// 3d_c_bn_bias <~~ V_3d_c_bn_bias
			float x2435, x2436;
			x2435 = 1;
			x2436 = decay_1;
			JCudaTensor x2437;
			x2437 = x2422;
			x418.update(x2437, x2435, x2436);

			// 3d_c_bn_scale <~~ V_3d_c_bn_scale
			float x2438, x2439;
			x2438 = 1;
			x2439 = decay_1;
			JCudaTensor x2440;
			x2440 = x2427;
			x417.update(x2440, x2438, x2439);

			// val X4103 = X4099 * d_ReLU()(X8696)/d_X8695
			JCudaTensor x2441;
			JCudaTensor x2442, x2443;
			x2442 = x2413;
			x2443 = x405;
			x2441 = x235.backward(x2442, x2443);

			// Dealloc(X8696)
			JCudaTensor x2444;
			x2444 = x405;
			x2444.free();

			// val X8519 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_3d_b_bn_bias
			JCudaTensor x2445;
			JCudaTensor x2446, x2447, x2448;
			x2446 = x2441;
			x2447 = x392;
			x2448 = x402;
			JCudaTensor[] x2449 = x404.backward(x2446,x2447,x2448);
			x2445 = x2449[2];

			// val X4104 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_X8694
			JCudaTensor x2450;
			x2450 = x2449[0];

			// val X8520 = X4103 * d_BatchNorm(3d_b_bn)(X8694,3d_b_bn_scale)/d_3d_b_bn_scale
			JCudaTensor x2454;
			x2454 = x2449[1];

			// Dealloc(X8694)
			JCudaTensor x2458;
			x2458 = x392;
			x2458.free();

			// V_3d_b_bn_bias <~~ X8519
			float x2460, x2461;
			x2460 = lrn_rate_1;
			x2461 = momentum;
			JCudaTensor x2462;
			x2462 = x2445;
			x2459.update(x2462, x2460, x2461);

			// Dealloc(X8519)
			JCudaTensor x2463;
			x2463 = x2445;
			x2463.free();

			// V_3d_b_bn_scale <~~ X8520
			float x2465, x2466;
			x2465 = lrn_rate_1;
			x2466 = momentum;
			JCudaTensor x2467;
			x2467 = x2454;
			x2464.update(x2467, x2465, x2466);

			// Dealloc(X8520)
			JCudaTensor x2468;
			x2468 = x2454;
			x2468.free();

			// val X4105 = X4104 * d_Convolv(1,1)(3d_b_cv_W)/d_X8693
			JCudaTensor x2469;
			JCudaTensor x2470, x2471;
			x2470 = x2450;
			x2471 = x396;
			x2469 = x242.backward_data(x2470, x2471);

			// V_3d_b_cv_W <~~ X4104 * d_Convolv(1,1)(X8693)/d_3d_b_cv_W
			float x2473, x2474;
			x2473 = lrn_rate_1;
			x2474 = momentum;
			JCudaTensor x2475, x2476;
			x2475 = x2450;
			x2476 = x390;
			x242.backward_filter(x2475, x2476, x2472, x2473, x2474);

			// Dealloc(X4104)
			JCudaTensor x2477;
			x2477 = x2450;
			x2477.free();

			// 3d_b_bn_bias <~~ V_3d_b_bn_bias
			float x2478, x2479;
			x2478 = 1;
			x2479 = decay_1;
			JCudaTensor x2480;
			x2480 = x2459;
			x403.update(x2480, x2478, x2479);

			// 3d_b_bn_scale <~~ V_3d_b_bn_scale
			float x2481, x2482;
			x2481 = 1;
			x2482 = decay_1;
			JCudaTensor x2483;
			x2483 = x2464;
			x402.update(x2483, x2481, x2482);

			// 3d_b_cv_W <~~ V_3d_b_cv_W
			float x2484, x2485;
			x2484 = 1;
			x2485 = decay_1;
			JCudaTensor x2486;
			x2486 = x2472;
			x396.update(x2486, x2484, x2485);

			// val X4109 = X4105 * d_ReLU()(X8693)/d_X8692
			JCudaTensor x2487;
			JCudaTensor x2488, x2489;
			x2488 = x2469;
			x2489 = x390;
			x2487 = x235.backward(x2488, x2489);

			// Dealloc(X8693)
			JCudaTensor x2490;
			x2490 = x390;
			x2490.free();

			// val X4110 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_X8691
			JCudaTensor x2491;
			JCudaTensor x2492, x2493, x2494;
			x2492 = x2487;
			x2493 = x377;
			x2494 = x387;
			JCudaTensor[] x2495 = x389.backward(x2492,x2493,x2494);
			x2491 = x2495[0];

			// val X8517 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_3d_a_bn_scale
			JCudaTensor x2496;
			x2496 = x2495[1];

			// val X8516 = X4109 * d_BatchNorm(3d_a_bn)(X8691,3d_a_bn_scale)/d_3d_a_bn_bias
			JCudaTensor x2500;
			x2500 = x2495[2];

			// Dealloc(X8691)
			JCudaTensor x2504;
			x2504 = x377;
			x2504.free();

			// V_3d_a_bn_scale <~~ X8517
			float x2506, x2507;
			x2506 = lrn_rate_1;
			x2507 = momentum;
			JCudaTensor x2508;
			x2508 = x2496;
			x2505.update(x2508, x2506, x2507);

			// Dealloc(X8517)
			JCudaTensor x2509;
			x2509 = x2496;
			x2509.free();

			// V_3d_a_bn_bias <~~ X8516
			float x2511, x2512;
			x2511 = lrn_rate_1;
			x2512 = momentum;
			JCudaTensor x2513;
			x2513 = x2500;
			x2510.update(x2513, x2511, x2512);

			// Dealloc(X8516)
			JCudaTensor x2514;
			x2514 = x2500;
			x2514.free();

			// val X4111 = X4110 * d_Convolv(1,0)(3d_a_cv_W)/d_X8690
			JCudaTensor x2515;
			JCudaTensor x2516, x2517;
			x2516 = x2491;
			x2517 = x381;
			x2515 = x282.backward_data(x2516, x2517);

			// V_3d_a_cv_W <~~ X4110 * d_Convolv(1,0)(X8690)/d_3d_a_cv_W
			float x2519, x2520;
			x2519 = lrn_rate_1;
			x2520 = momentum;
			JCudaTensor x2521, x2522;
			x2521 = x2491;
			x2522 = x375;
			x282.backward_filter(x2521, x2522, x2518, x2519, x2520);

			// Dealloc(X4110)
			JCudaTensor x2523;
			x2523 = x2491;
			x2523.free();

			// 3d_a_bn_scale <~~ V_3d_a_bn_scale
			float x2524, x2525;
			x2524 = 1;
			x2525 = decay_1;
			JCudaTensor x2526;
			x2526 = x2505;
			x387.update(x2526, x2524, x2525);

			// 3d_a_bn_bias <~~ V_3d_a_bn_bias
			float x2527, x2528;
			x2527 = 1;
			x2528 = decay_1;
			JCudaTensor x2529;
			x2529 = x2510;
			x388.update(x2529, x2527, x2528);

			// 3d_a_cv_W <~~ V_3d_a_cv_W
			float x2530, x2531;
			x2530 = 1;
			x2531 = decay_1;
			JCudaTensor x2532;
			x2532 = x2518;
			x381.update(x2532, x2530, x2531);

			// val X4112 = (X4111 + X4087)
			JCudaTensor x2533;
			JCudaTensor x2534, x2535;
			x2534 = x2515;
			x2535 = x2391;
			x2533 = x2534.plus_i(x2535);

			// Dealloc(X4087)
			JCudaTensor x2536;
			x2536 = x2391;
			x2536.free();

			// val X4124 = X4112 * d_ReLU()(X8690)/d_X8689
			JCudaTensor x2537;
			JCudaTensor x2538, x2539;
			x2538 = x2533;
			x2539 = x375;
			x2537 = x268.backward(x2538, x2539);

			// Dealloc(X8690)
			JCudaTensor x2540;
			x2540 = x375;
			x2540.free();

			// val X4134 = X4124.copy * d_ReLU()(X8688)/d_X8687
			JCudaTensor x2541;
			JCudaTensor x2542, x2543;
			x2542 = x2537;
			x2542 = x2542.clone();
			x2543 = x370;
			x2541 = x268.backward(x2542, x2543);

			// Dealloc(X8688)
			JCudaTensor x2544;
			x2544 = x370;
			x2544.free();

			// val X8513 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_3c_c_bn_bias
			JCudaTensor x2545;
			JCudaTensor x2546, x2547, x2548;
			x2546 = x2541;
			x2547 = x357;
			x2548 = x367;
			JCudaTensor[] x2549 = x369.backward(x2546,x2547,x2548);
			x2545 = x2549[2];

			// val X8514 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_3c_c_bn_scale
			JCudaTensor x2550;
			x2550 = x2549[1];

			// val X4135 = X4134 * d_BatchNorm(3c_c_bn)(X8686,3c_c_bn_scale)/d_X8686
			JCudaTensor x2554;
			x2554 = x2549[0];

			// Dealloc(X8686)
			JCudaTensor x2558;
			x2558 = x357;
			x2558.free();

			// V_3c_c_bn_bias <~~ X8513
			float x2560, x2561;
			x2560 = lrn_rate_1;
			x2561 = momentum;
			JCudaTensor x2562;
			x2562 = x2545;
			x2559.update(x2562, x2560, x2561);

			// Dealloc(X8513)
			JCudaTensor x2563;
			x2563 = x2545;
			x2563.free();

			// val X4136 = X4135 * d_Convolv(1,0)(3c_c_cv_W)/d_X8685
			JCudaTensor x2564;
			JCudaTensor x2565, x2566;
			x2565 = x2554;
			x2566 = x361;
			x2564 = x258.backward_data(x2565, x2566);

			// V_3c_c_cv_W <~~ X4135 * d_Convolv(1,0)(X8685)/d_3c_c_cv_W
			float x2568, x2569;
			x2568 = lrn_rate_1;
			x2569 = momentum;
			JCudaTensor x2570, x2571;
			x2570 = x2554;
			x2571 = x355;
			x258.backward_filter(x2570, x2571, x2567, x2568, x2569);

			// Dealloc(X4135)
			JCudaTensor x2572;
			x2572 = x2554;
			x2572.free();

			// V_3c_c_bn_scale <~~ X8514
			float x2574, x2575;
			x2574 = lrn_rate_1;
			x2575 = momentum;
			JCudaTensor x2576;
			x2576 = x2550;
			x2573.update(x2576, x2574, x2575);

			// Dealloc(X8514)
			JCudaTensor x2577;
			x2577 = x2550;
			x2577.free();

			// 3c_c_bn_bias <~~ V_3c_c_bn_bias
			float x2578, x2579;
			x2578 = 1;
			x2579 = decay_1;
			JCudaTensor x2580;
			x2580 = x2559;
			x368.update(x2580, x2578, x2579);

			// 3c_c_cv_W <~~ V_3c_c_cv_W
			float x2581, x2582;
			x2581 = 1;
			x2582 = decay_1;
			JCudaTensor x2583;
			x2583 = x2567;
			x361.update(x2583, x2581, x2582);

			// 3c_c_bn_scale <~~ V_3c_c_bn_scale
			float x2584, x2585;
			x2584 = 1;
			x2585 = decay_1;
			JCudaTensor x2586;
			x2586 = x2573;
			x367.update(x2586, x2584, x2585);

			// val X4140 = X4136 * d_ReLU()(X8685)/d_X8684
			JCudaTensor x2587;
			JCudaTensor x2588, x2589;
			x2588 = x2564;
			x2589 = x355;
			x2587 = x235.backward(x2588, x2589);

			// Dealloc(X8685)
			JCudaTensor x2590;
			x2590 = x355;
			x2590.free();

			// val X8510 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_3c_b_bn_bias
			JCudaTensor x2591;
			JCudaTensor x2592, x2593, x2594;
			x2592 = x2587;
			x2593 = x342;
			x2594 = x352;
			JCudaTensor[] x2595 = x354.backward(x2592,x2593,x2594);
			x2591 = x2595[2];

			// val X8511 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_3c_b_bn_scale
			JCudaTensor x2596;
			x2596 = x2595[1];

			// val X4141 = X4140 * d_BatchNorm(3c_b_bn)(X8683,3c_b_bn_scale)/d_X8683
			JCudaTensor x2600;
			x2600 = x2595[0];

			// Dealloc(X8683)
			JCudaTensor x2604;
			x2604 = x342;
			x2604.free();

			// val X4142 = X4141 * d_Convolv(1,1)(3c_b_cv_W)/d_X8682
			JCudaTensor x2605;
			JCudaTensor x2606, x2607;
			x2606 = x2600;
			x2607 = x346;
			x2605 = x242.backward_data(x2606, x2607);

			// V_3c_b_bn_scale <~~ X8511
			float x2609, x2610;
			x2609 = lrn_rate_1;
			x2610 = momentum;
			JCudaTensor x2611;
			x2611 = x2596;
			x2608.update(x2611, x2609, x2610);

			// Dealloc(X8511)
			JCudaTensor x2612;
			x2612 = x2596;
			x2612.free();

			// V_3c_b_cv_W <~~ X4141 * d_Convolv(1,1)(X8682)/d_3c_b_cv_W
			float x2614, x2615;
			x2614 = lrn_rate_1;
			x2615 = momentum;
			JCudaTensor x2616, x2617;
			x2616 = x2600;
			x2617 = x340;
			x242.backward_filter(x2616, x2617, x2613, x2614, x2615);

			// Dealloc(X4141)
			JCudaTensor x2618;
			x2618 = x2600;
			x2618.free();

			// V_3c_b_bn_bias <~~ X8510
			float x2620, x2621;
			x2620 = lrn_rate_1;
			x2621 = momentum;
			JCudaTensor x2622;
			x2622 = x2591;
			x2619.update(x2622, x2620, x2621);

			// Dealloc(X8510)
			JCudaTensor x2623;
			x2623 = x2591;
			x2623.free();

			// 3c_b_bn_scale <~~ V_3c_b_bn_scale
			float x2624, x2625;
			x2624 = 1;
			x2625 = decay_1;
			JCudaTensor x2626;
			x2626 = x2608;
			x352.update(x2626, x2624, x2625);

			// 3c_b_cv_W <~~ V_3c_b_cv_W
			float x2627, x2628;
			x2627 = 1;
			x2628 = decay_1;
			JCudaTensor x2629;
			x2629 = x2613;
			x346.update(x2629, x2627, x2628);

			// 3c_b_bn_bias <~~ V_3c_b_bn_bias
			float x2630, x2631;
			x2630 = 1;
			x2631 = decay_1;
			JCudaTensor x2632;
			x2632 = x2619;
			x353.update(x2632, x2630, x2631);

			// val X4146 = X4142 * d_ReLU()(X8682)/d_X8681
			JCudaTensor x2633;
			JCudaTensor x2634, x2635;
			x2634 = x2605;
			x2635 = x340;
			x2633 = x235.backward(x2634, x2635);

			// Dealloc(X8682)
			JCudaTensor x2636;
			x2636 = x340;
			x2636.free();

			// val X4147 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_X8680
			JCudaTensor x2637;
			JCudaTensor x2638, x2639, x2640;
			x2638 = x2633;
			x2639 = x327;
			x2640 = x337;
			JCudaTensor[] x2641 = x339.backward(x2638,x2639,x2640);
			x2637 = x2641[0];

			// val X8508 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_3c_a_bn_scale
			JCudaTensor x2642;
			x2642 = x2641[1];

			// val X8507 = X4146 * d_BatchNorm(3c_a_bn)(X8680,3c_a_bn_scale)/d_3c_a_bn_bias
			JCudaTensor x2646;
			x2646 = x2641[2];

			// Dealloc(X8680)
			JCudaTensor x2650;
			x2650 = x327;
			x2650.free();

			// val X4148 = X4147 * d_Convolv(1,0)(3c_a_cv_W)/d_X8679
			JCudaTensor x2651;
			JCudaTensor x2652, x2653;
			x2652 = x2637;
			x2653 = x331;
			x2651 = x282.backward_data(x2652, x2653);

			// V_3c_a_cv_W <~~ X4147 * d_Convolv(1,0)(X8679)/d_3c_a_cv_W
			float x2655, x2656;
			x2655 = lrn_rate_1;
			x2656 = momentum;
			JCudaTensor x2657, x2658;
			x2657 = x2637;
			x2658 = x325;
			x282.backward_filter(x2657, x2658, x2654, x2655, x2656);

			// Dealloc(X4147)
			JCudaTensor x2659;
			x2659 = x2637;
			x2659.free();

			// V_3c_a_bn_bias <~~ X8507
			float x2661, x2662;
			x2661 = lrn_rate_1;
			x2662 = momentum;
			JCudaTensor x2663;
			x2663 = x2646;
			x2660.update(x2663, x2661, x2662);

			// Dealloc(X8507)
			JCudaTensor x2664;
			x2664 = x2646;
			x2664.free();

			// V_3c_a_bn_scale <~~ X8508
			float x2666, x2667;
			x2666 = lrn_rate_1;
			x2667 = momentum;
			JCudaTensor x2668;
			x2668 = x2642;
			x2665.update(x2668, x2666, x2667);

			// Dealloc(X8508)
			JCudaTensor x2669;
			x2669 = x2642;
			x2669.free();

			// 3c_a_cv_W <~~ V_3c_a_cv_W
			float x2670, x2671;
			x2670 = 1;
			x2671 = decay_1;
			JCudaTensor x2672;
			x2672 = x2654;
			x331.update(x2672, x2670, x2671);

			// 3c_a_bn_bias <~~ V_3c_a_bn_bias
			float x2673, x2674;
			x2673 = 1;
			x2674 = decay_1;
			JCudaTensor x2675;
			x2675 = x2660;
			x338.update(x2675, x2673, x2674);

			// 3c_a_bn_scale <~~ V_3c_a_bn_scale
			float x2676, x2677;
			x2676 = 1;
			x2677 = decay_1;
			JCudaTensor x2678;
			x2678 = x2665;
			x337.update(x2678, x2676, x2677);

			// val X4149 = (X4148 + X4124)
			JCudaTensor x2679;
			JCudaTensor x2680, x2681;
			x2680 = x2651;
			x2681 = x2537;
			x2679 = x2680.plus_i(x2681);

			// Dealloc(X4124)
			JCudaTensor x2682;
			x2682 = x2537;
			x2682.free();

			// val X4161 = X4149 * d_ReLU()(X8679)/d_X8678
			JCudaTensor x2683;
			JCudaTensor x2684, x2685;
			x2684 = x2679;
			x2685 = x325;
			x2683 = x268.backward(x2684, x2685);

			// Dealloc(X8679)
			JCudaTensor x2686;
			x2686 = x325;
			x2686.free();

			// val X4171 = X4161.copy * d_ReLU()(X8677)/d_X8676
			JCudaTensor x2687;
			JCudaTensor x2688, x2689;
			x2688 = x2683;
			x2688 = x2688.clone();
			x2689 = x320;
			x2687 = x268.backward(x2688, x2689);

			// Dealloc(X8677)
			JCudaTensor x2690;
			x2690 = x320;
			x2690.free();

			// val X4172 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_X8675
			JCudaTensor x2691;
			JCudaTensor x2692, x2693, x2694;
			x2692 = x2687;
			x2693 = x307;
			x2694 = x317;
			JCudaTensor[] x2695 = x319.backward(x2692,x2693,x2694);
			x2691 = x2695[0];

			// val X8505 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_3b_c_bn_scale
			JCudaTensor x2696;
			x2696 = x2695[1];

			// val X8504 = X4171 * d_BatchNorm(3b_c_bn)(X8675,3b_c_bn_scale)/d_3b_c_bn_bias
			JCudaTensor x2700;
			x2700 = x2695[2];

			// Dealloc(X8675)
			JCudaTensor x2704;
			x2704 = x307;
			x2704.free();

			// val X4173 = X4172 * d_Convolv(1,0)(3b_c_cv_W)/d_X8674
			JCudaTensor x2705;
			JCudaTensor x2706, x2707;
			x2706 = x2691;
			x2707 = x311;
			x2705 = x258.backward_data(x2706, x2707);

			// V_3b_c_bn_bias <~~ X8504
			float x2709, x2710;
			x2709 = lrn_rate_1;
			x2710 = momentum;
			JCudaTensor x2711;
			x2711 = x2700;
			x2708.update(x2711, x2709, x2710);

			// Dealloc(X8504)
			JCudaTensor x2712;
			x2712 = x2700;
			x2712.free();

			// V_3b_c_bn_scale <~~ X8505
			float x2714, x2715;
			x2714 = lrn_rate_1;
			x2715 = momentum;
			JCudaTensor x2716;
			x2716 = x2696;
			x2713.update(x2716, x2714, x2715);

			// Dealloc(X8505)
			JCudaTensor x2717;
			x2717 = x2696;
			x2717.free();

			// V_3b_c_cv_W <~~ X4172 * d_Convolv(1,0)(X8674)/d_3b_c_cv_W
			float x2719, x2720;
			x2719 = lrn_rate_1;
			x2720 = momentum;
			JCudaTensor x2721, x2722;
			x2721 = x2691;
			x2722 = x305;
			x258.backward_filter(x2721, x2722, x2718, x2719, x2720);

			// Dealloc(X4172)
			JCudaTensor x2723;
			x2723 = x2691;
			x2723.free();

			// 3b_c_bn_bias <~~ V_3b_c_bn_bias
			float x2724, x2725;
			x2724 = 1;
			x2725 = decay_1;
			JCudaTensor x2726;
			x2726 = x2708;
			x318.update(x2726, x2724, x2725);

			// 3b_c_bn_scale <~~ V_3b_c_bn_scale
			float x2727, x2728;
			x2727 = 1;
			x2728 = decay_1;
			JCudaTensor x2729;
			x2729 = x2713;
			x317.update(x2729, x2727, x2728);

			// 3b_c_cv_W <~~ V_3b_c_cv_W
			float x2730, x2731;
			x2730 = 1;
			x2731 = decay_1;
			JCudaTensor x2732;
			x2732 = x2718;
			x311.update(x2732, x2730, x2731);

			// val X4177 = X4173 * d_ReLU()(X8674)/d_X8673
			JCudaTensor x2733;
			JCudaTensor x2734, x2735;
			x2734 = x2705;
			x2735 = x305;
			x2733 = x235.backward(x2734, x2735);

			// Dealloc(X8674)
			JCudaTensor x2736;
			x2736 = x305;
			x2736.free();

			// val X8501 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_3b_b_bn_bias
			JCudaTensor x2737;
			JCudaTensor x2738, x2739, x2740;
			x2738 = x2733;
			x2739 = x292;
			x2740 = x302;
			JCudaTensor[] x2741 = x304.backward(x2738,x2739,x2740);
			x2737 = x2741[2];

			// val X4178 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_X8672
			JCudaTensor x2742;
			x2742 = x2741[0];

			// val X8502 = X4177 * d_BatchNorm(3b_b_bn)(X8672,3b_b_bn_scale)/d_3b_b_bn_scale
			JCudaTensor x2746;
			x2746 = x2741[1];

			// Dealloc(X8672)
			JCudaTensor x2750;
			x2750 = x292;
			x2750.free();

			// V_3b_b_bn_scale <~~ X8502
			float x2752, x2753;
			x2752 = lrn_rate_1;
			x2753 = momentum;
			JCudaTensor x2754;
			x2754 = x2746;
			x2751.update(x2754, x2752, x2753);

			// Dealloc(X8502)
			JCudaTensor x2755;
			x2755 = x2746;
			x2755.free();

			// V_3b_b_bn_bias <~~ X8501
			float x2757, x2758;
			x2757 = lrn_rate_1;
			x2758 = momentum;
			JCudaTensor x2759;
			x2759 = x2737;
			x2756.update(x2759, x2757, x2758);

			// Dealloc(X8501)
			JCudaTensor x2760;
			x2760 = x2737;
			x2760.free();

			// val X4179 = X4178 * d_Convolv(1,1)(3b_b_cv_W)/d_X8671
			JCudaTensor x2761;
			JCudaTensor x2762, x2763;
			x2762 = x2742;
			x2763 = x296;
			x2761 = x242.backward_data(x2762, x2763);

			// V_3b_b_cv_W <~~ X4178 * d_Convolv(1,1)(X8671)/d_3b_b_cv_W
			float x2765, x2766;
			x2765 = lrn_rate_1;
			x2766 = momentum;
			JCudaTensor x2767, x2768;
			x2767 = x2742;
			x2768 = x290;
			x242.backward_filter(x2767, x2768, x2764, x2765, x2766);

			// Dealloc(X4178)
			JCudaTensor x2769;
			x2769 = x2742;
			x2769.free();

			// 3b_b_bn_scale <~~ V_3b_b_bn_scale
			float x2770, x2771;
			x2770 = 1;
			x2771 = decay_1;
			JCudaTensor x2772;
			x2772 = x2751;
			x302.update(x2772, x2770, x2771);

			// 3b_b_bn_bias <~~ V_3b_b_bn_bias
			float x2773, x2774;
			x2773 = 1;
			x2774 = decay_1;
			JCudaTensor x2775;
			x2775 = x2756;
			x303.update(x2775, x2773, x2774);

			// 3b_b_cv_W <~~ V_3b_b_cv_W
			float x2776, x2777;
			x2776 = 1;
			x2777 = decay_1;
			JCudaTensor x2778;
			x2778 = x2764;
			x296.update(x2778, x2776, x2777);

			// val X4183 = X4179 * d_ReLU()(X8671)/d_X8670
			JCudaTensor x2779;
			JCudaTensor x2780, x2781;
			x2780 = x2761;
			x2781 = x290;
			x2779 = x235.backward(x2780, x2781);

			// Dealloc(X8671)
			JCudaTensor x2782;
			x2782 = x290;
			x2782.free();

			// val X8498 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_3b_a_bn_bias
			JCudaTensor x2783;
			JCudaTensor x2784, x2785, x2786;
			x2784 = x2779;
			x2785 = x276;
			x2786 = x287;
			JCudaTensor[] x2787 = x289.backward(x2784,x2785,x2786);
			x2783 = x2787[2];

			// val X4184 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_X8669
			JCudaTensor x2788;
			x2788 = x2787[0];

			// val X8499 = X4183 * d_BatchNorm(3b_a_bn)(X8669,3b_a_bn_scale)/d_3b_a_bn_scale
			JCudaTensor x2792;
			x2792 = x2787[1];

			// Dealloc(X8669)
			JCudaTensor x2796;
			x2796 = x276;
			x2796.free();

			// V_3b_a_bn_scale <~~ X8499
			float x2798, x2799;
			x2798 = lrn_rate_1;
			x2799 = momentum;
			JCudaTensor x2800;
			x2800 = x2792;
			x2797.update(x2800, x2798, x2799);

			// Dealloc(X8499)
			JCudaTensor x2801;
			x2801 = x2792;
			x2801.free();

			// V_3b_a_bn_bias <~~ X8498
			float x2803, x2804;
			x2803 = lrn_rate_1;
			x2804 = momentum;
			JCudaTensor x2805;
			x2805 = x2783;
			x2802.update(x2805, x2803, x2804);

			// Dealloc(X8498)
			JCudaTensor x2806;
			x2806 = x2783;
			x2806.free();

			// val X4185 = X4184 * d_Convolv(1,0)(3b_a_cv_W)/d_X8668
			JCudaTensor x2807;
			JCudaTensor x2808, x2809;
			x2808 = x2788;
			x2809 = x280;
			x2807 = x282.backward_data(x2808, x2809);

			// V_3b_a_cv_W <~~ X4184 * d_Convolv(1,0)(X8668)/d_3b_a_cv_W
			float x2811, x2812;
			x2811 = lrn_rate_1;
			x2812 = momentum;
			JCudaTensor x2813, x2814;
			x2813 = x2788;
			x2814 = x274;
			x282.backward_filter(x2813, x2814, x2810, x2811, x2812);

			// Dealloc(X4184)
			JCudaTensor x2815;
			x2815 = x2788;
			x2815.free();

			// 3b_a_bn_scale <~~ V_3b_a_bn_scale
			float x2816, x2817;
			x2816 = 1;
			x2817 = decay_1;
			JCudaTensor x2818;
			x2818 = x2797;
			x287.update(x2818, x2816, x2817);

			// 3b_a_bn_bias <~~ V_3b_a_bn_bias
			float x2819, x2820;
			x2819 = 1;
			x2820 = decay_1;
			JCudaTensor x2821;
			x2821 = x2802;
			x288.update(x2821, x2819, x2820);

			// 3b_a_cv_W <~~ V_3b_a_cv_W
			float x2822, x2823;
			x2822 = 1;
			x2823 = decay_1;
			JCudaTensor x2824;
			x2824 = x2810;
			x280.update(x2824, x2822, x2823);

			// val X4186 = (X4185 + X4161)
			JCudaTensor x2825;
			JCudaTensor x2826, x2827;
			x2826 = x2807;
			x2827 = x2683;
			x2825 = x2826.plus_i(x2827);

			// Dealloc(X4161)
			JCudaTensor x2828;
			x2828 = x2683;
			x2828.free();

			// val X4201 = X4186 * d_ReLU()(X8668)/d_X8667
			JCudaTensor x2829;
			JCudaTensor x2830, x2831;
			x2830 = x2825;
			x2831 = x274;
			x2829 = x268.backward(x2830, x2831);

			// Dealloc(X8668)
			JCudaTensor x2832;
			x2832 = x274;
			x2832.free();

			// val X4205 = X4201.copy * d_ReLU()(X8657)/d_X8656
			JCudaTensor x2833;
			JCudaTensor x2834, x2835;
			x2834 = x2829;
			x2834 = x2834.clone();
			x2835 = x266;
			x2833 = x268.backward(x2834, x2835);

			// Dealloc(X8657)
			JCudaTensor x2836;
			x2836 = x266;
			x2836.free();

			// val X4217 = X4201.copy * d_ReLU()(X8666)/d_X8665
			JCudaTensor x2837;
			JCudaTensor x2838, x2839;
			x2838 = x2829;
			x2838 = x2838.clone();
			x2839 = x269;
			x2837 = x268.backward(x2838, x2839);

			// Dealloc(X4201)
			JCudaTensor x2840;
			x2840 = x2829;
			x2840.free();

			// Dealloc(X8666)
			JCudaTensor x2841;
			x2841 = x269;
			x2841.free();

			// val X4206 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_X8655
			JCudaTensor x2842;
			JCudaTensor x2843, x2844, x2845;
			x2843 = x2833;
			x2844 = x212;
			x2845 = x223;
			JCudaTensor[] x2846 = x225.backward(x2843,x2844,x2845);
			x2842 = x2846[0];

			// val X8495 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_3a2_c_bn_bias
			JCudaTensor x2847;
			JCudaTensor x2848, x2849, x2850;
			x2848 = x2837;
			x2849 = x252;
			x2850 = x263;
			JCudaTensor[] x2851 = x265.backward(x2848,x2849,x2850);
			x2847 = x2851[2];

			// val X8487 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_3a1_bn_scale
			JCudaTensor x2852;
			x2852 = x2846[1];

			// val X4218 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_X8664
			JCudaTensor x2856;
			x2856 = x2851[0];

			// val X8486 = X4205 * d_BatchNorm(3a1_bn)(X8655,3a1_bn_scale)/d_3a1_bn_bias
			JCudaTensor x2860;
			x2860 = x2846[2];

			// Dealloc(X8655)
			JCudaTensor x2864;
			x2864 = x212;
			x2864.free();

			// val X8496 = X4217 * d_BatchNorm(3a2_c_bn)(X8664,3a2_c_bn_scale)/d_3a2_c_bn_scale
			JCudaTensor x2865;
			x2865 = x2851[1];

			// Dealloc(X8664)
			JCudaTensor x2869;
			x2869 = x252;
			x2869.free();

			// V_3a1_bn_bias <~~ X8486
			float x2871, x2872;
			x2871 = lrn_rate_1;
			x2872 = momentum;
			JCudaTensor x2873;
			x2873 = x2860;
			x2870.update(x2873, x2871, x2872);

			// Dealloc(X8486)
			JCudaTensor x2874;
			x2874 = x2860;
			x2874.free();

			// val X4207 = X4206 * d_Convolv(2,0)(3a1_cv_W)/d_X8654
			JCudaTensor x2875;
			JCudaTensor x2876, x2877;
			x2876 = x2842;
			x2877 = x216;
			x2875 = x218.backward_data(x2876, x2877);

			// V_3a2_c_bn_bias <~~ X8495
			float x2879, x2880;
			x2879 = lrn_rate_1;
			x2880 = momentum;
			JCudaTensor x2881;
			x2881 = x2847;
			x2878.update(x2881, x2879, x2880);

			// Dealloc(X8495)
			JCudaTensor x2882;
			x2882 = x2847;
			x2882.free();

			// V_3a2_c_bn_scale <~~ X8496
			float x2884, x2885;
			x2884 = lrn_rate_1;
			x2885 = momentum;
			JCudaTensor x2886;
			x2886 = x2865;
			x2883.update(x2886, x2884, x2885);

			// Dealloc(X8496)
			JCudaTensor x2887;
			x2887 = x2865;
			x2887.free();

			// val X4219 = X4218 * d_Convolv(1,0)(3a2_c_cv_W)/d_X8663
			JCudaTensor x2888;
			JCudaTensor x2889, x2890;
			x2889 = x2856;
			x2890 = x256;
			x2888 = x258.backward_data(x2889, x2890);

			// V_3a2_c_cv_W <~~ X4218 * d_Convolv(1,0)(X8663)/d_3a2_c_cv_W
			float x2892, x2893;
			x2892 = lrn_rate_1;
			x2893 = momentum;
			JCudaTensor x2894, x2895;
			x2894 = x2856;
			x2895 = x250;
			x258.backward_filter(x2894, x2895, x2891, x2892, x2893);

			// Dealloc(X4218)
			JCudaTensor x2896;
			x2896 = x2856;
			x2896.free();

			// V_3a1_cv_W <~~ X4206 * d_Convolv(2,0)(X8654)/d_3a1_cv_W
			float x2898, x2899;
			x2898 = lrn_rate_1;
			x2899 = momentum;
			JCudaTensor x2900, x2901;
			x2900 = x2842;
			x2901 = x203;
			x218.backward_filter(x2900, x2901, x2897, x2898, x2899);

			// Dealloc(X4206)
			JCudaTensor x2902;
			x2902 = x2842;
			x2902.free();

			// V_3a1_bn_scale <~~ X8487
			float x2904, x2905;
			x2904 = lrn_rate_1;
			x2905 = momentum;
			JCudaTensor x2906;
			x2906 = x2852;
			x2903.update(x2906, x2904, x2905);

			// Dealloc(X8487)
			JCudaTensor x2907;
			x2907 = x2852;
			x2907.free();

			// 3a1_bn_scale <~~ V_3a1_bn_scale
			float x2908, x2909;
			x2908 = 1;
			x2909 = decay_1;
			JCudaTensor x2910;
			x2910 = x2903;
			x223.update(x2910, x2908, x2909);

			// 3a2_c_bn_scale <~~ V_3a2_c_bn_scale
			float x2911, x2912;
			x2911 = 1;
			x2912 = decay_1;
			JCudaTensor x2913;
			x2913 = x2883;
			x263.update(x2913, x2911, x2912);

			// 3a1_cv_W <~~ V_3a1_cv_W
			float x2914, x2915;
			x2914 = 1;
			x2915 = decay_1;
			JCudaTensor x2916;
			x2916 = x2897;
			x216.update(x2916, x2914, x2915);

			// 3a2_c_bn_bias <~~ V_3a2_c_bn_bias
			float x2917, x2918;
			x2917 = 1;
			x2918 = decay_1;
			JCudaTensor x2919;
			x2919 = x2878;
			x264.update(x2919, x2917, x2918);

			// 3a2_c_cv_W <~~ V_3a2_c_cv_W
			float x2920, x2921;
			x2920 = 1;
			x2921 = decay_1;
			JCudaTensor x2922;
			x2922 = x2891;
			x256.update(x2922, x2920, x2921);

			// 3a1_bn_bias <~~ V_3a1_bn_bias
			float x2923, x2924;
			x2923 = 1;
			x2924 = decay_1;
			JCudaTensor x2925;
			x2925 = x2870;
			x224.update(x2925, x2923, x2924);

			// val X4223 = X4219 * d_ReLU()(X8663)/d_X8662
			JCudaTensor x2926;
			JCudaTensor x2927, x2928;
			x2927 = x2888;
			x2928 = x250;
			x2926 = x235.backward(x2927, x2928);

			// Dealloc(X8663)
			JCudaTensor x2929;
			x2929 = x250;
			x2929.free();

			// val X4224 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_X8661
			JCudaTensor x2930;
			JCudaTensor x2931, x2932, x2933;
			x2931 = x2926;
			x2932 = x236;
			x2933 = x247;
			JCudaTensor[] x2934 = x249.backward(x2931,x2932,x2933);
			x2930 = x2934[0];

			// val X8493 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_3a2_b_bn_scale
			JCudaTensor x2935;
			x2935 = x2934[1];

			// val X8492 = X4223 * d_BatchNorm(3a2_b_bn)(X8661,3a2_b_bn_scale)/d_3a2_b_bn_bias
			JCudaTensor x2939;
			x2939 = x2934[2];

			// Dealloc(X8661)
			JCudaTensor x2943;
			x2943 = x236;
			x2943.free();

			// V_3a2_b_bn_bias <~~ X8492
			float x2945, x2946;
			x2945 = lrn_rate_1;
			x2946 = momentum;
			JCudaTensor x2947;
			x2947 = x2939;
			x2944.update(x2947, x2945, x2946);

			// Dealloc(X8492)
			JCudaTensor x2948;
			x2948 = x2939;
			x2948.free();

			// V_3a2_b_bn_scale <~~ X8493
			float x2950, x2951;
			x2950 = lrn_rate_1;
			x2951 = momentum;
			JCudaTensor x2952;
			x2952 = x2935;
			x2949.update(x2952, x2950, x2951);

			// Dealloc(X8493)
			JCudaTensor x2953;
			x2953 = x2935;
			x2953.free();

			// val X4225 = X4224 * d_Convolv(1,1)(3a2_b_cv_W)/d_X8660
			JCudaTensor x2954;
			JCudaTensor x2955, x2956;
			x2955 = x2930;
			x2956 = x240;
			x2954 = x242.backward_data(x2955, x2956);

			// V_3a2_b_cv_W <~~ X4224 * d_Convolv(1,1)(X8660)/d_3a2_b_cv_W
			float x2958, x2959;
			x2958 = lrn_rate_1;
			x2959 = momentum;
			JCudaTensor x2960, x2961;
			x2960 = x2930;
			x2961 = x233;
			x242.backward_filter(x2960, x2961, x2957, x2958, x2959);

			// Dealloc(X4224)
			JCudaTensor x2962;
			x2962 = x2930;
			x2962.free();

			// 3a2_b_bn_bias <~~ V_3a2_b_bn_bias
			float x2963, x2964;
			x2963 = 1;
			x2964 = decay_1;
			JCudaTensor x2965;
			x2965 = x2944;
			x248.update(x2965, x2963, x2964);

			// 3a2_b_bn_scale <~~ V_3a2_b_bn_scale
			float x2966, x2967;
			x2966 = 1;
			x2967 = decay_1;
			JCudaTensor x2968;
			x2968 = x2949;
			x247.update(x2968, x2966, x2967);

			// 3a2_b_cv_W <~~ V_3a2_b_cv_W
			float x2969, x2970;
			x2969 = 1;
			x2970 = decay_1;
			JCudaTensor x2971;
			x2971 = x2957;
			x240.update(x2971, x2969, x2970);

			// val X4229 = X4225 * d_ReLU()(X8660)/d_X8659
			JCudaTensor x2972;
			JCudaTensor x2973, x2974;
			x2973 = x2954;
			x2974 = x233;
			x2972 = x235.backward(x2973, x2974);

			// Dealloc(X8660)
			JCudaTensor x2975;
			x2975 = x233;
			x2975.free();

			// val X8489 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_3a2_a_bn_bias
			JCudaTensor x2976;
			JCudaTensor x2977, x2978, x2979;
			x2977 = x2972;
			x2978 = x205;
			x2979 = x230;
			JCudaTensor[] x2980 = x232.backward(x2977,x2978,x2979);
			x2976 = x2980[2];

			// val X4230 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_X8658
			JCudaTensor x2981;
			x2981 = x2980[0];

			// val X8490 = X4229 * d_BatchNorm(3a2_a_bn)(X8658,3a2_a_bn_scale)/d_3a2_a_bn_scale
			JCudaTensor x2985;
			x2985 = x2980[1];

			// Dealloc(X8658)
			JCudaTensor x2989;
			x2989 = x205;
			x2989.free();

			// val X4232 = (X4207 + X4230 * d_Convolv(2,0)(3a2_a_cv_W)/d_X8654)
			JCudaTensor x2990;
			JCudaTensor x2991;
			x2991 = x2875;
			JCudaTensor x2992, x2993;
			x2992 = x2981;
			x2993 = x209;
			x2990 = x211.backward_data(x2992,x2993, x2991);

			// V_3a2_a_cv_W <~~ X4230 * d_Convolv(2,0)(X8654)/d_3a2_a_cv_W
			float x2995, x2996;
			x2995 = lrn_rate_1;
			x2996 = momentum;
			JCudaTensor x2997, x2998;
			x2997 = x2981;
			x2998 = x203;
			x211.backward_filter(x2997, x2998, x2994, x2995, x2996);

			// Dealloc(X4230)
			JCudaTensor x2999;
			x2999 = x2981;
			x2999.free();

			// V_3a2_a_bn_scale <~~ X8490
			float x3001, x3002;
			x3001 = lrn_rate_1;
			x3002 = momentum;
			JCudaTensor x3003;
			x3003 = x2985;
			x3000.update(x3003, x3001, x3002);

			// Dealloc(X8490)
			JCudaTensor x3004;
			x3004 = x2985;
			x3004.free();

			// V_3a2_a_bn_bias <~~ X8489
			float x3006, x3007;
			x3006 = lrn_rate_1;
			x3007 = momentum;
			JCudaTensor x3008;
			x3008 = x2976;
			x3005.update(x3008, x3006, x3007);

			// Dealloc(X8489)
			JCudaTensor x3009;
			x3009 = x2976;
			x3009.free();

			// 3a2_a_cv_W <~~ V_3a2_a_cv_W
			float x3010, x3011;
			x3010 = 1;
			x3011 = decay_1;
			JCudaTensor x3012;
			x3012 = x2994;
			x209.update(x3012, x3010, x3011);

			// 3a2_a_bn_scale <~~ V_3a2_a_bn_scale
			float x3013, x3014;
			x3013 = 1;
			x3014 = decay_1;
			JCudaTensor x3015;
			x3015 = x3000;
			x230.update(x3015, x3013, x3014);

			// 3a2_a_bn_bias <~~ V_3a2_a_bn_bias
			float x3016, x3017;
			x3016 = 1;
			x3017 = decay_1;
			JCudaTensor x3018;
			x3018 = x3005;
			x231.update(x3018, x3016, x3017);

			// val X4269 = X4232 * d_ReLU()(X8654)/d_X8653
			JCudaTensor x3019;
			JCudaTensor x3020, x3021;
			x3020 = x2990;
			x3021 = x203;
			x3019 = x96.backward(x3020, x3021);

			// Dealloc(X8654)
			JCudaTensor x3022;
			x3022 = x203;
			x3022.free();

			// val X4279 = X4269.copy * d_ReLU()(X8652)/d_X8651
			JCudaTensor x3023;
			JCudaTensor x3024, x3025;
			x3024 = x3019;
			x3024 = x3024.clone();
			x3025 = x198;
			x3023 = x96.backward(x3024, x3025);

			// Dealloc(X8652)
			JCudaTensor x3026;
			x3026 = x198;
			x3026.free();

			// val X8483 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_2c_c_bn_bias
			JCudaTensor x3027;
			JCudaTensor x3028, x3029, x3030;
			x3028 = x3023;
			x3029 = x185;
			x3030 = x195;
			JCudaTensor[] x3031 = x197.backward(x3028,x3029,x3030);
			x3027 = x3031[2];

			// val X8484 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_2c_c_bn_scale
			JCudaTensor x3032;
			x3032 = x3031[1];

			// val X4280 = X4279 * d_BatchNorm(2c_c_bn)(X8650,2c_c_bn_scale)/d_X8650
			JCudaTensor x3036;
			x3036 = x3031[0];

			// Dealloc(X8650)
			JCudaTensor x3040;
			x3040 = x185;
			x3040.free();

			// val X4281 = X4280 * d_Convolv(1,0)(2c_c_cv_W)/d_X8649
			JCudaTensor x3041;
			JCudaTensor x3042, x3043;
			x3042 = x3036;
			x3043 = x189;
			x3041 = x47.backward_data(x3042, x3043);

			// V_2c_c_bn_bias <~~ X8483
			float x3045, x3046;
			x3045 = lrn_rate_1;
			x3046 = momentum;
			JCudaTensor x3047;
			x3047 = x3027;
			x3044.update(x3047, x3045, x3046);

			// Dealloc(X8483)
			JCudaTensor x3048;
			x3048 = x3027;
			x3048.free();

			// V_2c_c_bn_scale <~~ X8484
			float x3050, x3051;
			x3050 = lrn_rate_1;
			x3051 = momentum;
			JCudaTensor x3052;
			x3052 = x3032;
			x3049.update(x3052, x3050, x3051);

			// Dealloc(X8484)
			JCudaTensor x3053;
			x3053 = x3032;
			x3053.free();

			// V_2c_c_cv_W <~~ X4280 * d_Convolv(1,0)(X8649)/d_2c_c_cv_W
			float x3055, x3056;
			x3055 = lrn_rate_1;
			x3056 = momentum;
			JCudaTensor x3057, x3058;
			x3057 = x3036;
			x3058 = x183;
			x47.backward_filter(x3057, x3058, x3054, x3055, x3056);

			// Dealloc(X4280)
			JCudaTensor x3059;
			x3059 = x3036;
			x3059.free();

			// 2c_c_bn_bias <~~ V_2c_c_bn_bias
			float x3060, x3061;
			x3060 = 1;
			x3061 = decay_1;
			JCudaTensor x3062;
			x3062 = x3044;
			x196.update(x3062, x3060, x3061);

			// 2c_c_bn_scale <~~ V_2c_c_bn_scale
			float x3063, x3064;
			x3063 = 1;
			x3064 = decay_1;
			JCudaTensor x3065;
			x3065 = x3049;
			x195.update(x3065, x3063, x3064);

			// 2c_c_cv_W <~~ V_2c_c_cv_W
			float x3066, x3067;
			x3066 = 1;
			x3067 = decay_1;
			JCudaTensor x3068;
			x3068 = x3054;
			x189.update(x3068, x3066, x3067);

			// val X4285 = X4281 * d_ReLU()(X8649)/d_X8648
			JCudaTensor x3069;
			JCudaTensor x3070, x3071;
			x3070 = x3041;
			x3071 = x183;
			x3069 = x64.backward(x3070, x3071);

			// Dealloc(X8649)
			JCudaTensor x3072;
			x3072 = x183;
			x3072.free();

			// val X4286 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_X8647
			JCudaTensor x3073;
			JCudaTensor x3074, x3075, x3076;
			x3074 = x3069;
			x3075 = x170;
			x3076 = x180;
			JCudaTensor[] x3077 = x182.backward(x3074,x3075,x3076);
			x3073 = x3077[0];

			// val X8480 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_2c_b_bn_bias
			JCudaTensor x3078;
			x3078 = x3077[2];

			// val X8481 = X4285 * d_BatchNorm(2c_b_bn)(X8647,2c_b_bn_scale)/d_2c_b_bn_scale
			JCudaTensor x3082;
			x3082 = x3077[1];

			// Dealloc(X8647)
			JCudaTensor x3086;
			x3086 = x170;
			x3086.free();

			// val X4287 = X4286 * d_Convolv(1,1)(2c_b_cv_W)/d_X8646
			JCudaTensor x3087;
			JCudaTensor x3088, x3089;
			x3088 = x3073;
			x3089 = x174;
			x3087 = x71.backward_data(x3088, x3089);

			// V_2c_b_cv_W <~~ X4286 * d_Convolv(1,1)(X8646)/d_2c_b_cv_W
			float x3091, x3092;
			x3091 = lrn_rate_1;
			x3092 = momentum;
			JCudaTensor x3093, x3094;
			x3093 = x3073;
			x3094 = x168;
			x71.backward_filter(x3093, x3094, x3090, x3091, x3092);

			// Dealloc(X4286)
			JCudaTensor x3095;
			x3095 = x3073;
			x3095.free();

			// V_2c_b_bn_bias <~~ X8480
			float x3097, x3098;
			x3097 = lrn_rate_1;
			x3098 = momentum;
			JCudaTensor x3099;
			x3099 = x3078;
			x3096.update(x3099, x3097, x3098);

			// Dealloc(X8480)
			JCudaTensor x3100;
			x3100 = x3078;
			x3100.free();

			// V_2c_b_bn_scale <~~ X8481
			float x3102, x3103;
			x3102 = lrn_rate_1;
			x3103 = momentum;
			JCudaTensor x3104;
			x3104 = x3082;
			x3101.update(x3104, x3102, x3103);

			// Dealloc(X8481)
			JCudaTensor x3105;
			x3105 = x3082;
			x3105.free();

			// 2c_b_cv_W <~~ V_2c_b_cv_W
			float x3106, x3107;
			x3106 = 1;
			x3107 = decay_1;
			JCudaTensor x3108;
			x3108 = x3090;
			x174.update(x3108, x3106, x3107);

			// 2c_b_bn_bias <~~ V_2c_b_bn_bias
			float x3109, x3110;
			x3109 = 1;
			x3110 = decay_1;
			JCudaTensor x3111;
			x3111 = x3096;
			x181.update(x3111, x3109, x3110);

			// 2c_b_bn_scale <~~ V_2c_b_bn_scale
			float x3112, x3113;
			x3112 = 1;
			x3113 = decay_1;
			JCudaTensor x3114;
			x3114 = x3101;
			x180.update(x3114, x3112, x3113);

			// val X4291 = X4287 * d_ReLU()(X8646)/d_X8645
			JCudaTensor x3115;
			JCudaTensor x3116, x3117;
			x3116 = x3087;
			x3117 = x168;
			x3115 = x64.backward(x3116, x3117);

			// Dealloc(X8646)
			JCudaTensor x3118;
			x3118 = x168;
			x3118.free();

			// val X8478 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_2c_a_bn_scale
			JCudaTensor x3119;
			JCudaTensor x3120, x3121, x3122;
			x3120 = x3115;
			x3121 = x155;
			x3122 = x165;
			JCudaTensor[] x3123 = x167.backward(x3120,x3121,x3122);
			x3119 = x3123[1];

			// val X8477 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_2c_a_bn_bias
			JCudaTensor x3124;
			x3124 = x3123[2];

			// val X4292 = X4291 * d_BatchNorm(2c_a_bn)(X8644,2c_a_bn_scale)/d_X8644
			JCudaTensor x3128;
			x3128 = x3123[0];

			// Dealloc(X8644)
			JCudaTensor x3132;
			x3132 = x155;
			x3132.free();

			// val X4293 = X4292 * d_Convolv(1,0)(2c_a_cv_W)/d_X8643
			JCudaTensor x3133;
			JCudaTensor x3134, x3135;
			x3134 = x3128;
			x3135 = x159;
			x3133 = x110.backward_data(x3134, x3135);

			// V_2c_a_bn_scale <~~ X8478
			float x3137, x3138;
			x3137 = lrn_rate_1;
			x3138 = momentum;
			JCudaTensor x3139;
			x3139 = x3119;
			x3136.update(x3139, x3137, x3138);

			// Dealloc(X8478)
			JCudaTensor x3140;
			x3140 = x3119;
			x3140.free();

			// V_2c_a_cv_W <~~ X4292 * d_Convolv(1,0)(X8643)/d_2c_a_cv_W
			float x3142, x3143;
			x3142 = lrn_rate_1;
			x3143 = momentum;
			JCudaTensor x3144, x3145;
			x3144 = x3128;
			x3145 = x153;
			x110.backward_filter(x3144, x3145, x3141, x3142, x3143);

			// Dealloc(X4292)
			JCudaTensor x3146;
			x3146 = x3128;
			x3146.free();

			// V_2c_a_bn_bias <~~ X8477
			float x3148, x3149;
			x3148 = lrn_rate_1;
			x3149 = momentum;
			JCudaTensor x3150;
			x3150 = x3124;
			x3147.update(x3150, x3148, x3149);

			// Dealloc(X8477)
			JCudaTensor x3151;
			x3151 = x3124;
			x3151.free();

			// 2c_a_bn_scale <~~ V_2c_a_bn_scale
			float x3152, x3153;
			x3152 = 1;
			x3153 = decay_1;
			JCudaTensor x3154;
			x3154 = x3136;
			x165.update(x3154, x3152, x3153);

			// 2c_a_cv_W <~~ V_2c_a_cv_W
			float x3155, x3156;
			x3155 = 1;
			x3156 = decay_1;
			JCudaTensor x3157;
			x3157 = x3141;
			x159.update(x3157, x3155, x3156);

			// 2c_a_bn_bias <~~ V_2c_a_bn_bias
			float x3158, x3159;
			x3158 = 1;
			x3159 = decay_1;
			JCudaTensor x3160;
			x3160 = x3147;
			x166.update(x3160, x3158, x3159);

			// val X4294 = (X4293 + X4269)
			JCudaTensor x3161;
			JCudaTensor x3162, x3163;
			x3162 = x3133;
			x3163 = x3019;
			x3161 = x3162.plus_i(x3163);

			// Dealloc(X4269)
			JCudaTensor x3164;
			x3164 = x3019;
			x3164.free();

			// val X4306 = X4294 * d_ReLU()(X8643)/d_X8642
			JCudaTensor x3165;
			JCudaTensor x3166, x3167;
			x3166 = x3161;
			x3167 = x153;
			x3165 = x96.backward(x3166, x3167);

			// Dealloc(X8643)
			JCudaTensor x3168;
			x3168 = x153;
			x3168.free();

			// val X4316 = X4306.copy * d_ReLU()(X8641)/d_X8640
			JCudaTensor x3169;
			JCudaTensor x3170, x3171;
			x3170 = x3165;
			x3170 = x3170.clone();
			x3171 = x148;
			x3169 = x96.backward(x3170, x3171);

			// Dealloc(X8641)
			JCudaTensor x3172;
			x3172 = x148;
			x3172.free();

			// val X4317 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_X8639
			JCudaTensor x3173;
			JCudaTensor x3174, x3175, x3176;
			x3174 = x3169;
			x3175 = x135;
			x3176 = x145;
			JCudaTensor[] x3177 = x147.backward(x3174,x3175,x3176);
			x3173 = x3177[0];

			// val X8474 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_2b_c_bn_bias
			JCudaTensor x3178;
			x3178 = x3177[2];

			// val X8475 = X4316 * d_BatchNorm(2b_c_bn)(X8639,2b_c_bn_scale)/d_2b_c_bn_scale
			JCudaTensor x3182;
			x3182 = x3177[1];

			// Dealloc(X8639)
			JCudaTensor x3186;
			x3186 = x135;
			x3186.free();

			// V_2b_c_bn_scale <~~ X8475
			float x3188, x3189;
			x3188 = lrn_rate_1;
			x3189 = momentum;
			JCudaTensor x3190;
			x3190 = x3182;
			x3187.update(x3190, x3188, x3189);

			// Dealloc(X8475)
			JCudaTensor x3191;
			x3191 = x3182;
			x3191.free();

			// val X4318 = X4317 * d_Convolv(1,0)(2b_c_cv_W)/d_X8638
			JCudaTensor x3192;
			JCudaTensor x3193, x3194;
			x3193 = x3173;
			x3194 = x139;
			x3192 = x47.backward_data(x3193, x3194);

			// V_2b_c_bn_bias <~~ X8474
			float x3196, x3197;
			x3196 = lrn_rate_1;
			x3197 = momentum;
			JCudaTensor x3198;
			x3198 = x3178;
			x3195.update(x3198, x3196, x3197);

			// Dealloc(X8474)
			JCudaTensor x3199;
			x3199 = x3178;
			x3199.free();

			// V_2b_c_cv_W <~~ X4317 * d_Convolv(1,0)(X8638)/d_2b_c_cv_W
			float x3201, x3202;
			x3201 = lrn_rate_1;
			x3202 = momentum;
			JCudaTensor x3203, x3204;
			x3203 = x3173;
			x3204 = x133;
			x47.backward_filter(x3203, x3204, x3200, x3201, x3202);

			// Dealloc(X4317)
			JCudaTensor x3205;
			x3205 = x3173;
			x3205.free();

			// 2b_c_bn_scale <~~ V_2b_c_bn_scale
			float x3206, x3207;
			x3206 = 1;
			x3207 = decay_1;
			JCudaTensor x3208;
			x3208 = x3187;
			x145.update(x3208, x3206, x3207);

			// 2b_c_bn_bias <~~ V_2b_c_bn_bias
			float x3209, x3210;
			x3209 = 1;
			x3210 = decay_1;
			JCudaTensor x3211;
			x3211 = x3195;
			x146.update(x3211, x3209, x3210);

			// 2b_c_cv_W <~~ V_2b_c_cv_W
			float x3212, x3213;
			x3212 = 1;
			x3213 = decay_1;
			JCudaTensor x3214;
			x3214 = x3200;
			x139.update(x3214, x3212, x3213);

			// val X4322 = X4318 * d_ReLU()(X8638)/d_X8637
			JCudaTensor x3215;
			JCudaTensor x3216, x3217;
			x3216 = x3192;
			x3217 = x133;
			x3215 = x64.backward(x3216, x3217);

			// Dealloc(X8638)
			JCudaTensor x3218;
			x3218 = x133;
			x3218.free();

			// val X4323 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_X8636
			JCudaTensor x3219;
			JCudaTensor x3220, x3221, x3222;
			x3220 = x3215;
			x3221 = x120;
			x3222 = x130;
			JCudaTensor[] x3223 = x132.backward(x3220,x3221,x3222);
			x3219 = x3223[0];

			// val X8472 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_2b_b_bn_scale
			JCudaTensor x3224;
			x3224 = x3223[1];

			// val X8471 = X4322 * d_BatchNorm(2b_b_bn)(X8636,2b_b_bn_scale)/d_2b_b_bn_bias
			JCudaTensor x3228;
			x3228 = x3223[2];

			// Dealloc(X8636)
			JCudaTensor x3232;
			x3232 = x120;
			x3232.free();

			// V_2b_b_bn_scale <~~ X8472
			float x3234, x3235;
			x3234 = lrn_rate_1;
			x3235 = momentum;
			JCudaTensor x3236;
			x3236 = x3224;
			x3233.update(x3236, x3234, x3235);

			// Dealloc(X8472)
			JCudaTensor x3237;
			x3237 = x3224;
			x3237.free();

			// val X4324 = X4323 * d_Convolv(1,1)(2b_b_cv_W)/d_X8635
			JCudaTensor x3238;
			JCudaTensor x3239, x3240;
			x3239 = x3219;
			x3240 = x124;
			x3238 = x71.backward_data(x3239, x3240);

			// V_2b_b_bn_bias <~~ X8471
			float x3242, x3243;
			x3242 = lrn_rate_1;
			x3243 = momentum;
			JCudaTensor x3244;
			x3244 = x3228;
			x3241.update(x3244, x3242, x3243);

			// Dealloc(X8471)
			JCudaTensor x3245;
			x3245 = x3228;
			x3245.free();

			// V_2b_b_cv_W <~~ X4323 * d_Convolv(1,1)(X8635)/d_2b_b_cv_W
			float x3247, x3248;
			x3247 = lrn_rate_1;
			x3248 = momentum;
			JCudaTensor x3249, x3250;
			x3249 = x3219;
			x3250 = x118;
			x71.backward_filter(x3249, x3250, x3246, x3247, x3248);

			// Dealloc(X4323)
			JCudaTensor x3251;
			x3251 = x3219;
			x3251.free();

			// 2b_b_bn_scale <~~ V_2b_b_bn_scale
			float x3252, x3253;
			x3252 = 1;
			x3253 = decay_1;
			JCudaTensor x3254;
			x3254 = x3233;
			x130.update(x3254, x3252, x3253);

			// 2b_b_bn_bias <~~ V_2b_b_bn_bias
			float x3255, x3256;
			x3255 = 1;
			x3256 = decay_1;
			JCudaTensor x3257;
			x3257 = x3241;
			x131.update(x3257, x3255, x3256);

			// 2b_b_cv_W <~~ V_2b_b_cv_W
			float x3258, x3259;
			x3258 = 1;
			x3259 = decay_1;
			JCudaTensor x3260;
			x3260 = x3246;
			x124.update(x3260, x3258, x3259);

			// val X4328 = X4324 * d_ReLU()(X8635)/d_X8634
			JCudaTensor x3261;
			JCudaTensor x3262, x3263;
			x3262 = x3238;
			x3263 = x118;
			x3261 = x64.backward(x3262, x3263);

			// Dealloc(X8635)
			JCudaTensor x3264;
			x3264 = x118;
			x3264.free();

			// val X8469 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_2b_a_bn_scale
			JCudaTensor x3265;
			JCudaTensor x3266, x3267, x3268;
			x3266 = x3261;
			x3267 = x104;
			x3268 = x115;
			JCudaTensor[] x3269 = x117.backward(x3266,x3267,x3268);
			x3265 = x3269[1];

			// val X8468 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_2b_a_bn_bias
			JCudaTensor x3270;
			x3270 = x3269[2];

			// val X4329 = X4328 * d_BatchNorm(2b_a_bn)(X8633,2b_a_bn_scale)/d_X8633
			JCudaTensor x3274;
			x3274 = x3269[0];

			// Dealloc(X8633)
			JCudaTensor x3278;
			x3278 = x104;
			x3278.free();

			// V_2b_a_bn_scale <~~ X8469
			float x3280, x3281;
			x3280 = lrn_rate_1;
			x3281 = momentum;
			JCudaTensor x3282;
			x3282 = x3265;
			x3279.update(x3282, x3280, x3281);

			// Dealloc(X8469)
			JCudaTensor x3283;
			x3283 = x3265;
			x3283.free();

			// val X4330 = X4329 * d_Convolv(1,0)(2b_a_cv_W)/d_X8632
			JCudaTensor x3284;
			JCudaTensor x3285, x3286;
			x3285 = x3274;
			x3286 = x108;
			x3284 = x110.backward_data(x3285, x3286);

			// V_2b_a_cv_W <~~ X4329 * d_Convolv(1,0)(X8632)/d_2b_a_cv_W
			float x3288, x3289;
			x3288 = lrn_rate_1;
			x3289 = momentum;
			JCudaTensor x3290, x3291;
			x3290 = x3274;
			x3291 = x102;
			x110.backward_filter(x3290, x3291, x3287, x3288, x3289);

			// Dealloc(X4329)
			JCudaTensor x3292;
			x3292 = x3274;
			x3292.free();

			// V_2b_a_bn_bias <~~ X8468
			float x3294, x3295;
			x3294 = lrn_rate_1;
			x3295 = momentum;
			JCudaTensor x3296;
			x3296 = x3270;
			x3293.update(x3296, x3294, x3295);

			// Dealloc(X8468)
			JCudaTensor x3297;
			x3297 = x3270;
			x3297.free();

			// 2b_a_bn_scale <~~ V_2b_a_bn_scale
			float x3298, x3299;
			x3298 = 1;
			x3299 = decay_1;
			JCudaTensor x3300;
			x3300 = x3279;
			x115.update(x3300, x3298, x3299);

			// 2b_a_cv_W <~~ V_2b_a_cv_W
			float x3301, x3302;
			x3301 = 1;
			x3302 = decay_1;
			JCudaTensor x3303;
			x3303 = x3287;
			x108.update(x3303, x3301, x3302);

			// 2b_a_bn_bias <~~ V_2b_a_bn_bias
			float x3304, x3305;
			x3304 = 1;
			x3305 = decay_1;
			JCudaTensor x3306;
			x3306 = x3293;
			x116.update(x3306, x3304, x3305);

			// val X4331 = (X4330 + X4306)
			JCudaTensor x3307;
			JCudaTensor x3308, x3309;
			x3308 = x3284;
			x3309 = x3165;
			x3307 = x3308.plus_i(x3309);

			// Dealloc(X4306)
			JCudaTensor x3310;
			x3310 = x3165;
			x3310.free();

			// val X4346 = X4331 * d_ReLU()(X8632)/d_X8631
			JCudaTensor x3311;
			JCudaTensor x3312, x3313;
			x3312 = x3307;
			x3313 = x102;
			x3311 = x96.backward(x3312, x3313);

			// Dealloc(X8632)
			JCudaTensor x3314;
			x3314 = x102;
			x3314.free();

			// val X4362 = X4346.copy * d_ReLU()(X8630)/d_X8629
			JCudaTensor x3315;
			JCudaTensor x3316, x3317;
			x3316 = x3311;
			x3316 = x3316.clone();
			x3317 = x97;
			x3315 = x96.backward(x3316, x3317);

			// Dealloc(X8630)
			JCudaTensor x3318;
			x3318 = x97;
			x3318.free();

			// val X4350 = X4346.copy * d_ReLU()(X8621)/d_X8620
			JCudaTensor x3319;
			JCudaTensor x3320, x3321;
			x3320 = x3311;
			x3320 = x3320.clone();
			x3321 = x94;
			x3319 = x96.backward(x3320, x3321);

			// Dealloc(X4346)
			JCudaTensor x3322;
			x3322 = x3311;
			x3322.free();

			// Dealloc(X8621)
			JCudaTensor x3323;
			x3323 = x94;
			x3323.free();

			// val X8465 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_2a2_c_bn_bias
			JCudaTensor x3324;
			JCudaTensor x3325, x3326, x3327;
			x3325 = x3315;
			x3326 = x81;
			x3327 = x91;
			JCudaTensor[] x3328 = x93.backward(x3325,x3326,x3327);
			x3324 = x3328[2];

			// val X4351 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_X8619
			JCudaTensor x3329;
			JCudaTensor x3330, x3331, x3332;
			x3330 = x3319;
			x3331 = x41;
			x3332 = x59;
			JCudaTensor[] x3333 = x61.backward(x3330,x3331,x3332);
			x3329 = x3333[0];

			// val X4363 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_X8628
			JCudaTensor x3334;
			x3334 = x3328[0];

			// val X8456 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_2a1_bn_bias
			JCudaTensor x3338;
			x3338 = x3333[2];

			// val X8466 = X4362 * d_BatchNorm(2a2_c_bn)(X8628,2a2_c_bn_scale)/d_2a2_c_bn_scale
			JCudaTensor x3342;
			x3342 = x3328[1];

			// Dealloc(X8628)
			JCudaTensor x3346;
			x3346 = x81;
			x3346.free();

			// val X8457 = X4350 * d_BatchNorm(2a1_bn)(X8619,2a1_bn_scale)/d_2a1_bn_scale
			JCudaTensor x3347;
			x3347 = x3333[1];

			// Dealloc(X8619)
			JCudaTensor x3351;
			x3351 = x41;
			x3351.free();

			// val X4364 = X4363 * d_Convolv(1,0)(2a2_c_cv_W)/d_X8627
			JCudaTensor x3352;
			JCudaTensor x3353, x3354;
			x3353 = x3334;
			x3354 = x85;
			x3352 = x47.backward_data(x3353, x3354);

			// V_2a1_bn_scale <~~ X8457
			float x3356, x3357;
			x3356 = lrn_rate_1;
			x3357 = momentum;
			JCudaTensor x3358;
			x3358 = x3347;
			x3355.update(x3358, x3356, x3357);

			// Dealloc(X8457)
			JCudaTensor x3359;
			x3359 = x3347;
			x3359.free();

			// V_2a1_cv_W <~~ X4351 * d_Convolv(1,0)(X8618)/d_2a1_cv_W
			float x3361, x3362;
			x3361 = lrn_rate_1;
			x3362 = momentum;
			JCudaTensor x3363, x3364;
			x3363 = x3329;
			x3364 = x31;
			x47.backward_filter(x3363, x3364, x3360, x3361, x3362);

			// V_2a2_c_bn_bias <~~ X8465
			float x3366, x3367;
			x3366 = lrn_rate_1;
			x3367 = momentum;
			JCudaTensor x3368;
			x3368 = x3324;
			x3365.update(x3368, x3366, x3367);

			// Dealloc(X8465)
			JCudaTensor x3369;
			x3369 = x3324;
			x3369.free();

			// val X4352 = X4351 * d_Convolv(1,0)(2a1_cv_W)/d_X8618
			JCudaTensor x3370;
			JCudaTensor x3371, x3372;
			x3371 = x3329;
			x3372 = x45;
			x3370 = x47.backward_data(x3371, x3372);

			// Dealloc(X4351)
			JCudaTensor x3373;
			x3373 = x3329;
			x3373.free();

			// V_2a2_c_cv_W <~~ X4363 * d_Convolv(1,0)(X8627)/d_2a2_c_cv_W
			float x3375, x3376;
			x3375 = lrn_rate_1;
			x3376 = momentum;
			JCudaTensor x3377, x3378;
			x3377 = x3334;
			x3378 = x79;
			x47.backward_filter(x3377, x3378, x3374, x3375, x3376);

			// Dealloc(X4363)
			JCudaTensor x3379;
			x3379 = x3334;
			x3379.free();

			// V_2a2_c_bn_scale <~~ X8466
			float x3381, x3382;
			x3381 = lrn_rate_1;
			x3382 = momentum;
			JCudaTensor x3383;
			x3383 = x3342;
			x3380.update(x3383, x3381, x3382);

			// Dealloc(X8466)
			JCudaTensor x3384;
			x3384 = x3342;
			x3384.free();

			// V_2a1_bn_bias <~~ X8456
			float x3386, x3387;
			x3386 = lrn_rate_1;
			x3387 = momentum;
			JCudaTensor x3388;
			x3388 = x3338;
			x3385.update(x3388, x3386, x3387);

			// Dealloc(X8456)
			JCudaTensor x3389;
			x3389 = x3338;
			x3389.free();

			// 2a1_bn_scale <~~ V_2a1_bn_scale
			float x3390, x3391;
			x3390 = 1;
			x3391 = decay_1;
			JCudaTensor x3392;
			x3392 = x3355;
			x59.update(x3392, x3390, x3391);

			// 2a2_c_cv_W <~~ V_2a2_c_cv_W
			float x3393, x3394;
			x3393 = 1;
			x3394 = decay_1;
			JCudaTensor x3395;
			x3395 = x3374;
			x85.update(x3395, x3393, x3394);

			// 2a2_c_bn_bias <~~ V_2a2_c_bn_bias
			float x3396, x3397;
			x3396 = 1;
			x3397 = decay_1;
			JCudaTensor x3398;
			x3398 = x3365;
			x92.update(x3398, x3396, x3397);

			// 2a1_bn_bias <~~ V_2a1_bn_bias
			float x3399, x3400;
			x3399 = 1;
			x3400 = decay_1;
			JCudaTensor x3401;
			x3401 = x3385;
			x60.update(x3401, x3399, x3400);

			// 2a2_c_bn_scale <~~ V_2a2_c_bn_scale
			float x3402, x3403;
			x3402 = 1;
			x3403 = decay_1;
			JCudaTensor x3404;
			x3404 = x3380;
			x91.update(x3404, x3402, x3403);

			// 2a1_cv_W <~~ V_2a1_cv_W
			float x3405, x3406;
			x3405 = 1;
			x3406 = decay_1;
			JCudaTensor x3407;
			x3407 = x3360;
			x45.update(x3407, x3405, x3406);

			// val X4368 = X4364 * d_ReLU()(X8627)/d_X8626
			JCudaTensor x3408;
			JCudaTensor x3409, x3410;
			x3409 = x3352;
			x3410 = x79;
			x3408 = x64.backward(x3409, x3410);

			// Dealloc(X8627)
			JCudaTensor x3411;
			x3411 = x79;
			x3411.free();

			// val X8462 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_2a2_b_bn_bias
			JCudaTensor x3412;
			JCudaTensor x3413, x3414, x3415;
			x3413 = x3408;
			x3414 = x65;
			x3415 = x76;
			JCudaTensor[] x3416 = x78.backward(x3413,x3414,x3415);
			x3412 = x3416[2];

			// val X4369 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_X8625
			JCudaTensor x3417;
			x3417 = x3416[0];

			// val X8463 = X4368 * d_BatchNorm(2a2_b_bn)(X8625,2a2_b_bn_scale)/d_2a2_b_bn_scale
			JCudaTensor x3421;
			x3421 = x3416[1];

			// Dealloc(X8625)
			JCudaTensor x3425;
			x3425 = x65;
			x3425.free();

			// V_2a2_b_bn_bias <~~ X8462
			float x3427, x3428;
			x3427 = lrn_rate_1;
			x3428 = momentum;
			JCudaTensor x3429;
			x3429 = x3412;
			x3426.update(x3429, x3427, x3428);

			// Dealloc(X8462)
			JCudaTensor x3430;
			x3430 = x3412;
			x3430.free();

			// val X4370 = X4369 * d_Convolv(1,1)(2a2_b_cv_W)/d_X8624
			JCudaTensor x3431;
			JCudaTensor x3432, x3433;
			x3432 = x3417;
			x3433 = x69;
			x3431 = x71.backward_data(x3432, x3433);

			// V_2a2_b_cv_W <~~ X4369 * d_Convolv(1,1)(X8624)/d_2a2_b_cv_W
			float x3435, x3436;
			x3435 = lrn_rate_1;
			x3436 = momentum;
			JCudaTensor x3437, x3438;
			x3437 = x3417;
			x3438 = x62;
			x71.backward_filter(x3437, x3438, x3434, x3435, x3436);

			// Dealloc(X4369)
			JCudaTensor x3439;
			x3439 = x3417;
			x3439.free();

			// V_2a2_b_bn_scale <~~ X8463
			float x3441, x3442;
			x3441 = lrn_rate_1;
			x3442 = momentum;
			JCudaTensor x3443;
			x3443 = x3421;
			x3440.update(x3443, x3441, x3442);

			// Dealloc(X8463)
			JCudaTensor x3444;
			x3444 = x3421;
			x3444.free();

			// 2a2_b_bn_bias <~~ V_2a2_b_bn_bias
			float x3445, x3446;
			x3445 = 1;
			x3446 = decay_1;
			JCudaTensor x3447;
			x3447 = x3426;
			x77.update(x3447, x3445, x3446);

			// 2a2_b_cv_W <~~ V_2a2_b_cv_W
			float x3448, x3449;
			x3448 = 1;
			x3449 = decay_1;
			JCudaTensor x3450;
			x3450 = x3434;
			x69.update(x3450, x3448, x3449);

			// 2a2_b_bn_scale <~~ V_2a2_b_bn_scale
			float x3451, x3452;
			x3451 = 1;
			x3452 = decay_1;
			JCudaTensor x3453;
			x3453 = x3440;
			x76.update(x3453, x3451, x3452);

			// val X4374 = X4370 * d_ReLU()(X8624)/d_X8623
			JCudaTensor x3454;
			JCudaTensor x3455, x3456;
			x3455 = x3431;
			x3456 = x62;
			x3454 = x64.backward(x3455, x3456);

			// Dealloc(X8624)
			JCudaTensor x3457;
			x3457 = x62;
			x3457.free();

			// val X4375 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_X8622
			JCudaTensor x3458;
			JCudaTensor x3459, x3460, x3461;
			x3459 = x3454;
			x3460 = x34;
			x3461 = x52;
			JCudaTensor[] x3462 = x54.backward(x3459,x3460,x3461);
			x3458 = x3462[0];

			// val X8460 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_2a2_a_bn_scale
			JCudaTensor x3463;
			x3463 = x3462[1];

			// val X8459 = X4374 * d_BatchNorm(2a2_a_bn)(X8622,2a2_a_bn_scale)/d_2a2_a_bn_bias
			JCudaTensor x3467;
			x3467 = x3462[2];

			// Dealloc(X8622)
			JCudaTensor x3471;
			x3471 = x34;
			x3471.free();

			// val X4377 = (X4352 + X4375 * d_Convolv(1,0)(2a2_a_cv_W)/d_X8618)
			JCudaTensor x3472;
			JCudaTensor x3473;
			x3473 = x3370;
			JCudaTensor x3474, x3475;
			x3474 = x3458;
			x3475 = x38;
			x3472 = x40.backward_data(x3474,x3475, x3473);

			// V_2a2_a_cv_W <~~ X4375 * d_Convolv(1,0)(X8618)/d_2a2_a_cv_W
			float x3477, x3478;
			x3477 = lrn_rate_1;
			x3478 = momentum;
			JCudaTensor x3479, x3480;
			x3479 = x3458;
			x3480 = x31;
			x40.backward_filter(x3479, x3480, x3476, x3477, x3478);

			// Dealloc(X4375)
			JCudaTensor x3481;
			x3481 = x3458;
			x3481.free();

			// V_2a2_a_bn_scale <~~ X8460
			float x3483, x3484;
			x3483 = lrn_rate_1;
			x3484 = momentum;
			JCudaTensor x3485;
			x3485 = x3463;
			x3482.update(x3485, x3483, x3484);

			// Dealloc(X8460)
			JCudaTensor x3486;
			x3486 = x3463;
			x3486.free();

			// V_2a2_a_bn_bias <~~ X8459
			float x3488, x3489;
			x3488 = lrn_rate_1;
			x3489 = momentum;
			JCudaTensor x3490;
			x3490 = x3467;
			x3487.update(x3490, x3488, x3489);

			// Dealloc(X8459)
			JCudaTensor x3491;
			x3491 = x3467;
			x3491.free();

			// 2a2_a_cv_W <~~ V_2a2_a_cv_W
			float x3492, x3493;
			x3492 = 1;
			x3493 = decay_1;
			JCudaTensor x3494;
			x3494 = x3476;
			x38.update(x3494, x3492, x3493);

			// 2a2_a_bn_scale <~~ V_2a2_a_bn_scale
			float x3495, x3496;
			x3495 = 1;
			x3496 = decay_1;
			JCudaTensor x3497;
			x3497 = x3482;
			x52.update(x3497, x3495, x3496);

			// 2a2_a_bn_bias <~~ V_2a2_a_bn_bias
			float x3498, x3499;
			x3498 = 1;
			x3499 = decay_1;
			JCudaTensor x3500;
			x3500 = x3487;
			x53.update(x3500, x3498, x3499);

			// val X4379 = X4377 * d_Pooling(3,2,0,true)(X8618,X8617)/d_X8617
			JCudaTensor x3501;
			JCudaTensor x3502, x3503, x3504;
			x3502 = x3472;
			x3503 = x31;
			x3504 = x28;
			x3501 = x33.backward(x3502, x3503, x3504);

			// Dealloc(X4377)
			JCudaTensor x3505;
			x3505 = x3472;
			x3505.free();

			// Dealloc(X8618)
			JCudaTensor x3506;
			x3506 = x31;
			x3506.free();

			// val X8448 = X4379 * d_ReLU()(X8617)/d_X8616
			JCudaTensor x3507;
			JCudaTensor x3508, x3509;
			x3508 = x3501;
			x3509 = x28;
			x3507 = x30.backward(x3508, x3509);

			// Dealloc(X8617)
			JCudaTensor x3510;
			x3510 = x28;
			x3510.free();

			// val X8454 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_1_bn_scale
			JCudaTensor x3511;
			JCudaTensor x3512, x3513, x3514;
			x3512 = x3507;
			x3513 = x11;
			x3514 = x25;
			JCudaTensor[] x3515 = x27.backward(x3512,x3513,x3514);
			x3511 = x3515[1];

			// val X8449 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_X8615
			JCudaTensor x3516;
			x3516 = x3515[0];

			// val X8453 = X8448 * d_BatchNorm(1_bn)(X8615,1_bn_scale)/d_1_bn_bias
			JCudaTensor x3520;
			x3520 = x3515[2];

			// Dealloc(X8615)
			JCudaTensor x3524;
			x3524 = x11;
			x3524.free();

			// V_1_bn_bias <~~ X8453
			float x3526, x3527;
			x3526 = lrn_rate_1;
			x3527 = momentum;
			JCudaTensor x3528;
			x3528 = x3520;
			x3525.update(x3528, x3526, x3527);

			// Dealloc(X8453)
			JCudaTensor x3529;
			x3529 = x3520;
			x3529.free();

			// V_1_bn_scale <~~ X8454
			float x3531, x3532;
			x3531 = lrn_rate_1;
			x3532 = momentum;
			JCudaTensor x3533;
			x3533 = x3511;
			x3530.update(x3533, x3531, x3532);

			// Dealloc(X8454)
			JCudaTensor x3534;
			x3534 = x3511;
			x3534.free();

			// V_1_cv_W <~~ X8449 * d_Convolv(2,3)(X8614)/d_1_cv_W
			float x3536, x3537;
			x3536 = lrn_rate_1;
			x3537 = momentum;
			JCudaTensor x3538, x3539;
			x3538 = x3516;
			x3539 = x7;
			x17.backward_filter(x3538, x3539, x3535, x3536, x3537);

			// Dealloc(X8449)
			JCudaTensor x3540;
			x3540 = x3516;
			x3540.free();

			// Dealloc(X8614)
			JCudaTensor x3541;
			x3541 = x7;
			x3541.free();

			// 1_bn_bias <~~ V_1_bn_bias
			float x3542, x3543;
			x3542 = 1;
			x3543 = decay_1;
			JCudaTensor x3544;
			x3544 = x3525;
			x26.update(x3544, x3542, x3543);

			// 1_bn_scale <~~ V_1_bn_scale
			float x3545, x3546;
			x3545 = 1;
			x3546 = decay_1;
			JCudaTensor x3547;
			x3547 = x3530;
			x25.update(x3547, x3545, x3546);

			// 1_cv_W <~~ V_1_cv_W
			float x3548, x3549;
			x3548 = 1;
			x3549 = decay_1;
			JCudaTensor x3550;
			x3550 = x3535;
			x15.update(x3550, x3548, x3549);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X8813 = Cuda(X)
			JCudaTensor x3551;
			JTensorFloat x3552;
			x3552 = x3;
			x3551 = x3552.asJCudaTensor();

			// val X8814 = Convolv(2,3)(X8813,1_cv_W,1_cv_B)
			JCudaTensor x3553;
			JCudaTensor x3554, x3555, x3556;
			x3554 = x3551;
			x3555 = x15;
			x3556 = x16;
			x3553 = x17.forward(x3554, x3555, x3556);

			// Dealloc(X8813)
			JCudaTensor x3557;
			x3557 = x3551;
			x3557.free();

			// val X8815 = BatchNorm(1_bn)(X8814,1_bn_scale,1_bn_bias)
			JCudaTensor x3558;
			JCudaTensor x3559, x3560, x3561;
			x3559 = x3553;
			x3560 = x25;
			x3561 = x26;
			x3558 = x27.forward_inference(x3559, x3560, x3561);

			// Dealloc(X8814)
			JCudaTensor x3562;
			x3562 = x3553;
			x3562.free();

			// val X8816 = ReLU()(X8815)
			JCudaTensor x3563;
			JCudaTensor x3564;
			x3564 = x3558;
			x3563 = x30.forward(x3564);

			// val X8817 = Pooling(3,2,0,true)(X8816)
			JCudaTensor x3565;
			JCudaTensor x3566;
			x3566 = x3563;
			x3565 = x33.forward(x3566);

			// Dealloc(X8816)
			JCudaTensor x3567;
			x3567 = x3563;
			x3567.free();

			// val X8821 = Convolv(1,0)(X8817,2a2_a_cv_W,2a2_a_cv_B)
			JCudaTensor x3568;
			JCudaTensor x3569, x3570, x3571;
			x3569 = x3565;
			x3570 = x38;
			x3571 = x39;
			x3568 = x40.forward(x3569, x3570, x3571);

			// val X8818 = Convolv(1,0)(X8817,2a1_cv_W,2a1_cv_B)
			JCudaTensor x3572;
			JCudaTensor x3573, x3574, x3575;
			x3573 = x3565;
			x3574 = x45;
			x3575 = x46;
			x3572 = x47.forward(x3573, x3574, x3575);

			// Dealloc(X8817)
			JCudaTensor x3576;
			x3576 = x3565;
			x3576.free();

			// val X8822 = BatchNorm(2a2_a_bn)(X8821,2a2_a_bn_scale,2a2_a_bn_bias)
			JCudaTensor x3577;
			JCudaTensor x3578, x3579, x3580;
			x3578 = x3568;
			x3579 = x52;
			x3580 = x53;
			x3577 = x54.forward_inference(x3578, x3579, x3580);

			// Dealloc(X8821)
			JCudaTensor x3581;
			x3581 = x3568;
			x3581.free();

			// val X8819 = BatchNorm(2a1_bn)(X8818,2a1_bn_scale,2a1_bn_bias)
			JCudaTensor x3582;
			JCudaTensor x3583, x3584, x3585;
			x3583 = x3572;
			x3584 = x59;
			x3585 = x60;
			x3582 = x61.forward_inference(x3583, x3584, x3585);

			// Dealloc(X8818)
			JCudaTensor x3586;
			x3586 = x3572;
			x3586.free();

			// val X8823 = ReLU()(X8822)
			JCudaTensor x3587;
			JCudaTensor x3588;
			x3588 = x3577;
			x3587 = x64.forward(x3588);

			// val X8824 = Convolv(1,1)(X8823,2a2_b_cv_W,2a2_b_cv_B)
			JCudaTensor x3589;
			JCudaTensor x3590, x3591, x3592;
			x3590 = x3587;
			x3591 = x69;
			x3592 = x70;
			x3589 = x71.forward(x3590, x3591, x3592);

			// Dealloc(X8823)
			JCudaTensor x3593;
			x3593 = x3587;
			x3593.free();

			// val X8825 = BatchNorm(2a2_b_bn)(X8824,2a2_b_bn_scale,2a2_b_bn_bias)
			JCudaTensor x3594;
			JCudaTensor x3595, x3596, x3597;
			x3595 = x3589;
			x3596 = x76;
			x3597 = x77;
			x3594 = x78.forward_inference(x3595, x3596, x3597);

			// Dealloc(X8824)
			JCudaTensor x3598;
			x3598 = x3589;
			x3598.free();

			// val X8826 = ReLU()(X8825)
			JCudaTensor x3599;
			JCudaTensor x3600;
			x3600 = x3594;
			x3599 = x64.forward(x3600);

			// val X8827 = Convolv(1,0)(X8826,2a2_c_cv_W,2a2_c_cv_B)
			JCudaTensor x3601;
			JCudaTensor x3602, x3603, x3604;
			x3602 = x3599;
			x3603 = x85;
			x3604 = x86;
			x3601 = x47.forward(x3602, x3603, x3604);

			// Dealloc(X8826)
			JCudaTensor x3605;
			x3605 = x3599;
			x3605.free();

			// val X8828 = BatchNorm(2a2_c_bn)(X8827,2a2_c_bn_scale,2a2_c_bn_bias)
			JCudaTensor x3606;
			JCudaTensor x3607, x3608, x3609;
			x3607 = x3601;
			x3608 = x91;
			x3609 = x92;
			x3606 = x93.forward_inference(x3607, x3608, x3609);

			// Dealloc(X8827)
			JCudaTensor x3610;
			x3610 = x3601;
			x3610.free();

			// val X8820 = ReLU()(X8819)
			JCudaTensor x3611;
			JCudaTensor x3612;
			x3612 = x3582;
			x3611 = x96.forward(x3612);

			// val X8829 = ReLU()(X8828)
			JCudaTensor x3613;
			JCudaTensor x3614;
			x3614 = x3606;
			x3613 = x96.forward(x3614);

			// val X8830 = (X8820 + X8829)
			JCudaTensor x3615;
			JCudaTensor x3616, x3617;
			x3616 = x3611;
			x3617 = x3613;
			x3615 = x3616.plus_i(x3617);

			// Dealloc(X8829)
			JCudaTensor x3618;
			x3618 = x3613;
			x3618.free();

			// val X8831 = ReLU()(X8830)
			JCudaTensor x3619;
			JCudaTensor x3620;
			x3620 = x3615;
			x3619 = x96.forward(x3620);

			// val X8832 = Convolv(1,0)(X8831,2b_a_cv_W,2b_a_cv_B)
			JCudaTensor x3621;
			JCudaTensor x3622, x3623, x3624;
			x3622 = x3619;
			x3623 = x108;
			x3624 = x109;
			x3621 = x110.forward(x3622, x3623, x3624);

			// val X8833 = BatchNorm(2b_a_bn)(X8832,2b_a_bn_scale,2b_a_bn_bias)
			JCudaTensor x3625;
			JCudaTensor x3626, x3627, x3628;
			x3626 = x3621;
			x3627 = x115;
			x3628 = x116;
			x3625 = x117.forward_inference(x3626, x3627, x3628);

			// Dealloc(X8832)
			JCudaTensor x3629;
			x3629 = x3621;
			x3629.free();

			// val X8834 = ReLU()(X8833)
			JCudaTensor x3630;
			JCudaTensor x3631;
			x3631 = x3625;
			x3630 = x64.forward(x3631);

			// val X8835 = Convolv(1,1)(X8834,2b_b_cv_W,2b_b_cv_B)
			JCudaTensor x3632;
			JCudaTensor x3633, x3634, x3635;
			x3633 = x3630;
			x3634 = x124;
			x3635 = x125;
			x3632 = x71.forward(x3633, x3634, x3635);

			// Dealloc(X8834)
			JCudaTensor x3636;
			x3636 = x3630;
			x3636.free();

			// val X8836 = BatchNorm(2b_b_bn)(X8835,2b_b_bn_scale,2b_b_bn_bias)
			JCudaTensor x3637;
			JCudaTensor x3638, x3639, x3640;
			x3638 = x3632;
			x3639 = x130;
			x3640 = x131;
			x3637 = x132.forward_inference(x3638, x3639, x3640);

			// Dealloc(X8835)
			JCudaTensor x3641;
			x3641 = x3632;
			x3641.free();

			// val X8837 = ReLU()(X8836)
			JCudaTensor x3642;
			JCudaTensor x3643;
			x3643 = x3637;
			x3642 = x64.forward(x3643);

			// val X8838 = Convolv(1,0)(X8837,2b_c_cv_W,2b_c_cv_B)
			JCudaTensor x3644;
			JCudaTensor x3645, x3646, x3647;
			x3645 = x3642;
			x3646 = x139;
			x3647 = x140;
			x3644 = x47.forward(x3645, x3646, x3647);

			// Dealloc(X8837)
			JCudaTensor x3648;
			x3648 = x3642;
			x3648.free();

			// val X8839 = BatchNorm(2b_c_bn)(X8838,2b_c_bn_scale,2b_c_bn_bias)
			JCudaTensor x3649;
			JCudaTensor x3650, x3651, x3652;
			x3650 = x3644;
			x3651 = x145;
			x3652 = x146;
			x3649 = x147.forward_inference(x3650, x3651, x3652);

			// Dealloc(X8838)
			JCudaTensor x3653;
			x3653 = x3644;
			x3653.free();

			// val X8840 = ReLU()(X8839)
			JCudaTensor x3654;
			JCudaTensor x3655;
			x3655 = x3649;
			x3654 = x96.forward(x3655);

			// val X8841 = (X8840 + X8831)
			JCudaTensor x3656;
			JCudaTensor x3657, x3658;
			x3657 = x3654;
			x3658 = x3619;
			x3656 = x3657.plus_i(x3658);

			// Dealloc(X8831)
			JCudaTensor x3659;
			x3659 = x3619;
			x3659.free();

			// val X8842 = ReLU()(X8841)
			JCudaTensor x3660;
			JCudaTensor x3661;
			x3661 = x3656;
			x3660 = x96.forward(x3661);

			// val X8843 = Convolv(1,0)(X8842,2c_a_cv_W,2c_a_cv_B)
			JCudaTensor x3662;
			JCudaTensor x3663, x3664, x3665;
			x3663 = x3660;
			x3664 = x159;
			x3665 = x160;
			x3662 = x110.forward(x3663, x3664, x3665);

			// val X8844 = BatchNorm(2c_a_bn)(X8843,2c_a_bn_scale,2c_a_bn_bias)
			JCudaTensor x3666;
			JCudaTensor x3667, x3668, x3669;
			x3667 = x3662;
			x3668 = x165;
			x3669 = x166;
			x3666 = x167.forward_inference(x3667, x3668, x3669);

			// Dealloc(X8843)
			JCudaTensor x3670;
			x3670 = x3662;
			x3670.free();

			// val X8845 = ReLU()(X8844)
			JCudaTensor x3671;
			JCudaTensor x3672;
			x3672 = x3666;
			x3671 = x64.forward(x3672);

			// val X8846 = Convolv(1,1)(X8845,2c_b_cv_W,2c_b_cv_B)
			JCudaTensor x3673;
			JCudaTensor x3674, x3675, x3676;
			x3674 = x3671;
			x3675 = x174;
			x3676 = x175;
			x3673 = x71.forward(x3674, x3675, x3676);

			// Dealloc(X8845)
			JCudaTensor x3677;
			x3677 = x3671;
			x3677.free();

			// val X8847 = BatchNorm(2c_b_bn)(X8846,2c_b_bn_scale,2c_b_bn_bias)
			JCudaTensor x3678;
			JCudaTensor x3679, x3680, x3681;
			x3679 = x3673;
			x3680 = x180;
			x3681 = x181;
			x3678 = x182.forward_inference(x3679, x3680, x3681);

			// Dealloc(X8846)
			JCudaTensor x3682;
			x3682 = x3673;
			x3682.free();

			// val X8848 = ReLU()(X8847)
			JCudaTensor x3683;
			JCudaTensor x3684;
			x3684 = x3678;
			x3683 = x64.forward(x3684);

			// val X8849 = Convolv(1,0)(X8848,2c_c_cv_W,2c_c_cv_B)
			JCudaTensor x3685;
			JCudaTensor x3686, x3687, x3688;
			x3686 = x3683;
			x3687 = x189;
			x3688 = x190;
			x3685 = x47.forward(x3686, x3687, x3688);

			// Dealloc(X8848)
			JCudaTensor x3689;
			x3689 = x3683;
			x3689.free();

			// val X8850 = BatchNorm(2c_c_bn)(X8849,2c_c_bn_scale,2c_c_bn_bias)
			JCudaTensor x3690;
			JCudaTensor x3691, x3692, x3693;
			x3691 = x3685;
			x3692 = x195;
			x3693 = x196;
			x3690 = x197.forward_inference(x3691, x3692, x3693);

			// Dealloc(X8849)
			JCudaTensor x3694;
			x3694 = x3685;
			x3694.free();

			// val X8851 = ReLU()(X8850)
			JCudaTensor x3695;
			JCudaTensor x3696;
			x3696 = x3690;
			x3695 = x96.forward(x3696);

			// val X8852 = (X8851 + X8842)
			JCudaTensor x3697;
			JCudaTensor x3698, x3699;
			x3698 = x3695;
			x3699 = x3660;
			x3697 = x3698.plus_i(x3699);

			// Dealloc(X8842)
			JCudaTensor x3700;
			x3700 = x3660;
			x3700.free();

			// val X8853 = ReLU()(X8852)
			JCudaTensor x3701;
			JCudaTensor x3702;
			x3702 = x3697;
			x3701 = x96.forward(x3702);

			// val X8854 = Convolv(2,0)(X8853,3a1_cv_W,3a1_cv_B)
			JCudaTensor x3703;
			JCudaTensor x3704, x3705, x3706;
			x3704 = x3701;
			x3705 = x216;
			x3706 = x217;
			x3703 = x218.forward(x3704, x3705, x3706);

			// val X8857 = Convolv(2,0)(X8853,3a2_a_cv_W,3a2_a_cv_B)
			JCudaTensor x3707;
			JCudaTensor x3708, x3709, x3710;
			x3708 = x3701;
			x3709 = x209;
			x3710 = x210;
			x3707 = x211.forward(x3708, x3709, x3710);

			// Dealloc(X8853)
			JCudaTensor x3711;
			x3711 = x3701;
			x3711.free();

			// val X8858 = BatchNorm(3a2_a_bn)(X8857,3a2_a_bn_scale,3a2_a_bn_bias)
			JCudaTensor x3712;
			JCudaTensor x3713, x3714, x3715;
			x3713 = x3707;
			x3714 = x230;
			x3715 = x231;
			x3712 = x232.forward_inference(x3713, x3714, x3715);

			// Dealloc(X8857)
			JCudaTensor x3716;
			x3716 = x3707;
			x3716.free();

			// val X8855 = BatchNorm(3a1_bn)(X8854,3a1_bn_scale,3a1_bn_bias)
			JCudaTensor x3717;
			JCudaTensor x3718, x3719, x3720;
			x3718 = x3703;
			x3719 = x223;
			x3720 = x224;
			x3717 = x225.forward_inference(x3718, x3719, x3720);

			// Dealloc(X8854)
			JCudaTensor x3721;
			x3721 = x3703;
			x3721.free();

			// val X8859 = ReLU()(X8858)
			JCudaTensor x3722;
			JCudaTensor x3723;
			x3723 = x3712;
			x3722 = x235.forward(x3723);

			// val X8860 = Convolv(1,1)(X8859,3a2_b_cv_W,3a2_b_cv_B)
			JCudaTensor x3724;
			JCudaTensor x3725, x3726, x3727;
			x3725 = x3722;
			x3726 = x240;
			x3727 = x241;
			x3724 = x242.forward(x3725, x3726, x3727);

			// Dealloc(X8859)
			JCudaTensor x3728;
			x3728 = x3722;
			x3728.free();

			// val X8861 = BatchNorm(3a2_b_bn)(X8860,3a2_b_bn_scale,3a2_b_bn_bias)
			JCudaTensor x3729;
			JCudaTensor x3730, x3731, x3732;
			x3730 = x3724;
			x3731 = x247;
			x3732 = x248;
			x3729 = x249.forward_inference(x3730, x3731, x3732);

			// Dealloc(X8860)
			JCudaTensor x3733;
			x3733 = x3724;
			x3733.free();

			// val X8862 = ReLU()(X8861)
			JCudaTensor x3734;
			JCudaTensor x3735;
			x3735 = x3729;
			x3734 = x235.forward(x3735);

			// val X8863 = Convolv(1,0)(X8862,3a2_c_cv_W,3a2_c_cv_B)
			JCudaTensor x3736;
			JCudaTensor x3737, x3738, x3739;
			x3737 = x3734;
			x3738 = x256;
			x3739 = x257;
			x3736 = x258.forward(x3737, x3738, x3739);

			// Dealloc(X8862)
			JCudaTensor x3740;
			x3740 = x3734;
			x3740.free();

			// val X8864 = BatchNorm(3a2_c_bn)(X8863,3a2_c_bn_scale,3a2_c_bn_bias)
			JCudaTensor x3741;
			JCudaTensor x3742, x3743, x3744;
			x3742 = x3736;
			x3743 = x263;
			x3744 = x264;
			x3741 = x265.forward_inference(x3742, x3743, x3744);

			// Dealloc(X8863)
			JCudaTensor x3745;
			x3745 = x3736;
			x3745.free();

			// val X8856 = ReLU()(X8855)
			JCudaTensor x3746;
			JCudaTensor x3747;
			x3747 = x3717;
			x3746 = x268.forward(x3747);

			// val X8865 = ReLU()(X8864)
			JCudaTensor x3748;
			JCudaTensor x3749;
			x3749 = x3741;
			x3748 = x268.forward(x3749);

			// val X8866 = (X8856 + X8865)
			JCudaTensor x3750;
			JCudaTensor x3751, x3752;
			x3751 = x3746;
			x3752 = x3748;
			x3750 = x3751.plus_i(x3752);

			// Dealloc(X8865)
			JCudaTensor x3753;
			x3753 = x3748;
			x3753.free();

			// val X8867 = ReLU()(X8866)
			JCudaTensor x3754;
			JCudaTensor x3755;
			x3755 = x3750;
			x3754 = x268.forward(x3755);

			// val X8868 = Convolv(1,0)(X8867,3b_a_cv_W,3b_a_cv_B)
			JCudaTensor x3756;
			JCudaTensor x3757, x3758, x3759;
			x3757 = x3754;
			x3758 = x280;
			x3759 = x281;
			x3756 = x282.forward(x3757, x3758, x3759);

			// val X8869 = BatchNorm(3b_a_bn)(X8868,3b_a_bn_scale,3b_a_bn_bias)
			JCudaTensor x3760;
			JCudaTensor x3761, x3762, x3763;
			x3761 = x3756;
			x3762 = x287;
			x3763 = x288;
			x3760 = x289.forward_inference(x3761, x3762, x3763);

			// Dealloc(X8868)
			JCudaTensor x3764;
			x3764 = x3756;
			x3764.free();

			// val X8870 = ReLU()(X8869)
			JCudaTensor x3765;
			JCudaTensor x3766;
			x3766 = x3760;
			x3765 = x235.forward(x3766);

			// val X8871 = Convolv(1,1)(X8870,3b_b_cv_W,3b_b_cv_B)
			JCudaTensor x3767;
			JCudaTensor x3768, x3769, x3770;
			x3768 = x3765;
			x3769 = x296;
			x3770 = x297;
			x3767 = x242.forward(x3768, x3769, x3770);

			// Dealloc(X8870)
			JCudaTensor x3771;
			x3771 = x3765;
			x3771.free();

			// val X8872 = BatchNorm(3b_b_bn)(X8871,3b_b_bn_scale,3b_b_bn_bias)
			JCudaTensor x3772;
			JCudaTensor x3773, x3774, x3775;
			x3773 = x3767;
			x3774 = x302;
			x3775 = x303;
			x3772 = x304.forward_inference(x3773, x3774, x3775);

			// Dealloc(X8871)
			JCudaTensor x3776;
			x3776 = x3767;
			x3776.free();

			// val X8873 = ReLU()(X8872)
			JCudaTensor x3777;
			JCudaTensor x3778;
			x3778 = x3772;
			x3777 = x235.forward(x3778);

			// val X8874 = Convolv(1,0)(X8873,3b_c_cv_W,3b_c_cv_B)
			JCudaTensor x3779;
			JCudaTensor x3780, x3781, x3782;
			x3780 = x3777;
			x3781 = x311;
			x3782 = x312;
			x3779 = x258.forward(x3780, x3781, x3782);

			// Dealloc(X8873)
			JCudaTensor x3783;
			x3783 = x3777;
			x3783.free();

			// val X8875 = BatchNorm(3b_c_bn)(X8874,3b_c_bn_scale,3b_c_bn_bias)
			JCudaTensor x3784;
			JCudaTensor x3785, x3786, x3787;
			x3785 = x3779;
			x3786 = x317;
			x3787 = x318;
			x3784 = x319.forward_inference(x3785, x3786, x3787);

			// Dealloc(X8874)
			JCudaTensor x3788;
			x3788 = x3779;
			x3788.free();

			// val X8876 = ReLU()(X8875)
			JCudaTensor x3789;
			JCudaTensor x3790;
			x3790 = x3784;
			x3789 = x268.forward(x3790);

			// val X8877 = (X8876 + X8867)
			JCudaTensor x3791;
			JCudaTensor x3792, x3793;
			x3792 = x3789;
			x3793 = x3754;
			x3791 = x3792.plus_i(x3793);

			// Dealloc(X8867)
			JCudaTensor x3794;
			x3794 = x3754;
			x3794.free();

			// val X8878 = ReLU()(X8877)
			JCudaTensor x3795;
			JCudaTensor x3796;
			x3796 = x3791;
			x3795 = x268.forward(x3796);

			// val X8879 = Convolv(1,0)(X8878,3c_a_cv_W,3c_a_cv_B)
			JCudaTensor x3797;
			JCudaTensor x3798, x3799, x3800;
			x3798 = x3795;
			x3799 = x331;
			x3800 = x332;
			x3797 = x282.forward(x3798, x3799, x3800);

			// val X8880 = BatchNorm(3c_a_bn)(X8879,3c_a_bn_scale,3c_a_bn_bias)
			JCudaTensor x3801;
			JCudaTensor x3802, x3803, x3804;
			x3802 = x3797;
			x3803 = x337;
			x3804 = x338;
			x3801 = x339.forward_inference(x3802, x3803, x3804);

			// Dealloc(X8879)
			JCudaTensor x3805;
			x3805 = x3797;
			x3805.free();

			// val X8881 = ReLU()(X8880)
			JCudaTensor x3806;
			JCudaTensor x3807;
			x3807 = x3801;
			x3806 = x235.forward(x3807);

			// val X8882 = Convolv(1,1)(X8881,3c_b_cv_W,3c_b_cv_B)
			JCudaTensor x3808;
			JCudaTensor x3809, x3810, x3811;
			x3809 = x3806;
			x3810 = x346;
			x3811 = x347;
			x3808 = x242.forward(x3809, x3810, x3811);

			// Dealloc(X8881)
			JCudaTensor x3812;
			x3812 = x3806;
			x3812.free();

			// val X8883 = BatchNorm(3c_b_bn)(X8882,3c_b_bn_scale,3c_b_bn_bias)
			JCudaTensor x3813;
			JCudaTensor x3814, x3815, x3816;
			x3814 = x3808;
			x3815 = x352;
			x3816 = x353;
			x3813 = x354.forward_inference(x3814, x3815, x3816);

			// Dealloc(X8882)
			JCudaTensor x3817;
			x3817 = x3808;
			x3817.free();

			// val X8884 = ReLU()(X8883)
			JCudaTensor x3818;
			JCudaTensor x3819;
			x3819 = x3813;
			x3818 = x235.forward(x3819);

			// val X8885 = Convolv(1,0)(X8884,3c_c_cv_W,3c_c_cv_B)
			JCudaTensor x3820;
			JCudaTensor x3821, x3822, x3823;
			x3821 = x3818;
			x3822 = x361;
			x3823 = x362;
			x3820 = x258.forward(x3821, x3822, x3823);

			// Dealloc(X8884)
			JCudaTensor x3824;
			x3824 = x3818;
			x3824.free();

			// val X8886 = BatchNorm(3c_c_bn)(X8885,3c_c_bn_scale,3c_c_bn_bias)
			JCudaTensor x3825;
			JCudaTensor x3826, x3827, x3828;
			x3826 = x3820;
			x3827 = x367;
			x3828 = x368;
			x3825 = x369.forward_inference(x3826, x3827, x3828);

			// Dealloc(X8885)
			JCudaTensor x3829;
			x3829 = x3820;
			x3829.free();

			// val X8887 = ReLU()(X8886)
			JCudaTensor x3830;
			JCudaTensor x3831;
			x3831 = x3825;
			x3830 = x268.forward(x3831);

			// val X8888 = (X8887 + X8878)
			JCudaTensor x3832;
			JCudaTensor x3833, x3834;
			x3833 = x3830;
			x3834 = x3795;
			x3832 = x3833.plus_i(x3834);

			// Dealloc(X8878)
			JCudaTensor x3835;
			x3835 = x3795;
			x3835.free();

			// val X8889 = ReLU()(X8888)
			JCudaTensor x3836;
			JCudaTensor x3837;
			x3837 = x3832;
			x3836 = x268.forward(x3837);

			// val X8890 = Convolv(1,0)(X8889,3d_a_cv_W,3d_a_cv_B)
			JCudaTensor x3838;
			JCudaTensor x3839, x3840, x3841;
			x3839 = x3836;
			x3840 = x381;
			x3841 = x382;
			x3838 = x282.forward(x3839, x3840, x3841);

			// val X8891 = BatchNorm(3d_a_bn)(X8890,3d_a_bn_scale,3d_a_bn_bias)
			JCudaTensor x3842;
			JCudaTensor x3843, x3844, x3845;
			x3843 = x3838;
			x3844 = x387;
			x3845 = x388;
			x3842 = x389.forward_inference(x3843, x3844, x3845);

			// Dealloc(X8890)
			JCudaTensor x3846;
			x3846 = x3838;
			x3846.free();

			// val X8892 = ReLU()(X8891)
			JCudaTensor x3847;
			JCudaTensor x3848;
			x3848 = x3842;
			x3847 = x235.forward(x3848);

			// val X8893 = Convolv(1,1)(X8892,3d_b_cv_W,3d_b_cv_B)
			JCudaTensor x3849;
			JCudaTensor x3850, x3851, x3852;
			x3850 = x3847;
			x3851 = x396;
			x3852 = x397;
			x3849 = x242.forward(x3850, x3851, x3852);

			// Dealloc(X8892)
			JCudaTensor x3853;
			x3853 = x3847;
			x3853.free();

			// val X8894 = BatchNorm(3d_b_bn)(X8893,3d_b_bn_scale,3d_b_bn_bias)
			JCudaTensor x3854;
			JCudaTensor x3855, x3856, x3857;
			x3855 = x3849;
			x3856 = x402;
			x3857 = x403;
			x3854 = x404.forward_inference(x3855, x3856, x3857);

			// Dealloc(X8893)
			JCudaTensor x3858;
			x3858 = x3849;
			x3858.free();

			// val X8895 = ReLU()(X8894)
			JCudaTensor x3859;
			JCudaTensor x3860;
			x3860 = x3854;
			x3859 = x235.forward(x3860);

			// val X8896 = Convolv(1,0)(X8895,3d_c_cv_W,3d_c_cv_B)
			JCudaTensor x3861;
			JCudaTensor x3862, x3863, x3864;
			x3862 = x3859;
			x3863 = x411;
			x3864 = x412;
			x3861 = x258.forward(x3862, x3863, x3864);

			// Dealloc(X8895)
			JCudaTensor x3865;
			x3865 = x3859;
			x3865.free();

			// val X8897 = BatchNorm(3d_c_bn)(X8896,3d_c_bn_scale,3d_c_bn_bias)
			JCudaTensor x3866;
			JCudaTensor x3867, x3868, x3869;
			x3867 = x3861;
			x3868 = x417;
			x3869 = x418;
			x3866 = x419.forward_inference(x3867, x3868, x3869);

			// Dealloc(X8896)
			JCudaTensor x3870;
			x3870 = x3861;
			x3870.free();

			// val X8898 = ReLU()(X8897)
			JCudaTensor x3871;
			JCudaTensor x3872;
			x3872 = x3866;
			x3871 = x268.forward(x3872);

			// val X8899 = (X8898 + X8889)
			JCudaTensor x3873;
			JCudaTensor x3874, x3875;
			x3874 = x3871;
			x3875 = x3836;
			x3873 = x3874.plus_i(x3875);

			// Dealloc(X8889)
			JCudaTensor x3876;
			x3876 = x3836;
			x3876.free();

			// val X8900 = ReLU()(X8899)
			JCudaTensor x3877;
			JCudaTensor x3878;
			x3878 = x3873;
			x3877 = x268.forward(x3878);

			// val X8904 = Convolv(2,0)(X8900,4a2_a_cv_W,4a2_a_cv_B)
			JCudaTensor x3879;
			JCudaTensor x3880, x3881, x3882;
			x3880 = x3877;
			x3881 = x438;
			x3882 = x439;
			x3879 = x440.forward(x3880, x3881, x3882);

			// val X8901 = Convolv(2,0)(X8900,4a1_cv_W,4a1_cv_B)
			JCudaTensor x3883;
			JCudaTensor x3884, x3885, x3886;
			x3884 = x3877;
			x3885 = x431;
			x3886 = x432;
			x3883 = x433.forward(x3884, x3885, x3886);

			// Dealloc(X8900)
			JCudaTensor x3887;
			x3887 = x3877;
			x3887.free();

			// val X8905 = BatchNorm(4a2_a_bn)(X8904,4a2_a_bn_scale,4a2_a_bn_bias)
			JCudaTensor x3888;
			JCudaTensor x3889, x3890, x3891;
			x3889 = x3879;
			x3890 = x452;
			x3891 = x453;
			x3888 = x454.forward_inference(x3889, x3890, x3891);

			// Dealloc(X8904)
			JCudaTensor x3892;
			x3892 = x3879;
			x3892.free();

			// val X8902 = BatchNorm(4a1_bn)(X8901,4a1_bn_scale,4a1_bn_bias)
			JCudaTensor x3893;
			JCudaTensor x3894, x3895, x3896;
			x3894 = x3883;
			x3895 = x445;
			x3896 = x446;
			x3893 = x447.forward_inference(x3894, x3895, x3896);

			// Dealloc(X8901)
			JCudaTensor x3897;
			x3897 = x3883;
			x3897.free();

			// val X8906 = ReLU()(X8905)
			JCudaTensor x3898;
			JCudaTensor x3899;
			x3899 = x3888;
			x3898 = x457.forward(x3899);

			// val X8907 = Convolv(1,1)(X8906,4a2_b_cv_W,4a2_b_cv_B)
			JCudaTensor x3900;
			JCudaTensor x3901, x3902, x3903;
			x3901 = x3898;
			x3902 = x462;
			x3903 = x463;
			x3900 = x464.forward(x3901, x3902, x3903);

			// Dealloc(X8906)
			JCudaTensor x3904;
			x3904 = x3898;
			x3904.free();

			// val X8908 = BatchNorm(4a2_b_bn)(X8907,4a2_b_bn_scale,4a2_b_bn_bias)
			JCudaTensor x3905;
			JCudaTensor x3906, x3907, x3908;
			x3906 = x3900;
			x3907 = x469;
			x3908 = x470;
			x3905 = x471.forward_inference(x3906, x3907, x3908);

			// Dealloc(X8907)
			JCudaTensor x3909;
			x3909 = x3900;
			x3909.free();

			// val X8909 = ReLU()(X8908)
			JCudaTensor x3910;
			JCudaTensor x3911;
			x3911 = x3905;
			x3910 = x457.forward(x3911);

			// val X8910 = Convolv(1,0)(X8909,4a2_c_cv_W,4a2_c_cv_B)
			JCudaTensor x3912;
			JCudaTensor x3913, x3914, x3915;
			x3913 = x3910;
			x3914 = x478;
			x3915 = x479;
			x3912 = x480.forward(x3913, x3914, x3915);

			// Dealloc(X8909)
			JCudaTensor x3916;
			x3916 = x3910;
			x3916.free();

			// val X8911 = BatchNorm(4a2_c_bn)(X8910,4a2_c_bn_scale,4a2_c_bn_bias)
			JCudaTensor x3917;
			JCudaTensor x3918, x3919, x3920;
			x3918 = x3912;
			x3919 = x485;
			x3920 = x486;
			x3917 = x487.forward_inference(x3918, x3919, x3920);

			// Dealloc(X8910)
			JCudaTensor x3921;
			x3921 = x3912;
			x3921.free();

			// val X8903 = ReLU()(X8902)
			JCudaTensor x3922;
			JCudaTensor x3923;
			x3923 = x3893;
			x3922 = x490.forward(x3923);

			// val X8912 = ReLU()(X8911)
			JCudaTensor x3924;
			JCudaTensor x3925;
			x3925 = x3917;
			x3924 = x490.forward(x3925);

			// val X8913 = (X8903 + X8912)
			JCudaTensor x3926;
			JCudaTensor x3927, x3928;
			x3927 = x3922;
			x3928 = x3924;
			x3926 = x3927.plus_i(x3928);

			// Dealloc(X8912)
			JCudaTensor x3929;
			x3929 = x3924;
			x3929.free();

			// val X8914 = ReLU()(X8913)
			JCudaTensor x3930;
			JCudaTensor x3931;
			x3931 = x3926;
			x3930 = x490.forward(x3931);

			// val X8915 = Convolv(1,0)(X8914,4b_a_cv_W,4b_a_cv_B)
			JCudaTensor x3932;
			JCudaTensor x3933, x3934, x3935;
			x3933 = x3930;
			x3934 = x502;
			x3935 = x503;
			x3932 = x504.forward(x3933, x3934, x3935);

			// val X8916 = BatchNorm(4b_a_bn)(X8915,4b_a_bn_scale,4b_a_bn_bias)
			JCudaTensor x3936;
			JCudaTensor x3937, x3938, x3939;
			x3937 = x3932;
			x3938 = x509;
			x3939 = x510;
			x3936 = x511.forward_inference(x3937, x3938, x3939);

			// Dealloc(X8915)
			JCudaTensor x3940;
			x3940 = x3932;
			x3940.free();

			// val X8917 = ReLU()(X8916)
			JCudaTensor x3941;
			JCudaTensor x3942;
			x3942 = x3936;
			x3941 = x457.forward(x3942);

			// val X8918 = Convolv(1,1)(X8917,4b_b_cv_W,4b_b_cv_B)
			JCudaTensor x3943;
			JCudaTensor x3944, x3945, x3946;
			x3944 = x3941;
			x3945 = x518;
			x3946 = x519;
			x3943 = x464.forward(x3944, x3945, x3946);

			// Dealloc(X8917)
			JCudaTensor x3947;
			x3947 = x3941;
			x3947.free();

			// val X8919 = BatchNorm(4b_b_bn)(X8918,4b_b_bn_scale,4b_b_bn_bias)
			JCudaTensor x3948;
			JCudaTensor x3949, x3950, x3951;
			x3949 = x3943;
			x3950 = x524;
			x3951 = x525;
			x3948 = x526.forward_inference(x3949, x3950, x3951);

			// Dealloc(X8918)
			JCudaTensor x3952;
			x3952 = x3943;
			x3952.free();

			// val X8920 = ReLU()(X8919)
			JCudaTensor x3953;
			JCudaTensor x3954;
			x3954 = x3948;
			x3953 = x457.forward(x3954);

			// val X8921 = Convolv(1,0)(X8920,4b_c_cv_W,4b_c_cv_B)
			JCudaTensor x3955;
			JCudaTensor x3956, x3957, x3958;
			x3956 = x3953;
			x3957 = x533;
			x3958 = x534;
			x3955 = x480.forward(x3956, x3957, x3958);

			// Dealloc(X8920)
			JCudaTensor x3959;
			x3959 = x3953;
			x3959.free();

			// val X8922 = BatchNorm(4b_c_bn)(X8921,4b_c_bn_scale,4b_c_bn_bias)
			JCudaTensor x3960;
			JCudaTensor x3961, x3962, x3963;
			x3961 = x3955;
			x3962 = x539;
			x3963 = x540;
			x3960 = x541.forward_inference(x3961, x3962, x3963);

			// Dealloc(X8921)
			JCudaTensor x3964;
			x3964 = x3955;
			x3964.free();

			// val X8923 = ReLU()(X8922)
			JCudaTensor x3965;
			JCudaTensor x3966;
			x3966 = x3960;
			x3965 = x490.forward(x3966);

			// val X8924 = (X8923 + X8914)
			JCudaTensor x3967;
			JCudaTensor x3968, x3969;
			x3968 = x3965;
			x3969 = x3930;
			x3967 = x3968.plus_i(x3969);

			// Dealloc(X8914)
			JCudaTensor x3970;
			x3970 = x3930;
			x3970.free();

			// val X8925 = ReLU()(X8924)
			JCudaTensor x3971;
			JCudaTensor x3972;
			x3972 = x3967;
			x3971 = x490.forward(x3972);

			// val X8926 = Convolv(1,0)(X8925,4c_a_cv_W,4c_a_cv_B)
			JCudaTensor x3973;
			JCudaTensor x3974, x3975, x3976;
			x3974 = x3971;
			x3975 = x553;
			x3976 = x554;
			x3973 = x504.forward(x3974, x3975, x3976);

			// val X8927 = BatchNorm(4c_a_bn)(X8926,4c_a_bn_scale,4c_a_bn_bias)
			JCudaTensor x3977;
			JCudaTensor x3978, x3979, x3980;
			x3978 = x3973;
			x3979 = x559;
			x3980 = x560;
			x3977 = x561.forward_inference(x3978, x3979, x3980);

			// Dealloc(X8926)
			JCudaTensor x3981;
			x3981 = x3973;
			x3981.free();

			// val X8928 = ReLU()(X8927)
			JCudaTensor x3982;
			JCudaTensor x3983;
			x3983 = x3977;
			x3982 = x457.forward(x3983);

			// val X8929 = Convolv(1,1)(X8928,4c_b_cv_W,4c_b_cv_B)
			JCudaTensor x3984;
			JCudaTensor x3985, x3986, x3987;
			x3985 = x3982;
			x3986 = x568;
			x3987 = x569;
			x3984 = x464.forward(x3985, x3986, x3987);

			// Dealloc(X8928)
			JCudaTensor x3988;
			x3988 = x3982;
			x3988.free();

			// val X8930 = BatchNorm(4c_b_bn)(X8929,4c_b_bn_scale,4c_b_bn_bias)
			JCudaTensor x3989;
			JCudaTensor x3990, x3991, x3992;
			x3990 = x3984;
			x3991 = x574;
			x3992 = x575;
			x3989 = x576.forward_inference(x3990, x3991, x3992);

			// Dealloc(X8929)
			JCudaTensor x3993;
			x3993 = x3984;
			x3993.free();

			// val X8931 = ReLU()(X8930)
			JCudaTensor x3994;
			JCudaTensor x3995;
			x3995 = x3989;
			x3994 = x457.forward(x3995);

			// val X8932 = Convolv(1,0)(X8931,4c_c_cv_W,4c_c_cv_B)
			JCudaTensor x3996;
			JCudaTensor x3997, x3998, x3999;
			x3997 = x3994;
			x3998 = x583;
			x3999 = x584;
			x3996 = x480.forward(x3997, x3998, x3999);

			// Dealloc(X8931)
			JCudaTensor x4000;
			x4000 = x3994;
			x4000.free();

			// val X8933 = BatchNorm(4c_c_bn)(X8932,4c_c_bn_scale,4c_c_bn_bias)
			JCudaTensor x4001;
			JCudaTensor x4002, x4003, x4004;
			x4002 = x3996;
			x4003 = x589;
			x4004 = x590;
			x4001 = x591.forward_inference(x4002, x4003, x4004);

			// Dealloc(X8932)
			JCudaTensor x4005;
			x4005 = x3996;
			x4005.free();

			// val X8934 = ReLU()(X8933)
			JCudaTensor x4006;
			JCudaTensor x4007;
			x4007 = x4001;
			x4006 = x490.forward(x4007);

			// val X8935 = (X8934 + X8925)
			JCudaTensor x4008;
			JCudaTensor x4009, x4010;
			x4009 = x4006;
			x4010 = x3971;
			x4008 = x4009.plus_i(x4010);

			// Dealloc(X8925)
			JCudaTensor x4011;
			x4011 = x3971;
			x4011.free();

			// val X8936 = ReLU()(X8935)
			JCudaTensor x4012;
			JCudaTensor x4013;
			x4013 = x4008;
			x4012 = x490.forward(x4013);

			// val X8937 = Convolv(1,0)(X8936,4d_a_cv_W,4d_a_cv_B)
			JCudaTensor x4014;
			JCudaTensor x4015, x4016, x4017;
			x4015 = x4012;
			x4016 = x603;
			x4017 = x604;
			x4014 = x504.forward(x4015, x4016, x4017);

			// val X8938 = BatchNorm(4d_a_bn)(X8937,4d_a_bn_scale,4d_a_bn_bias)
			JCudaTensor x4018;
			JCudaTensor x4019, x4020, x4021;
			x4019 = x4014;
			x4020 = x609;
			x4021 = x610;
			x4018 = x611.forward_inference(x4019, x4020, x4021);

			// Dealloc(X8937)
			JCudaTensor x4022;
			x4022 = x4014;
			x4022.free();

			// val X8939 = ReLU()(X8938)
			JCudaTensor x4023;
			JCudaTensor x4024;
			x4024 = x4018;
			x4023 = x457.forward(x4024);

			// val X8940 = Convolv(1,1)(X8939,4d_b_cv_W,4d_b_cv_B)
			JCudaTensor x4025;
			JCudaTensor x4026, x4027, x4028;
			x4026 = x4023;
			x4027 = x618;
			x4028 = x619;
			x4025 = x464.forward(x4026, x4027, x4028);

			// Dealloc(X8939)
			JCudaTensor x4029;
			x4029 = x4023;
			x4029.free();

			// val X8941 = BatchNorm(4d_b_bn)(X8940,4d_b_bn_scale,4d_b_bn_bias)
			JCudaTensor x4030;
			JCudaTensor x4031, x4032, x4033;
			x4031 = x4025;
			x4032 = x624;
			x4033 = x625;
			x4030 = x626.forward_inference(x4031, x4032, x4033);

			// Dealloc(X8940)
			JCudaTensor x4034;
			x4034 = x4025;
			x4034.free();

			// val X8942 = ReLU()(X8941)
			JCudaTensor x4035;
			JCudaTensor x4036;
			x4036 = x4030;
			x4035 = x457.forward(x4036);

			// val X8943 = Convolv(1,0)(X8942,4d_c_cv_W,4d_c_cv_B)
			JCudaTensor x4037;
			JCudaTensor x4038, x4039, x4040;
			x4038 = x4035;
			x4039 = x633;
			x4040 = x634;
			x4037 = x480.forward(x4038, x4039, x4040);

			// Dealloc(X8942)
			JCudaTensor x4041;
			x4041 = x4035;
			x4041.free();

			// val X8944 = BatchNorm(4d_c_bn)(X8943,4d_c_bn_scale,4d_c_bn_bias)
			JCudaTensor x4042;
			JCudaTensor x4043, x4044, x4045;
			x4043 = x4037;
			x4044 = x639;
			x4045 = x640;
			x4042 = x641.forward_inference(x4043, x4044, x4045);

			// Dealloc(X8943)
			JCudaTensor x4046;
			x4046 = x4037;
			x4046.free();

			// val X8945 = ReLU()(X8944)
			JCudaTensor x4047;
			JCudaTensor x4048;
			x4048 = x4042;
			x4047 = x490.forward(x4048);

			// val X8946 = (X8945 + X8936)
			JCudaTensor x4049;
			JCudaTensor x4050, x4051;
			x4050 = x4047;
			x4051 = x4012;
			x4049 = x4050.plus_i(x4051);

			// Dealloc(X8936)
			JCudaTensor x4052;
			x4052 = x4012;
			x4052.free();

			// val X8947 = ReLU()(X8946)
			JCudaTensor x4053;
			JCudaTensor x4054;
			x4054 = x4049;
			x4053 = x490.forward(x4054);

			// val X8948 = Convolv(1,0)(X8947,4e_a_cv_W,4e_a_cv_B)
			JCudaTensor x4055;
			JCudaTensor x4056, x4057, x4058;
			x4056 = x4053;
			x4057 = x653;
			x4058 = x654;
			x4055 = x504.forward(x4056, x4057, x4058);

			// val X8949 = BatchNorm(4e_a_bn)(X8948,4e_a_bn_scale,4e_a_bn_bias)
			JCudaTensor x4059;
			JCudaTensor x4060, x4061, x4062;
			x4060 = x4055;
			x4061 = x659;
			x4062 = x660;
			x4059 = x661.forward_inference(x4060, x4061, x4062);

			// Dealloc(X8948)
			JCudaTensor x4063;
			x4063 = x4055;
			x4063.free();

			// val X8950 = ReLU()(X8949)
			JCudaTensor x4064;
			JCudaTensor x4065;
			x4065 = x4059;
			x4064 = x457.forward(x4065);

			// val X8951 = Convolv(1,1)(X8950,4e_b_cv_W,4e_b_cv_B)
			JCudaTensor x4066;
			JCudaTensor x4067, x4068, x4069;
			x4067 = x4064;
			x4068 = x668;
			x4069 = x669;
			x4066 = x464.forward(x4067, x4068, x4069);

			// Dealloc(X8950)
			JCudaTensor x4070;
			x4070 = x4064;
			x4070.free();

			// val X8952 = BatchNorm(4e_b_bn)(X8951,4e_b_bn_scale,4e_b_bn_bias)
			JCudaTensor x4071;
			JCudaTensor x4072, x4073, x4074;
			x4072 = x4066;
			x4073 = x674;
			x4074 = x675;
			x4071 = x676.forward_inference(x4072, x4073, x4074);

			// Dealloc(X8951)
			JCudaTensor x4075;
			x4075 = x4066;
			x4075.free();

			// val X8953 = ReLU()(X8952)
			JCudaTensor x4076;
			JCudaTensor x4077;
			x4077 = x4071;
			x4076 = x457.forward(x4077);

			// val X8954 = Convolv(1,0)(X8953,4e_c_cv_W,4e_c_cv_B)
			JCudaTensor x4078;
			JCudaTensor x4079, x4080, x4081;
			x4079 = x4076;
			x4080 = x683;
			x4081 = x684;
			x4078 = x480.forward(x4079, x4080, x4081);

			// Dealloc(X8953)
			JCudaTensor x4082;
			x4082 = x4076;
			x4082.free();

			// val X8955 = BatchNorm(4e_c_bn)(X8954,4e_c_bn_scale,4e_c_bn_bias)
			JCudaTensor x4083;
			JCudaTensor x4084, x4085, x4086;
			x4084 = x4078;
			x4085 = x689;
			x4086 = x690;
			x4083 = x691.forward_inference(x4084, x4085, x4086);

			// Dealloc(X8954)
			JCudaTensor x4087;
			x4087 = x4078;
			x4087.free();

			// val X8956 = ReLU()(X8955)
			JCudaTensor x4088;
			JCudaTensor x4089;
			x4089 = x4083;
			x4088 = x490.forward(x4089);

			// val X8957 = (X8956 + X8947)
			JCudaTensor x4090;
			JCudaTensor x4091, x4092;
			x4091 = x4088;
			x4092 = x4053;
			x4090 = x4091.plus_i(x4092);

			// Dealloc(X8947)
			JCudaTensor x4093;
			x4093 = x4053;
			x4093.free();

			// val X8958 = ReLU()(X8957)
			JCudaTensor x4094;
			JCudaTensor x4095;
			x4095 = x4090;
			x4094 = x490.forward(x4095);

			// val X8959 = Convolv(1,0)(X8958,4f_a_cv_W,4f_a_cv_B)
			JCudaTensor x4096;
			JCudaTensor x4097, x4098, x4099;
			x4097 = x4094;
			x4098 = x703;
			x4099 = x704;
			x4096 = x504.forward(x4097, x4098, x4099);

			// val X8960 = BatchNorm(4f_a_bn)(X8959,4f_a_bn_scale,4f_a_bn_bias)
			JCudaTensor x4100;
			JCudaTensor x4101, x4102, x4103;
			x4101 = x4096;
			x4102 = x709;
			x4103 = x710;
			x4100 = x711.forward_inference(x4101, x4102, x4103);

			// Dealloc(X8959)
			JCudaTensor x4104;
			x4104 = x4096;
			x4104.free();

			// val X8961 = ReLU()(X8960)
			JCudaTensor x4105;
			JCudaTensor x4106;
			x4106 = x4100;
			x4105 = x457.forward(x4106);

			// val X8962 = Convolv(1,1)(X8961,4f_b_cv_W,4f_b_cv_B)
			JCudaTensor x4107;
			JCudaTensor x4108, x4109, x4110;
			x4108 = x4105;
			x4109 = x718;
			x4110 = x719;
			x4107 = x464.forward(x4108, x4109, x4110);

			// Dealloc(X8961)
			JCudaTensor x4111;
			x4111 = x4105;
			x4111.free();

			// val X8963 = BatchNorm(4f_b_bn)(X8962,4f_b_bn_scale,4f_b_bn_bias)
			JCudaTensor x4112;
			JCudaTensor x4113, x4114, x4115;
			x4113 = x4107;
			x4114 = x724;
			x4115 = x725;
			x4112 = x726.forward_inference(x4113, x4114, x4115);

			// Dealloc(X8962)
			JCudaTensor x4116;
			x4116 = x4107;
			x4116.free();

			// val X8964 = ReLU()(X8963)
			JCudaTensor x4117;
			JCudaTensor x4118;
			x4118 = x4112;
			x4117 = x457.forward(x4118);

			// val X8965 = Convolv(1,0)(X8964,4f_c_cv_W,4f_c_cv_B)
			JCudaTensor x4119;
			JCudaTensor x4120, x4121, x4122;
			x4120 = x4117;
			x4121 = x733;
			x4122 = x734;
			x4119 = x480.forward(x4120, x4121, x4122);

			// Dealloc(X8964)
			JCudaTensor x4123;
			x4123 = x4117;
			x4123.free();

			// val X8966 = BatchNorm(4f_c_bn)(X8965,4f_c_bn_scale,4f_c_bn_bias)
			JCudaTensor x4124;
			JCudaTensor x4125, x4126, x4127;
			x4125 = x4119;
			x4126 = x739;
			x4127 = x740;
			x4124 = x741.forward_inference(x4125, x4126, x4127);

			// Dealloc(X8965)
			JCudaTensor x4128;
			x4128 = x4119;
			x4128.free();

			// val X8967 = ReLU()(X8966)
			JCudaTensor x4129;
			JCudaTensor x4130;
			x4130 = x4124;
			x4129 = x490.forward(x4130);

			// val X8968 = (X8967 + X8958)
			JCudaTensor x4131;
			JCudaTensor x4132, x4133;
			x4132 = x4129;
			x4133 = x4094;
			x4131 = x4132.plus_i(x4133);

			// Dealloc(X8958)
			JCudaTensor x4134;
			x4134 = x4094;
			x4134.free();

			// val X8969 = ReLU()(X8968)
			JCudaTensor x4135;
			JCudaTensor x4136;
			x4136 = x4131;
			x4135 = x490.forward(x4136);

			// val X8970 = Convolv(2,0)(X8969,5a1_cv_W,5a1_cv_B)
			JCudaTensor x4137;
			JCudaTensor x4138, x4139, x4140;
			x4138 = x4135;
			x4139 = x760;
			x4140 = x761;
			x4137 = x762.forward(x4138, x4139, x4140);

			// val X8973 = Convolv(2,0)(X8969,5a2_a_cv_W,5a2_a_cv_B)
			JCudaTensor x4141;
			JCudaTensor x4142, x4143, x4144;
			x4142 = x4135;
			x4143 = x753;
			x4144 = x754;
			x4141 = x755.forward(x4142, x4143, x4144);

			// Dealloc(X8969)
			JCudaTensor x4145;
			x4145 = x4135;
			x4145.free();

			// val X8974 = BatchNorm(5a2_a_bn)(X8973,5a2_a_bn_scale,5a2_a_bn_bias)
			JCudaTensor x4146;
			JCudaTensor x4147, x4148, x4149;
			x4147 = x4141;
			x4148 = x774;
			x4149 = x775;
			x4146 = x776.forward_inference(x4147, x4148, x4149);

			// Dealloc(X8973)
			JCudaTensor x4150;
			x4150 = x4141;
			x4150.free();

			// val X8971 = BatchNorm(5a1_bn)(X8970,5a1_bn_scale,5a1_bn_bias)
			JCudaTensor x4151;
			JCudaTensor x4152, x4153, x4154;
			x4152 = x4137;
			x4153 = x767;
			x4154 = x768;
			x4151 = x769.forward_inference(x4152, x4153, x4154);

			// Dealloc(X8970)
			JCudaTensor x4155;
			x4155 = x4137;
			x4155.free();

			// val X8975 = ReLU()(X8974)
			JCudaTensor x4156;
			JCudaTensor x4157;
			x4157 = x4146;
			x4156 = x779.forward(x4157);

			// val X8976 = Convolv(1,1)(X8975,5a2_b_cv_W,5a2_b_cv_B)
			JCudaTensor x4158;
			JCudaTensor x4159, x4160, x4161;
			x4159 = x4156;
			x4160 = x784;
			x4161 = x785;
			x4158 = x786.forward(x4159, x4160, x4161);

			// Dealloc(X8975)
			JCudaTensor x4162;
			x4162 = x4156;
			x4162.free();

			// val X8977 = BatchNorm(5a2_b_bn)(X8976,5a2_b_bn_scale,5a2_b_bn_bias)
			JCudaTensor x4163;
			JCudaTensor x4164, x4165, x4166;
			x4164 = x4158;
			x4165 = x791;
			x4166 = x792;
			x4163 = x793.forward_inference(x4164, x4165, x4166);

			// Dealloc(X8976)
			JCudaTensor x4167;
			x4167 = x4158;
			x4167.free();

			// val X8978 = ReLU()(X8977)
			JCudaTensor x4168;
			JCudaTensor x4169;
			x4169 = x4163;
			x4168 = x779.forward(x4169);

			// val X8979 = Convolv(1,0)(X8978,5a2_c_cv_W,5a2_c_cv_B)
			JCudaTensor x4170;
			JCudaTensor x4171, x4172, x4173;
			x4171 = x4168;
			x4172 = x800;
			x4173 = x801;
			x4170 = x802.forward(x4171, x4172, x4173);

			// Dealloc(X8978)
			JCudaTensor x4174;
			x4174 = x4168;
			x4174.free();

			// val X8980 = BatchNorm(5a2_c_bn)(X8979,5a2_c_bn_scale,5a2_c_bn_bias)
			JCudaTensor x4175;
			JCudaTensor x4176, x4177, x4178;
			x4176 = x4170;
			x4177 = x807;
			x4178 = x808;
			x4175 = x809.forward_inference(x4176, x4177, x4178);

			// Dealloc(X8979)
			JCudaTensor x4179;
			x4179 = x4170;
			x4179.free();

			// val X8972 = ReLU()(X8971)
			JCudaTensor x4180;
			JCudaTensor x4181;
			x4181 = x4151;
			x4180 = x812.forward(x4181);

			// val X8981 = ReLU()(X8980)
			JCudaTensor x4182;
			JCudaTensor x4183;
			x4183 = x4175;
			x4182 = x812.forward(x4183);

			// val X8982 = (X8972 + X8981)
			JCudaTensor x4184;
			JCudaTensor x4185, x4186;
			x4185 = x4180;
			x4186 = x4182;
			x4184 = x4185.plus_i(x4186);

			// Dealloc(X8981)
			JCudaTensor x4187;
			x4187 = x4182;
			x4187.free();

			// val X8983 = ReLU()(X8982)
			JCudaTensor x4188;
			JCudaTensor x4189;
			x4189 = x4184;
			x4188 = x812.forward(x4189);

			// val X8984 = Convolv(1,0)(X8983,5b_a_cv_W,5b_a_cv_B)
			JCudaTensor x4190;
			JCudaTensor x4191, x4192, x4193;
			x4191 = x4188;
			x4192 = x824;
			x4193 = x825;
			x4190 = x826.forward(x4191, x4192, x4193);

			// val X8985 = BatchNorm(5b_a_bn)(X8984,5b_a_bn_scale,5b_a_bn_bias)
			JCudaTensor x4194;
			JCudaTensor x4195, x4196, x4197;
			x4195 = x4190;
			x4196 = x831;
			x4197 = x832;
			x4194 = x833.forward_inference(x4195, x4196, x4197);

			// Dealloc(X8984)
			JCudaTensor x4198;
			x4198 = x4190;
			x4198.free();

			// val X8986 = ReLU()(X8985)
			JCudaTensor x4199;
			JCudaTensor x4200;
			x4200 = x4194;
			x4199 = x779.forward(x4200);

			// val X8987 = Convolv(1,1)(X8986,5b_b_cv_W,5b_b_cv_B)
			JCudaTensor x4201;
			JCudaTensor x4202, x4203, x4204;
			x4202 = x4199;
			x4203 = x840;
			x4204 = x841;
			x4201 = x786.forward(x4202, x4203, x4204);

			// Dealloc(X8986)
			JCudaTensor x4205;
			x4205 = x4199;
			x4205.free();

			// val X8988 = BatchNorm(5b_b_bn)(X8987,5b_b_bn_scale,5b_b_bn_bias)
			JCudaTensor x4206;
			JCudaTensor x4207, x4208, x4209;
			x4207 = x4201;
			x4208 = x846;
			x4209 = x847;
			x4206 = x848.forward_inference(x4207, x4208, x4209);

			// Dealloc(X8987)
			JCudaTensor x4210;
			x4210 = x4201;
			x4210.free();

			// val X8989 = ReLU()(X8988)
			JCudaTensor x4211;
			JCudaTensor x4212;
			x4212 = x4206;
			x4211 = x779.forward(x4212);

			// val X8990 = Convolv(1,0)(X8989,5b_c_cv_W,5b_c_cv_B)
			JCudaTensor x4213;
			JCudaTensor x4214, x4215, x4216;
			x4214 = x4211;
			x4215 = x855;
			x4216 = x856;
			x4213 = x802.forward(x4214, x4215, x4216);

			// Dealloc(X8989)
			JCudaTensor x4217;
			x4217 = x4211;
			x4217.free();

			// val X8991 = BatchNorm(5b_c_bn)(X8990,5b_c_bn_scale,5b_c_bn_bias)
			JCudaTensor x4218;
			JCudaTensor x4219, x4220, x4221;
			x4219 = x4213;
			x4220 = x861;
			x4221 = x862;
			x4218 = x863.forward_inference(x4219, x4220, x4221);

			// Dealloc(X8990)
			JCudaTensor x4222;
			x4222 = x4213;
			x4222.free();

			// val X8992 = ReLU()(X8991)
			JCudaTensor x4223;
			JCudaTensor x4224;
			x4224 = x4218;
			x4223 = x812.forward(x4224);

			// val X8993 = (X8992 + X8983)
			JCudaTensor x4225;
			JCudaTensor x4226, x4227;
			x4226 = x4223;
			x4227 = x4188;
			x4225 = x4226.plus_i(x4227);

			// Dealloc(X8983)
			JCudaTensor x4228;
			x4228 = x4188;
			x4228.free();

			// val X8994 = ReLU()(X8993)
			JCudaTensor x4229;
			JCudaTensor x4230;
			x4230 = x4225;
			x4229 = x812.forward(x4230);

			// val X8995 = Convolv(1,0)(X8994,5c_a_cv_W,5c_a_cv_B)
			JCudaTensor x4231;
			JCudaTensor x4232, x4233, x4234;
			x4232 = x4229;
			x4233 = x875;
			x4234 = x876;
			x4231 = x826.forward(x4232, x4233, x4234);

			// val X8996 = BatchNorm(5c_a_bn)(X8995,5c_a_bn_scale,5c_a_bn_bias)
			JCudaTensor x4235;
			JCudaTensor x4236, x4237, x4238;
			x4236 = x4231;
			x4237 = x881;
			x4238 = x882;
			x4235 = x883.forward_inference(x4236, x4237, x4238);

			// Dealloc(X8995)
			JCudaTensor x4239;
			x4239 = x4231;
			x4239.free();

			// val X8997 = ReLU()(X8996)
			JCudaTensor x4240;
			JCudaTensor x4241;
			x4241 = x4235;
			x4240 = x779.forward(x4241);

			// val X8998 = Convolv(1,1)(X8997,5c_b_cv_W,5c_b_cv_B)
			JCudaTensor x4242;
			JCudaTensor x4243, x4244, x4245;
			x4243 = x4240;
			x4244 = x890;
			x4245 = x891;
			x4242 = x786.forward(x4243, x4244, x4245);

			// Dealloc(X8997)
			JCudaTensor x4246;
			x4246 = x4240;
			x4246.free();

			// val X8999 = BatchNorm(5c_b_bn)(X8998,5c_b_bn_scale,5c_b_bn_bias)
			JCudaTensor x4247;
			JCudaTensor x4248, x4249, x4250;
			x4248 = x4242;
			x4249 = x896;
			x4250 = x897;
			x4247 = x898.forward_inference(x4248, x4249, x4250);

			// Dealloc(X8998)
			JCudaTensor x4251;
			x4251 = x4242;
			x4251.free();

			// val X9000 = ReLU()(X8999)
			JCudaTensor x4252;
			JCudaTensor x4253;
			x4253 = x4247;
			x4252 = x779.forward(x4253);

			// val X9001 = Convolv(1,0)(X9000,5c_c_cv_W,5c_c_cv_B)
			JCudaTensor x4254;
			JCudaTensor x4255, x4256, x4257;
			x4255 = x4252;
			x4256 = x905;
			x4257 = x906;
			x4254 = x802.forward(x4255, x4256, x4257);

			// Dealloc(X9000)
			JCudaTensor x4258;
			x4258 = x4252;
			x4258.free();

			// val X9002 = BatchNorm(5c_c_bn)(X9001,5c_c_bn_scale,5c_c_bn_bias)
			JCudaTensor x4259;
			JCudaTensor x4260, x4261, x4262;
			x4260 = x4254;
			x4261 = x911;
			x4262 = x912;
			x4259 = x913.forward_inference(x4260, x4261, x4262);

			// Dealloc(X9001)
			JCudaTensor x4263;
			x4263 = x4254;
			x4263.free();

			// val X9003 = ReLU()(X9002)
			JCudaTensor x4264;
			JCudaTensor x4265;
			x4265 = x4259;
			x4264 = x812.forward(x4265);

			// val X9004 = (X9003 + X8994)
			JCudaTensor x4266;
			JCudaTensor x4267, x4268;
			x4267 = x4264;
			x4268 = x4229;
			x4266 = x4267.plus_i(x4268);

			// Dealloc(X8994)
			JCudaTensor x4269;
			x4269 = x4229;
			x4269.free();

			// val X9005 = ReLU()(X9004)
			JCudaTensor x4270;
			JCudaTensor x4271;
			x4271 = x4266;
			x4270 = x812.forward(x4271);

			// val X9006 = Pooling(7,1,0,false)(X9005)
			JCudaTensor x4272;
			JCudaTensor x4273;
			x4273 = x4270;
			x4272 = x923.forward(x4273);

			// Dealloc(X9005)
			JCudaTensor x4274;
			x4274 = x4270;
			x4274.free();

			// val X9007 = (X9006[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x4275;
			JCudaMatrix x4276;
			JCudaMatrix x4277;
			JCudaTensor x4278;
			JCudaTensor x4279;
			x4279 = x4272;
			x4278 = x4279.flatten(1, new int[]{2048, 1, 1});
			x4276 = x4278.asMatrix(1, true);
			JCudaTensor x4280;
			x4280 = x930;
			x4277 = x4280.asMatrix(1, true);
			x4275 = x4276.times(x4277);

			// Dealloc(X9006)
			JCudaTensor x4281;
			x4281 = x4272;
			x4281.free();

			// val X9009 = (X9007 + (i) => fc_B)
			JCudaTensor x4282;
			JCudaTensor x4283, x4284;
			x4283 = x4275;
			x4284 = x934;
			x4282 = x4284.copy(64, x4283);

			// Precision(Accuracy(1))
			float x4286;
			JCudaTensor x4287;
			JTensorFloat x4288;
			x4287 = x4282;
			x4288 = x4;
			x4286 = x4287.accuracy(x4288, 1);
			System.out.println(x5 + " test precision "  + x4286);
			x4285 += x4286;

			// Dealloc(X9009)
			JCudaTensor x4289;
			x4289 = x4282;
			x4289.free();

		}
		System.out.println();
		System.out.println("average precision: " + x4285/10);
		System.out.println(); 
	}

}