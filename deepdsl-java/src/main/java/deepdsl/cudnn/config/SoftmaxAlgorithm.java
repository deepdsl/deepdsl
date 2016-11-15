package deepdsl.cudnn.config;

import jcuda.jcudnn.cudnnSoftmaxAlgorithm;

public enum SoftmaxAlgorithm {
	ACCURATE(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE),
	FAST(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_FAST),
	LOG(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_LOG);
	
	private final int algorithm;
    
    public int value() { return algorithm; }
 
    SoftmaxAlgorithm(int algorithm) {
		this.algorithm = algorithm;
	}
}
