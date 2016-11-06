package deepdsl.cudnn.config.softmax;

import jcuda.jcudnn.cudnnSoftmaxAlgorithm;

public enum SoftmaxAlgorithm {
	ACCURATE(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE),
	FAST(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_FAST),
	LOG(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_LOG);
	
	private final int algorithm;
    
    public int algorithm() { return algorithm; }
 
    SoftmaxAlgorithm(int algorithm) {
		this.algorithm = algorithm;
	}
}
