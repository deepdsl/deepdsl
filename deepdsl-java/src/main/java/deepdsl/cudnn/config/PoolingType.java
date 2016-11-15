package deepdsl.cudnn.config;

import jcuda.jcudnn.cudnnPoolingMode;

public enum PoolingType {
	MAX (cudnnPoolingMode.CUDNN_POOLING_MAX),
	AVERAGE_EXCLUDE_PADDING (cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
	AVERAGE_INCLUDE_PADDING (cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
	
	private final int poolingTpe;
    
    public int value() { return poolingTpe; }
 
    PoolingType(int poolingTpe) {
		this.poolingTpe = poolingTpe;
	}
}
