package deepdsl.cudnn.config;

import jcuda.jcudnn.cudnnDataType;

public enum DataType {
	FLOAT (cudnnDataType.CUDNN_DATA_FLOAT), 
	DOUBLE (cudnnDataType.CUDNN_DATA_DOUBLE), 
	//The data is 16-bit floating point.
	HALF (cudnnDataType.CUDNN_DATA_HALF);
	
	private final int dataType;
    
    public int tpe() { return dataType; }
 
	DataType(int dataType) {
		this.dataType = dataType;
	}
}
