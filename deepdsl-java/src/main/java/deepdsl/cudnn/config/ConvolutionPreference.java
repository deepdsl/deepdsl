package deepdsl.cudnn.config;
 

public class ConvolutionPreference { 
    public static final int CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0;
    public static final int CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1;
    public static final int CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2;
	
    public static final int CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1;
    public static final int CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2;
	
    public static final int CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0;
    public static final int CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1;
    public static final int CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2;
}
