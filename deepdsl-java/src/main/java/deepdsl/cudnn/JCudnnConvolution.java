package deepdsl.cudnn;

import static jcuda.jcudnn.JCudnn.*; 
import jcuda.Pointer;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor; 
import deepdsl.cudnn.config.ConvolutionMode;
import deepdsl.cudnn.config.ConvolutionPreference;
import deepdsl.cudnn.config.ConvolutionType;
import deepdsl.cudnn.config.TensorFormat;
import deepdsl.util.ArithStats; 

public class JCudnnConvolution extends JCudaFunction {
	// limit = -1 unlimited workspace
	// limit = 0 no workspace
	// otherwise limited workspace
	public static long limit = -1; // 3 * 1000_000_000; // 3000 MB
	
	private cudnnFilterDescriptor filter_dptr;
	private cudnnConvolutionDescriptor convolv_dptr; 

	private int[] padding_array =  new int[] { 0, 0 };;
	private int[] stride_array = new int[] { 1, 1 }; // TODO: stride 1 for 2 dimensions?
	private int[] upscale = new int[] { 1, 1 };
	private ConvolutionType convType = ConvolutionType.TwoD;
	private ConvolutionMode mode = ConvolutionMode.CROSS_CORRELATION;
	 
	int[] x_dims, y_dims, w_dims, b_dims;
	JCudnnDescriptor x_dptr, y_dptr, b_dptr;
	int forward_algorithm, backward_data_algorithm, backward_filter_algorithm;
 
	private long forwardSize = 0, backwardDataSize = 0, backwardFilterSize = 0; 

	public JCudnnConvolution(int[] x_dims, int[] w_dims, int[] b_dims) {
		this(x_dims, w_dims, b_dims, 1, 0);
	}
			
	public JCudnnConvolution(int[] x_dims, int[] w_dims, int[] b_dims, int stride, int padding) {
		this.stride_array = new int[] {stride, stride};
		this.padding_array = new int[] {padding, padding}; // (x_dim(3) + w_dim(3) + padding - stride) / stride > 0 
		
		this.x_dims = x_dims;
		this.w_dims = w_dims;
		this.b_dims = b_dims;  
		
		x_dptr = new JCudnnDescriptor(x_dims);
		filter_dptr = new cudnnFilterDescriptor();
		convolv_dptr = new cudnnConvolutionDescriptor();

		checkError(cudnnCreateFilterDescriptor(filter_dptr));
		checkError(cudnnCreateConvolutionDescriptor(convolv_dptr));
		
		checkError(cudnnSetFilter4dDescriptor(filter_dptr, JCudnnDescriptor.dataType.value(), TensorFormat.NCHW.value(), 
				w_dims[0], w_dims[1], w_dims[2], w_dims[3])); 
//		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], stride[0], stride[1], upscale[0], upscale[1], mode); 
		// TODO: why not use 2-d?
		checkError(cudnnSetConvolutionNdDescriptor(convolv_dptr, convType.value(), padding_array, stride_array, 
				upscale, mode.value(), JCudnnDescriptor.dataType.value()));
		
		y_dims = new int[x_dims.length];
		checkError(cudnnGetConvolutionNdForwardOutputDim(convolv_dptr, x_dptr.descriptor, filter_dptr, x_dims.length, y_dims));
		
		y_dptr = new JCudnnDescriptor(y_dims);
		b_dptr = new JCudnnDescriptor(b_dims);

		selectAlgorithm();
		findWorkspaceSize();
		reserveWorkspace();
	}
	
	public long[] workspaceSize() { return new long[] { forwardSize, backwardDataSize, backwardFilterSize }; }
	
	private void selectAlgorithm() {
		// Choose the best according to the preference

		int algoArray[] = {0};
		checkError(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, x_dptr.descriptor, filter_dptr, convolv_dptr, y_dptr.descriptor, 
				limit != 0? 
						(limit < 0? ConvolutionPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST : ConvolutionPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
						: ConvolutionPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, limit, algoArray));
		forward_algorithm =  algoArray[0];  // 1 is the best
		checkError(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filter_dptr, y_dptr.descriptor, convolv_dptr, x_dptr.descriptor, 
				limit != 0? 
						(limit < 0? ConvolutionPreference.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST : ConvolutionPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
						: ConvolutionPreference.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, limit, algoArray));
		backward_data_algorithm = algoArray[0]; // 0 is the best
		checkError(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, x_dptr.descriptor, y_dptr.descriptor, convolv_dptr, filter_dptr, 
				limit != 0? 
						(limit < 0? ConvolutionPreference.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST : ConvolutionPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
						: ConvolutionPreference.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, limit, algoArray));
		backward_filter_algorithm =  algoArray[0]; // 1 is the best
	}
	
	private void findWorkspaceSize() {
		long sizeInBytesArray[] = {0}; 
		
		checkError(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, x_dptr.descriptor, filter_dptr, convolv_dptr, y_dptr.descriptor, forward_algorithm, sizeInBytesArray)); 
		forwardSize = sizeInBytesArray[0]; 
		 
		checkError(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filter_dptr, y_dptr.descriptor, convolv_dptr, x_dptr.descriptor, backward_data_algorithm, sizeInBytesArray));
        backwardDataSize = sizeInBytesArray[0];  
         
		checkError(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, x_dptr.descriptor, y_dptr.descriptor, convolv_dptr, filter_dptr, backward_filter_algorithm, sizeInBytesArray));
		backwardFilterSize = sizeInBytesArray[0]; 
	}
	
	private void reserveWorkspace() { 
		long size = 0;
		size = size >= forwardSize ? size : forwardSize;
		size = size >= backwardDataSize ? size : backwardDataSize;
		size = size >= backwardFilterSize ? size : backwardFilterSize;
		
		if(size != 0) { reserveWorkspace(size); } 
	}
 
	public void free() {
		cudnnDestroyConvolutionDescriptor(convolv_dptr);
		cudnnDestroyFilterDescriptor(filter_dptr); 
		x_dptr.free();
		y_dptr.free();
		b_dptr.free();
	}
	// y = forward(x, w, b)	
	public JCudaTensor forward(JCudaTensor x, JCudaTensor w, JCudaTensor b) {  
		return forward(x, w, b, new JCudaTensor(y_dims), one, zero); 
	}
	// y = forward(x, w, b) * alpha + y * beta
	public JCudaTensor forward(JCudaTensor x, JCudaTensor w, JCudaTensor b, JCudaTensor y, Pointer alpha, Pointer beta) { 
		int algorithm = forward_algorithm; 

        long begin = System.nanoTime(); 
        
		int ret = cudnnConvolutionForward(cudnnHandle, alpha, x_dptr.descriptor, x.getData(),
				filter_dptr, w.getData(), convolv_dptr, algorithm,
				allocWorkspace(forwardSize), forwardSize, beta, y_dptr.descriptor, y.getData());
		deallocWorkspace();
		checkError(ret);
		
        ret = cudnnAddTensor(cudnnHandle, alpha, b_dptr.descriptor, b.getData(), one, y_dptr.descriptor, y.getData());
 
        checkError(ret);
        
		ArithStats.cuda_timing("convolution forward", begin);

		return y;
	}
	// dx = backward_data(dy, w)
	public JCudaTensor backward_data(JCudaTensor dy, JCudaTensor w) {
		JCudaTensor dx = new JCudaTensor(x_dims) ;  
		return backward_data(dy, w, dx, one, zero);
	}
	// dx += backward_data(dy, w)
	public JCudaTensor backward_data(JCudaTensor dy, JCudaTensor w, JCudaTensor dx) {
		return backward_data(dy, w, dx, one, one);
	}
	// dx = backward_data(dy, w) * alpha + dx * beta
	public JCudaTensor backward_data(JCudaTensor dy, JCudaTensor w, JCudaTensor dx, Pointer alpha, Pointer beta) {  
        long begin = System.nanoTime();
		int algorithm = backward_data_algorithm; 
  
        int ret = cudnnConvolutionBackwardData(cudnnHandle, alpha, filter_dptr, w.getData(), y_dptr.descriptor, dy.getData(), convolv_dptr, 
        		algorithm, allocWorkspace(backwardDataSize), backwardDataSize, beta, x_dptr.descriptor, dx.getData());
        deallocWorkspace();
        checkError(ret);
 
		ArithStats.cuda_timing("convolution backward data", begin);
		return dx;
	}
	// dw = backward_filter(dy, x)
	public JCudaTensor backward_filter(JCudaTensor dy, JCudaTensor x) {
		return backward_filter(dy, x, new JCudaTensor(w_dims), one, zero);
	}
	
	public JCudaTensor backward_filter(JCudaTensor dy, JCudaTensor x, JCudaTensor dw, float alpha, float beta) {
		return backward_filter(dy, x, dw, pointerTo(alpha), pointerTo(beta));
	}
 			
	// dw = backward_filter(dy, x) * alpha + dw * beta
	public JCudaTensor backward_filter(JCudaTensor dy, JCudaTensor x, JCudaTensor dw, Pointer alpha, Pointer beta) {  
        long begin = System.nanoTime();
		int algorithm = backward_filter_algorithm; 
        
        int ret = cudnnConvolutionBackwardFilter(cudnnHandle, alpha, x_dptr.descriptor, x.getData(), y_dptr.descriptor, dy.getData(), convolv_dptr, 
        		algorithm, allocWorkspace(backwardFilterSize), backwardFilterSize, beta, filter_dptr, dw.getData());   
        deallocWorkspace();
        checkError(ret);
		ArithStats.cuda_timing("convolution backward filter", begin);
		return dw;
	}
	// db = backward_bias(dy)
	public JCudaTensor backward_bias(JCudaTensor dy) {
		return backward_bias(dy, new JCudaTensor(b_dims), one, zero);
	}
	
	public JCudaTensor backward_bias(JCudaTensor dy, JCudaTensor db, float alpha, float beta) {
		return backward_bias(dy, db, pointerTo(alpha), pointerTo(beta));
	}
	
	// db = backward_bias(dy) * alpha + db * beta
	public JCudaTensor backward_bias(JCudaTensor dy, JCudaTensor db, Pointer alpha, Pointer beta) {
        long begin = System.nanoTime(); 
        
        int ret = cudnnConvolutionBackwardBias(cudnnHandle, alpha, y_dptr.descriptor, dy.getData(), beta, b_dptr.descriptor, db.getData());
        checkError(ret);
        
		ArithStats.cuda_timing("convolution backward bias", begin);
		return db;
	}
}
