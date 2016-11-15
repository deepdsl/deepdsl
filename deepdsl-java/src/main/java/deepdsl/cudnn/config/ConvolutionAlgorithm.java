package deepdsl.cudnn.config;
 
import jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo;
import jcuda.jcudnn.cudnnConvolutionBwdDataAlgo;
import jcuda.jcudnn.cudnnConvolutionFwdAlgo;

public enum ConvolutionAlgorithm {

	FORWARD_ALGORITHM_IMPLICIT_GEMM(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),

	FORWARD_ALGORITHM_IMPLICIT_PRECOMP_GEMM(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM),

	FORWARD_ALGORITHM_GEMM(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM),

	FORWARD_ALGORITHM_DIRECT(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT),

	FORWARD_ALGORITHM_FFT(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT),

	FORWARD_ALGORITHM_FFT_TILING(
			cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING),

	BACKWARD_FILTER_ALGORITHM_0(
			cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0),

	BACKWARD_FILTER_ALGORITHM_1(
			cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1),

	BACKWARD_FILTER_ALGORITHM_FFT(
			cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT),

	BACKWARD_DATA_ALGORITHM_0(
			cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0),

	BACKWARD_DATA_ALGORITHM_1(
			cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1),

	BACKWARD_DATA_ALGORITHM_FFT(
			cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT),

	BACKWARD_DATA_ALGORITHM_FFT_TILING(
			cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);

	private final int algorithm;

	public int value() {
		return algorithm;
	}

	ConvolutionAlgorithm(int algorithm) {
		this.algorithm = algorithm;
	}
}
