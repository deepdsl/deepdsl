package deepdsl.cudnn;
 
import java.util.HashMap;
import java.util.Map;

import deepdsl.tensor.JMatFloat;
import deepdsl.util.ArithStats; 
import jcuda.Pointer; 
import jcuda.jcublas.JCublas2; 
import jcuda.vec.VecFloat; 
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;

public class JCudaMatrix extends JCudaFunction {
	final Pointer data;
	final int[] dims;
	final int numOfRows, numOfColumns;
	boolean transposed;  
	
	final int size;
	
	public JCudaMatrix(JMatFloat mat) {
		this(new Pointer(), mat.dim, mat.array.length/mat.columnSize(), true);
		int size = mat.array.length; 
		allocFloat(data, size);
		copyHostToDevice(mat.array, data); 
	}

	public JCudaMatrix(Pointer data, int[] dims, int numOfColumns, boolean transposed) {
		this.data = data;
		this.dims = dims;
		this.numOfRows = numOfRows();
		this.numOfColumns = numOfColumns;
		this.transposed = transposed;
		this.size = numOfRows * numOfColumns;
	}
	
	private int numOfRows() {
		int size = 1;
		for(int i=0; i<dims.length; i++) { size = size * dims[i]; }
		return size;
	}

	// TODO: too lazy to implement transpose for now.
	public JMatFloat asJMat() {
		if(transposed) return new JMatFloat(asArray(), dims);
		else throw new RuntimeException("don't support copying-out column-major cuda matrix yet");
	}
	
	// This is expensive as it copies to the host, computes max, and then copies to the device.
	public JCudaTensor max() {
		return asJMat().max().asJCudaTensor();
	}
	
    /* The method is for testing only. 
     * It should not be called since it is always constructed from JCudaTensor 
     * */
	public void free() { cudaFree(data); }
	
	public float[] asArray() {
		int size = this.numOfColumns*this.numOfRows;
		float[] ret = new float[size];
		copyDeviceToHost(data, ret); 
		
		return ret;
	}

	private int getTransposeFlag(boolean transpose) {
		return (this.transposed ^ transpose) ? CUBLAS_OP_T: CUBLAS_OP_N;
	}
	
	// strides of the matrix = row size if transposed (row major) and column size otherwise.
	int leadingDimension() {
		return this.transposed ? this.numOfColumns: this.numOfRows;
	} 
 
	// this = this * beta + that * alpha
	public JCudaMatrix update(JCudaMatrix that, float alpha, float beta) { 
		long begin = System.nanoTime();
		if(this.size != that.size) {
			throw new RuntimeException("matrix sizes do not match");
		}

		if (beta != 1) {
			VecFloat.mulScalar(size, data, data, beta);
		}
		if (alpha != 1) {
			int ret = JCublas2.cublasSaxpy(cublasHandle, size, pointerTo(alpha), that.data, 1, data, 1); 
			checkError(ret);
		}
		else {
			VecFloat.add(size, data, data, that.data);
		}
 
		ArithStats.cuda_timing("cuda update", begin);
		return this;
	}
	
	// this x that' (row-major) = that x this' (column-major)
	public JCudaTensor times(JCudaMatrix that) {  
		JCudaTensor ret = that.times_direct(this, new int[]{this.numOfRows, that.numOfRows});  
		return ret;
	}
	
	// ret = ret * beta + (this x that') * alpha (row-major).  
	public JCudaTensor times(JCudaMatrix that, JCudaTensor ret, float alpha, float beta) {  
		that.times_direct(this, ret, pointerTo(alpha), pointerTo(beta)); 
		return ret;
	}
	// this x that' (column-major)
	private JCudaTensor times_direct(JCudaMatrix that, int[] dims) {
		return times_direct(that, new JCudaTensor(dims), one, zero);
	}
	// this x that' (column-major)
	private JCudaTensor times_direct(JCudaMatrix that, JCudaTensor ret, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		
		int m = this.numOfRows, n = that.numOfRows, k = this.numOfColumns;
		int lda = this.leadingDimension(), ldb = that.leadingDimension(), ldc = m;

		if(k != that.numOfColumns) {
			throw new RuntimeException("matrix dimensions do not match");
		}
		Pointer d_A = this.data, d_B = that.data;
		 
		Pointer d_C = ret.getData();

		int transpose_A = getTransposeFlag(false), transpose_B = that.getTransposeFlag(true);

		// Execute sgemm
		int flag = JCublas2.cublasSgemm(cublasHandle, transpose_A, transpose_B, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
		checkError(flag);
		
		ArithStats.cuda_timing("cuda times", begin);
		return ret; // result is column major already
	}
	
	// sum each row and return a column vector of size "numOfRows"
	public JCudaTensor sum() {  
		return times_direct(constantVector(1, numOfColumns, 1f), new int[]{numOfRows}); 
	}
	// ret = beta * ret + alpha * this.sum()
	public JCudaTensor sum(JCudaTensor ret, float alpha, float beta) {  
		return times_direct(constantVector(1, numOfColumns, 1f), ret, pointerTo(alpha), pointerTo(beta));  
	}
	
	// making n copies and return n x numOfRows matrix (row-major)
	public JCudaTensor copy(int n) { 
		if(numOfColumns != 1) {
			throw new RuntimeException("only column vector can be copied");
		}
		int[] new_dims = new int[dims.length+1];
		
		new_dims[0] = n;
		for(int i=1; i<dims.length+1; i++) {
			new_dims[i] = dims[i-1];
		}
		JCudaTensor ret = times_direct(constantVector(n, 1, 1f), new_dims);  
		return ret;
	}
	
	public JCudaTensor copy(int n, JCudaTensor ret) {
		return times_direct(constantVector(n, 1, 1f), ret, one, one);
	}
	
	// constant matrix of value "value"
	public static JCudaMatrix constantVector(int numOfRows, int numOfColumns, float value) {
		String key = numOfRows + " " + numOfColumns + " " + value;
		JCudaMatrix ret = constants.get(key);
		int size = numOfRows*numOfColumns;
		
		if (ret == null) {
			Pointer pointer = new Pointer();
			allocFloat(pointer, size);
			
			VecFloat.set(size, pointer, value);
			ret = new JCudaMatrix(pointer, new int[]{ numOfRows }, numOfColumns, false);
			constants.put(key, ret);
		}
		return ret;
	}
	// caching constant matrices
	static Map<String, JCudaMatrix> constants = new HashMap<>();
}
