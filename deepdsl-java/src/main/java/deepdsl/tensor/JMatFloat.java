package deepdsl.tensor;
 
import deepdsl.util.ArithStats;

public class JMatFloat extends JMat {
	public final float[] array;

	public JMatFloat(float[] array, int[] dim) {
		super(dim);
		this.array = array;
	}
	public JMatFloat(float[][] arrays, int[] dim) {
		super(dim);
		int colSize = arrays.length;
		int rowSize = arrays[0].length;
		this.array = new float[colSize * rowSize];
		for(int i=0; i<colSize; i++) {
			System.arraycopy(arrays[i], 0, array, i * rowSize, rowSize);
		}
	}

	public JTensorFloat times(JMatFloat m) {
		long begin = System.nanoTime();
		int col1 = columnSize(), col2 = m.columnSize();
		int row1 = array.length/col1, row2 = m.array.length/col2;
		if(row1 != row2) {
			throw new RuntimeException("Matrix dimensions do not match");
		}

		float[] ret = timesSequential(array, m.array, col1, col2, row1); 

		ArithStats.timing("times", begin);
		return new JTensorFloat(ret, addDim(dim, m.dim));
	}

	public static float[] timesSequential(float[] a, float[] b, int col1, int col2, int row) {
		float[] ret = new float[col1*col2];

		for(int i=0; i<col1; i++) {
			int offset1 = i*row;

			for(int j=0; j<col2; j++) {
				int offset2 = j*row;
				float x = 0;

				for(int k=0; k<row; k++) {
					x += a[offset1 + k] * b[offset2 + k];
				}
				ret[i*col2 + j] = x;
			}
		}
		return ret;
	}

	public JTensorFloat max() {
		long begin = System.nanoTime();
		int col = columnSize();
		float[] a = new float[col];
		int row = array.length / col;

		float x;
		for(int i=0; i<col; i++) {
			int start = i*row;
			float max = array[start];
			int end = (i+1)*row;
			for(int j=start+1; j<end; j++) {
				x = array[j];
				max = (max >= x)? max:x;
			}
			a[i] = max;
		}
		ArithStats.timing("max", begin);
		return new JTensorFloat(a, dim);
	}
	public JTensorFloat sum() {
		long begin = System.nanoTime();
		int col = columnSize();
		float[] a = new float[col];
		int row = array.length / col;
 
		for(int i=0; i<col; i++) {
			int start = i*row;
			float sum = 0;
			int end = (i+1)*row;
			for(int j=start; j<end; j++) {
				sum += array[j]; 
			}
			a[i] = sum;
		}
		ArithStats.timing("sum", begin);
		return new JTensorFloat(a, dim);
	}
}
