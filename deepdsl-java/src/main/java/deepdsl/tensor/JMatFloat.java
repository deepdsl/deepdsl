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

	public int[] maxIndex() { 
		int col = columnSize();
		int[] a = new int[col];
		int row = array.length / col; 
		
		for(int i=0; i<col; i++) {
			a[i] = maxIndex(array, i*row, row);
		}
		return a;
	}
	private static int maxIndex(float[] array, int start, int size) {
		int ret = 0;
		float max = array[start];
		
		for(int i=1; i<size; i++) {
			int j = start + i;
			if(max < array[j]) {
				max = array[j];
				ret = i;
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
