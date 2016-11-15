package deepdsl.tensor;
 
public abstract class JMat {
	public final int[] dim;
	
	protected JMat(int[] dim) {
		this.dim = dim;
	}
	
	public int columnSize() { 
		int ret = 1;
		for(int i : dim) { ret *= i; } 
		return ret;
	} 

	public static int[] addDim(int[] dim1, int[] dim2) { 
		int[] ret = new int[dim1.length + dim2.length];
		for(int i=0; i<dim1.length; i++) {
			ret[i] = dim1[i];
		}
		for(int i=0; i<dim2.length; i++) {
			ret[i+dim1.length] = dim2[i];
		} 
		return ret;
	}
}
