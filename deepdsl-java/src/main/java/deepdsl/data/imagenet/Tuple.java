package deepdsl.data.imagenet;

public class Tuple {
	public final int[] dims; // channel x width x height
    public final byte[] image;
    public final int label;

    public Tuple(byte[] image, int label, int[] dims) {
        this.image = image;
        this.label = label;
        this.dims = dims;
    } 
}
