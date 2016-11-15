package deepdsl.tensor;

public class JTensorFloatTuple {
    public final JTensorFloat image;
    public final JTensorFloat label;

    public JTensorFloatTuple(JTensorFloat image, JTensorFloat label) {
        this.image = image;
        this.label = label;
    }

    public JTensorFloat getImage() {
        return image;
    }

    public JTensorFloat getLabel() {
        return label;
    }
}
