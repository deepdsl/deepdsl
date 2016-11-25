package deepdsl.data.imagenet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * This is a class for image collection, which contains a collection of images, a collection of dimensions of each image, and a collection of image labels
 * This class is made serializable so that the Imagenet data can be serialized to / from the disk.
 *
 * Note the collection size here may or may not be same as the batch size for deep learning. Normally it is
 * larger (for example, this collection size here can be 1024 so that we don't need to store that many files for
 * Imagenet data (while the typical batch of deep learning can be 16/32/64/128 or so). Such difference is handled
 * transparently inside #deepdsl.data.imagenet.ImagenetDataHandler.
 */
public class ImageCol implements Serializable {

    private static final long serialVersionUID = -1L;

    private final List<byte[]> images = new ArrayList<>();
    private final List<int[]> dims = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();

    public void addImageAndLabel(byte[] image, int[] dim, Integer label) {
        images.add(image);
        dims.add(dim);
        labels.add(label);
    }

    public byte[] getImage(int index) {
        return images.get(index);
    }

    public int[] getDim(int index) {
        return dims.get(index);
    }

    public Integer getLabel(int index) {
        return labels.get(index);
    }
}
