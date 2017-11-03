package deepdsl.data.imagenet;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * The class, handles serialization and resizing of all raw Imagenet data to disk in a series of files. Each file contains a collection
 * of image data (See #deepdsl.data.imagenet.ImageCol for details) when used standalone. It will deserialize
 * the files and load image one at a time with the nextTuple(#cursor) method call (caching and correct image indexing is handled transparently
 * for performance), when used by internal data loading structure, e.g. #deepdsl.tensor.ImagenetFactory.
 *
 */
public class ImagenetDataHandler {

    private static final String FILE_SEPARATOR = "/";
    private int height;
    private int width;
    private Pair[] pairs;
    private int numTotalItems;
    private String pathToDataset;
    private String pathToNewDataset;
    private String pathToGroundTruthFile;
    private int batchSize;
    private final java.util.List<Path> batchFiles = new ArrayList<>();
    private ImageCol currImageCol;
    private final Map<Integer, ImageCol> cachedBatchMap = new HashMap<>();

    /**
     * The constructor to process the original Imagenet data, this constructor is only used when you run Main in this class as a standalone program
     *
     * @param height
     * @param width
     * @param pathToDataset
     * @param pathToNewDataset
     * @param pathToGroundTruthFile
     * @param batchSize
     */
    public ImagenetDataHandler(int height, int width, String pathToDataset, String pathToNewDataset, String pathToGroundTruthFile, int batchSize) {
        this.height = height;
        this.width = width;
        this.pathToDataset = pathToDataset;
        this.pathToNewDataset = pathToNewDataset;
        this.pathToGroundTruthFile = pathToGroundTruthFile;
        this.batchSize = batchSize;
    }

    /**
     * The constructor for training / testing
     * Since we need to use the # of files (# of batches in other words), we store that in a local variable
     *
     * @param pathToNewDataset
     * @param pathToGroundTruthFile
     * @param batchSize
     */
    public ImagenetDataHandler(String pathToNewDataset, String pathToGroundTruthFile, int batchSize) {
        this.pathToNewDataset = pathToNewDataset;
        this.pathToGroundTruthFile = pathToGroundTruthFile;
        this.batchSize = batchSize;
        setBatchFiles();
    }


    public static void main (String[] argv) throws Exception {
        if (argv.length != 6) {
            System.out.println("USAGE: java deepdsl.data.imagenet.ImagenetDataHandler resize_image_height resize_image_width raw_image_path output_image_path ground_truth_file_path batch_size");
            System.out.println("e.g. java deepdsl.data.imagenet.ImagenetDataHandler 256 256 ~/tmp/ILSVRC2012_img_train ~/tmp1/ILSVRC2012_img_train dataset/imagenet/train.txt 512");
        }
        else {
            ImagenetDataHandler handler = new ImagenetDataHandler(Integer.parseInt(argv[0]), Integer.parseInt(argv[1]), argv[2], argv[3], argv[4], Integer.parseInt(argv[5]));
//            Uncomment the below two lines to prepare the dataset
            handler.prepareImagenetData();
            handler.resizeAndPersistAll();

//            Testing with 2049 images that have been stored in 3 batches (batch_size = 1024)
            handler.setBatchFiles();
            for (int i = 0; i < 2049; i++) {
                handler.nextTuple(i);
            }

        }
    }

    /**
     * @return the tuple at the current cursor
     */
    public Tuple nextTuple(int cursor) {
        int batchNum = (cursor / batchSize);
        int inBatchInex = (cursor - batchNum * batchSize);
        try {
            if (!checkOpened(batchNum)) {
                currImageCol = loadBatch(batchNum);
                //This enables always only one <key, value> pair in the map
                cachedBatchMap.clear();
                cachedBatchMap.put(batchNum, currImageCol);
            } else currImageCol = cachedBatchMap.get(batchNum);

            byte[] imageBytes = currImageCol.getImage(inBatchInex);
            int[] dim = currImageCol.getDim(inBatchInex);
            Integer label = currImageCol.getLabel(inBatchInex);
            return new Tuple(imageBytes, label, dim);
        } catch (ClassNotFoundException | IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    /**
     * The class to reflect the Imagenet groundtruth data
     * check dataset/imagenet/train.txt or val.txt for mapping
     */
    private class Pair {
        private String path;
        private int label;

        Pair(final String path, final int label) {
            this.path = path;
            this.label = label;
        }

        public String getPath() {
            return path;
        }

        public int getLabel() {
            return label;
        }

    }

    /**
     * The Java 8 file handling using DirectoryStream
     *
     */
    private void setBatchFiles() {
        Path path = Paths.get(pathToNewDataset);
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
            for (Path entry: stream) {
                batchFiles.add(entry);
            }
        } catch (DirectoryIteratorException | IOException ex) {
            throw new RuntimeException(ex);
        }
        //The beow is used to show how many batches are available
        //batchFiles.forEach( file -> System.out.println(file));
    }

    private void prepareImagenetData() throws IOException {
        pairs = Files.lines(Paths.get(pathToGroundTruthFile), Charset.defaultCharset())
                .map(s -> s.split(" "))
                .map(ar -> new Pair(ar[0], Integer.parseInt(ar[1])))
                .collect(Collectors.toList()).toArray(new Pair[]{});
        shuffleArray(pairs);
        numTotalItems = pairs.length;
        System.out.println("numTotalItems" + numTotalItems);
    }

    private void resizeAndPersistAll() throws IOException {
        ImageCol imageCol = new ImageCol();
        int index = 0;
        for (int i = 0; i < numTotalItems; i++) {
            BufferedImage originalImage;
            try {
                originalImage = ImageIO.read(new File(pathToDataset + FILE_SEPARATOR + pairs[i].getPath()));
                ++index;
            } catch (Exception ex) {
                System.out.println("Image reading exception for " + pathToDataset + FILE_SEPARATOR + pairs[i].getPath() + ", skip it, exception is: " + ex.getMessage());
                continue;
            }
            int type = originalImage.getType() == 0 ? BufferedImage.TYPE_INT_ARGB : originalImage.getType();
            BufferedImage resizeImageJpg = resizeImage(originalImage, type);
            int[] dim = new int[3];
            int imageType = resizeImageJpg.getType();
            if (BufferedImage.TYPE_BYTE_GRAY == imageType || BufferedImage.TYPE_USHORT_GRAY == imageType) {
                dim[0] = 1;
            } else dim[0] = 3;
            dim[1] = resizeImageJpg.getHeight();
            dim[2] = resizeImageJpg.getWidth();
            byte[] imageBytes = ((DataBufferByte) resizeImageJpg.getRaster().getDataBuffer()).getData();
            imageCol.addImageAndLabel(imageBytes, dim, pairs[i].getLabel());
            if (index % 500 == 0) System.out.println(index + " resized image(" + i + ")");
            if (index % batchSize == 0) {
                persistBatch(imageCol, index / batchSize);
                imageCol = new ImageCol();
            }
        }
        if ( index % batchSize != 0) {
            persistBatch(imageCol, index / batchSize + 1);
        }
    }

    /**
     * Fisher–Yates shuffle
     */
    private void shuffleArray(Pair[] ar)
    {
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            Pair a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    /**
     *
     * @param originalImage
     * @param type
     * @return
     */
    private BufferedImage resizeImage(BufferedImage originalImage, int type){
        BufferedImage resizedImage = new BufferedImage(width, height, type);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return resizedImage;
    }

    private void persistBatch(ImageCol imageCol, int count) throws IOException {
        String pathToNewFile = pathToNewDataset + FILE_SEPARATOR + count;
        Path path = Paths.get(pathToNewFile).getParent();
        if (Files.notExists(path)) {
            Files.createDirectories(path);
        }
        FileOutputStream fout = new FileOutputStream(pathToNewFile);
        ObjectOutputStream oos = new ObjectOutputStream(fout);
        oos.writeObject(imageCol);
        oos.close();
        System.out.println("Persisted batch " + count);
    }

    private ImageCol loadBatch(int index) throws IOException, ClassNotFoundException {
        FileInputStream fin = new FileInputStream(batchFiles.get(index).toFile());
        ObjectInputStream in = new ObjectInputStream(fin);
        ImageCol imageCol = (ImageCol) in.readObject();
        in.close();
        System.out.println("load batch " + index);
        return imageCol;
    }

    private boolean checkOpened(int batchNum) {
        if (cachedBatchMap.containsKey(batchNum)) return true; else  return false;
    }
}
