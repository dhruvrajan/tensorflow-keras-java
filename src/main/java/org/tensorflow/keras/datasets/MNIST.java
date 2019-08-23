// package io.gitlab.keras.datasets;
//
// import io.gitlab.keras.utils.DataUtils;
// import io.gitlab.keras.utils.Keras;
//
// import java.io.DataInputStream;
// import java.io.FileInputStream;
// import java.io.IOException;
// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.List;
// import java.util.zip.GZIPInputStream;
//
//
//
// public class MNIST {
//    // File names and directory structure
//    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
//    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
//    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
//    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";
//
//    private static final String ORIGIN_BASE =  "http://yann.lecun.com/exdb/mnist/";
//
//    private static final String LOCAL_PREFIX = "datasets/mnist";
//
//    // File contents
//    private static final int OUTPUT_CLASSES = 10;
//    private static final int IMAGE_MAGIC = 2051;
//    private static final int LABELS_MAGIC = 2049;
//
//    public static void main(String[] args) throws IOException {
//        MNIST.download();
//        loadData();
//    }
//
//    public static void download() throws IOException {
//        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES,
//                "aaf33078836e3b1c5e40fa8db90a95b7c1a160a013d05dbfa40ee1a50c554be2", "sha256");
//        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS,
//                "fcdfeedb53b53c99384b2cd314206a08fdf6aa97070e19921427a179ea123d19", "sha256");
//        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
//                "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4", "sha256");
//        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
//                "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", "sha256");
//    }
//
//    public static Dataset<List<float[][]>, List<float[][]>> loadData() throws IOException {
//        List<float[][]> TRAIN_IMAGE_PATH = readImages(Keras.kerasPath(LOCAL_PREFIX,
// TRAIN_IMAGES).toString(), 100);
//        List<float[][]> TRAIN_LABEL_PATH = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX,
// TRAIN_LABELS).toString(), 100);
//
//        List<float[][]> TEST_IMAGE_PATH = readImages(Keras.kerasPath(LOCAL_PREFIX ,
// TEST_IMAGES).toString(), 100);
//        List<float[][]> TEST_LABEL_PATH = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX,
// TEST_LABELS).toString(), 100);
//
//        return new Dataset<>(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, TEST_IMAGE_PATH,
// TEST_LABEL_PATH);
//    }
//
//    private static List<float[][]> readImages(String imagesPath, int batchSize) throws IOException
// {
//        try (DataInputStream inputStream =
//                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {
//
//            if (inputStream.readInt() != IMAGE_MAGIC) {
//                throw new IllegalArgumentException("Invalid Image Data File");
//            }
//
//            int numImages = inputStream.readInt();
//            int rows = inputStream.readInt();
//            int cols = inputStream.readInt();
//
//            if (numImages % batchSize != 0) {
//                throw new IllegalArgumentException("Batch Size must divide num elements" +
// numImages + ", " + batchSize);
//            }
//
//            List<float[][]> batches = new ArrayList<>();
//            for (int i = 0; i < numImages / batchSize; i++) {
//                float[][] batch = readImageBuffer(inputStream, batchSize, rows *
// cols).toFloatMatrix();
//                batches.add(batch);
//            }
//
//            return batches;
//        }
//    }
//
//    private static List<float[][]> readLabelsOneHot(String labelsPath, int batchSize) throws
// IOException {
//        try (DataInputStream inputStream =
//                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
//
//            if (inputStream.readInt() != LABELS_MAGIC) {
//                throw new IllegalArgumentException("Invalid Label Data File");
//            }
//
//            int numLabels = inputStream.readInt();
//
//            List<float[][]> batches = new ArrayList<>();
//            for (int i = 0; i < numLabels / batchSize; i++) {
//                batches.add(readLabelBuffer(inputStream, batchSize).toFloatMatrix());
//            }
//
//            return batches;
//        }
//    }
//
//    private static byte[][] readBatchedBytes(DataInputStream inputStream, int batches, int
// bytesPerBatch) throws IOException {
//        byte[][] entries = new byte[batches][bytesPerBatch];
//        for (int i = 0; i < batches; i++) {
//            inputStream.readFully(entries[i]);
//        }
//        return entries;
//    }
//
//    private static INDArray readImageBuffer(DataInputStream inputStream, int numImages, int
// imageSize) throws IOException {
//        byte[][] entries = readBatchedBytes(inputStream, numImages, imageSize);
//
//        float[][] unsignedEntries = new float[numImages][imageSize];
//        for (int i = 0; i < unsignedEntries.length; i++) {
//            for(int j = 0; j < unsignedEntries[0].length; j++) {
//                unsignedEntries[i][j] = (float) (entries[i][j] & 0xFF) / 255.0f;
//            }
//        }
//
//        return Nd4j.create(unsignedEntries);
//    }
//
//    private static INDArray readLabelBuffer(DataInputStream inputStream, int numLabels) throws
// IOException {
//        byte[][] entries = readBatchedBytes(inputStream, numLabels, 1);
//
//        float[][] labels = new float[numLabels][OUTPUT_CLASSES];
//        for (int i = 0; i < entries.length; i++) {
//            labelToOneHotVector(entries[i][0] & 0xFF, labels[i], false);
//        }
//
//        return Nd4j.create(labels);
//    }
//
//    private static void labelToOneHotVector(int label, float[] oneHot, boolean fill) {
//        if (label >= oneHot.length) {
//            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
//        }
//
//        if (fill) Arrays.fill(oneHot, 0);
//        oneHot[label] = 1.0f;
//    }
// }
