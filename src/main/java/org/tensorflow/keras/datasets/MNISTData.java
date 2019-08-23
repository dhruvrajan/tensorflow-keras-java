// package io.gitlab.keras.datasets;
//
//
// import java.io.DataInputStream;
// import java.io.FileInputStream;
// import java.io.IOException;
// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.List;
// import java.util.zip.GZIPInputStream;
//
/// **
// * Code based on example found at:
// * -
// https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
// */
//
// public class MNISTData {
//    private static final int IMAGE_MAGIC = 2051;
//    private static final int LABELS_MAGIC = 2049;
//    private static final int OUTPUT_CLASSES = 10;
//    public static final int BATCH_SIZE = 100;
//    public static String TEST_IMAGE_PATH =
// "C:\\Users\\dhruv\\data\\mnist\\t10k-images-idx3-ubyte.gz";
//    public static String TRAIN_IMAGE_PATH =
// "C:\\Users\\dhruv\\data\\mnist\\train-images-idx3-ubyte.gz";
//    public static String TEST_LABEL_PATH =
// "C:\\Users\\dhruv\\data\\mnist\\t10k-labels-idx1-ubyte.gz";
//    public static String TRAIN_LABEL_PATH =
// "C:\\Users\\dhruv\\data\\mnist\\train-labels-idx1-ubyte.gz";
//
//
//    public static void main(String[] args) throws IOException {
//
//    }
//
//    public static Dataset<List<float[][]>, List<float[][]>> loadMNISTDataset() throws IOException
// {
//        return  MNISTData.loadMNISTDataset(MNISTData.TRAIN_IMAGE_PATH, MNISTData.TRAIN_LABEL_PATH,
// MNISTData.TEST_IMAGE_PATH, MNISTData.TEST_LABEL_PATH);
//    }
//
//    public static Dataset<List<float[][]>, List<float[][]>> loadMNISTDataset(
//            String trainImagesPath, String trainLabelsPath,
//            String testImagesPath, String testLabelsPath) throws IOException {
//
//        List<float[][]> TRAIN_IMAGE_PATH = readImages(trainImagesPath, BATCH_SIZE);
//        List<float[][]> TRAIN_LABEL_PATH = readLabelsOneHot(trainLabelsPath, BATCH_SIZE);
//        List<float[][]> TEST_IMAGE_PATH = readImages(testImagesPath, 10000);
//        List<float[][]> TEST_LABEL_PATH = readLabelsOneHot(testLabelsPath, 10000);
//
//        return new Dataset<float[], float[]>(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, TEST_IMAGE_PATH,
// TEST_LABEL_PATH);
//    }
//
//
//    private static List<float[][]> readImages(String imagesPath, int batchSize) throws IOException
// {
//
//        try (DataInputStream inputStream =
//                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {
//
//            if (inputStream.readInt() != IMAGE_MAGIC) {
//                throw new IllegalArgumentException("Invalid Image Data File");
//            }
//
//
//            int numImages = inputStream.readInt();
//            int rows = inputStream.readInt();
//            int cols = inputStream.readInt();
//
//
//
//            if (numImages % batchSize != 0) {
//                throw new IllegalArgumentException("Batch Size must divide num elements" +
// numImages + ", " + batchSize);
//            }
//
//            List<float[][]> batches = new ArrayList<>();
//            for (int i = 0; i < numImages / batchSize; i++) {
//                batches.add(readImageBuffer(inputStream, batchSize, rows * cols));
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
//                batches.add(readLabelBuffer(inputStream, batchSize));
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
//
//    private static float[][] readImageBuffer(DataInputStream inputStream, int numImages, int
// imageSize) throws IOException {
//        byte[][] entries = readBatchedBytes(inputStream, numImages, imageSize);
//        float[][] unsignedEntries = new float[numImages][imageSize];
//        for (int i = 0; i < unsignedEntries.length; i++) {
//            for(int j = 0; j < unsignedEntries[0].length; j++) {
//                unsignedEntries[i][j] = (float) (entries[i][j] & 0xFF) / 255.0f;
//            }
//        }
//
//        return unsignedEntries;
//    }
//
//    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels) throws
// IOException {
//        byte[][] entries = readBatchedBytes(inputStream, numLabels, 1);
//
//        float[][] labels = new float[numLabels][OUTPUT_CLASSES];
//        for (int i = 0; i < entries.length; i++) {
//            labelToOneHotVector(entries[i][0] & 0xFF, labels[i], false);
//        }
//
//        return labels;
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
