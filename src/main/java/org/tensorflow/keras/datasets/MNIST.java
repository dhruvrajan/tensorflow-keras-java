package org.tensorflow.keras.datasets;

import org.tensorflow.Graph;
import org.tensorflow.Tensors;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.data.GraphModeTensorFrame;
import org.tensorflow.keras.utils.DataUtils;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.utils.Pair;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * Code based on example found at:
 * https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
 * <p>
 * Utility for downloading and using MNIST data with a local keras installation.
 */
public class MNIST {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABELS_MAGIC = 2049;
    private static final int OUTPUT_CLASSES = 10;

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final String ORIGIN_BASE = "http://yann.lecun.com/exdb/mnist/";

    private static final String LOCAL_PREFIX = "datasets/mnist/";

    /**
     * Download MNIST dataset files to the local .keras/ directory.
     *
     * @throws IOException when the download fails
     */
    public static void download() throws IOException {
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES,
                "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS,
                "fcdfeedb53b53c99384b2cd314206a08fdf6aa97070e19921427a179ea123d19", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
                "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
                "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", DataUtils.Checksum.sha256);
    }

    public static void main(String[] args) throws IOException {
        try (Graph graph = new Graph()) {

            graphLoaders(graph);
        }
    }

    public static Pair<GraphLoader<Float>, GraphLoader<Float>> graphLoaders(Graph graph) throws IOException {
        // Download MNIST files if they don't exist.
        MNIST.download();

        // Read data files into arrays
        float[][] trainImages = readImages(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString());
        float[][] trainLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString());
        float[][] testImages = readImages(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString());
        float[][] testLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString());

        // Return a pair of graph loaders; train and test sets
        return new Pair<>(
                new GraphModeTensorFrame<>(
                        graph, Float.class, Tensors.create(trainImages), Tensors.create(trainLabels)),
                new GraphModeTensorFrame<>(
                        graph, Float.class, Tensors.create(testImages), Tensors.create(testLabels)));
    }

    public static Pair<GraphLoader<Float>, GraphLoader<Float>> graphLoaders2D(Graph graph) throws IOException {
        // Download MNIST files if they don't exist.
        MNIST.download();

        // Read data files into arrays
        float[][][] trainImages = readImages2D(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString());
        float[][] trainLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString());
        float[][][] testImages = readImages2D(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString());
        float[][] testLabels = readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString());

        // Return a pair of graph loaders; train and test sets
        return new Pair<>(
                new GraphModeTensorFrame<>(
                        graph, Float.class, Tensors.create(trainImages), Tensors.create(trainLabels)),
                new GraphModeTensorFrame<>(
                        graph, Float.class, Tensors.create(testImages), Tensors.create(testLabels)));
    }

    /**
     * Reads MNIST images into an array, given the image datafile path.
     *
     * @param imagesPath MNIST image datafile path
     * @return an array of shape (# examples, # pixels) containing the image data
     * @throws IOException when the file reading fails.
     */
    static float[][] readImages(String imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            return readImageBuffer(inputStream, numImages, rows * cols);
        }
    }

    static float[][][] readImages2D(String imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            return readImageBuffer2D(inputStream, numImages, rows);
        }
    }


    /**
     * Reads MNIST label files into an array, given a label datafile path.
     *
     * @param labelsPath MNIST label datafile path
     * @return an array of shape (# examples, # classes) containing the label data
     * @throws IOException when the file reading fails.
     */
    static float[][] readLabelsOneHot(String labelsPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();
            return readLabelBuffer(inputStream, numLabels);
        }
    }

    public static FloatBuffer readImagesToFloatBuffer(String imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            FloatBuffer images = FloatBuffer.allocate(numImages * rows * cols);
            readLabelBytesToFloatBuffer(inputStream, images, images.capacity());
            return images;
        }
    }

    public static FloatBuffer readLabelsToFloatBuffer(String labelsPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {

            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();

            FloatBuffer labels = FloatBuffer.allocate(numLabels * OUTPUT_CLASSES);
            readLabelBytesToFloatBuffer(inputStream, labels, numLabels);
            return labels;
        }
    }

    public static void readLabelBytesToFloatBuffer(DataInputStream inputStream, FloatBuffer fb, int numBytes) throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numBytes];
        inputStream.readFully(bytes);

        // Convert Bytes to Float Labels
        float[] floats = new float[numBytes * OUTPUT_CLASSES];
        for (int i = 0; i < numBytes; i++) {
            int label = bytes[i] & 0xFF;
            float[] labelOneHot = labelToOneHotVector(label, true);
            for (int j = 0; j < labelOneHot.length; j++) {
                floats[i + j] = labelOneHot[j];
            }
        }
        // Write floats to buffer
        fb.put(floats);
    }

    public static void readImageBytesToFloatBuffer(DataInputStream inputStream, FloatBuffer fb, int numBytes) throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numBytes];
        inputStream.readFully(bytes);

        // Convert Bytes to Floats
        float[] floats = new float[numBytes];
        for (int i = 0; i < numBytes; i++) {
            floats[i] = bytes[i] & 0xFF;
        }

        // Write floats to buffer
        fb.put(floats);
    }

    private static byte[] readBytes(DataInputStream inputStream, int numBytes) throws IOException {
        byte[] bytes = new byte[numBytes];
        inputStream.readFully(bytes);
        return bytes;
    }

    private static byte[][] readBatchedBytes(DataInputStream inputStream, int batches, int bytesPerBatch) throws IOException {
        byte[][] entries = new byte[batches][bytesPerBatch];
        for (int i = 0; i < batches; i++) {
            inputStream.readFully(entries[i]);
        }
        return entries;
    }

    private static float[][][] readImageBuffer2D(
            DataInputStream inputStream, int numImages, int imageWidth) throws IOException {
        float[][][] unsignedEntries = new float[numImages][imageWidth][imageWidth];
        for (int i = 0; i < unsignedEntries.length; i++) {
            byte[][] entries = readBatchedBytes(inputStream, imageWidth, imageWidth);
            for (int j = 0; j < unsignedEntries[0].length; j++) {
                for (int k = 0; k < unsignedEntries[0][0].length; k++) {
                    unsignedEntries[i][j][k] = (float) (entries[j][k] & 0xFF) / 255.0f;
                }
            }
        }
        return unsignedEntries;
    }

    private static float[][] readImageBuffer(DataInputStream inputStream, int numImages, int imageSize) throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numImages, imageSize);
        float[][] unsignedEntries = new float[numImages][imageSize];
        for (int i = 0; i < unsignedEntries.length; i++) {
            for (int j = 0; j < unsignedEntries[0].length; j++) {
                unsignedEntries[i][j] = (float) (entries[i][j] & 0xFF) / 255.0f;
            }
        }

        return unsignedEntries;
    }

    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels)
            throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numLabels, 1);

        float[][] labels = new float[numLabels][OUTPUT_CLASSES];
        for (int i = 0; i < entries.length; i++) {
            labelToOneHotVector(entries[i][0] & 0xFF, labels[i], false);
        }

        return labels;
    }

    private static void labelToOneHotVector(int label, float[] oneHot, boolean fill) {
        if (label >= oneHot.length) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        if (fill) Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
    }

    private static float[] labelToOneHotVector(int label, boolean fill) {
        float[] oneHot = new float[OUTPUT_CLASSES];
        if (label >= oneHot.length) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        if (fill) Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
        return oneHot;
    }
}
