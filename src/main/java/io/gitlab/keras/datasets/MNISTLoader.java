package io.gitlab.keras.datasets;


import io.gitlab.keras.data.CompactTensorSplit;
import io.gitlab.keras.data.TensorDataset;
import io.gitlab.keras.data.TensorSplit;
import org.tensorflow.Tensors;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * Code based on example found at:
 * - https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
 */

public class MNISTLoader {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABELS_MAGIC = 2049;
    private static final int OUTPUT_CLASSES = 10;

    private static String TEST_IMAGE_PATH = "C:\\Users\\dhruv\\data\\mnist\\t10k-images-idx3-ubyte.gz";
    private static String TRAIN_IMAGE_PATH = "C:\\Users\\dhruv\\data\\mnist\\train-images-idx3-ubyte.gz";
    private static String TEST_LABEL_PATH = "C:\\Users\\dhruv\\data\\mnist\\t10k-labels-idx1-ubyte.gz";
    private static String TRAIN_LABEL_PATH = "C:\\Users\\dhruv\\data\\mnist\\train-labels-idx1-ubyte.gz";


    public static TensorDataset<Float> loadData() throws IOException {
        float[][] trainImages = readImages(TRAIN_IMAGE_PATH);
        float[][] trainLabels = readLabelsOneHot(TRAIN_LABEL_PATH);
        float[][] testImages = readImages(TEST_IMAGE_PATH);
        float[][] testLabels = readLabelsOneHot(TEST_LABEL_PATH);

        return new TensorDataset<>(
                new CompactTensorSplit<>(Tensors.create(trainImages), Tensors.create(trainLabels), Float.class),
                new CompactTensorSplit<>(Tensors.create(testImages), Tensors.create(testLabels), Float.class)
        );
    }

    private static float[][] readImages(String imagesPath) throws IOException {
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

    private static float[][] readLabelsOneHot(String labelsPath) throws IOException {
        try(DataInputStream inputStream =
                    new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {
            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();
            return readLabelBuffer(inputStream, numLabels);
        }
    }

    private static byte[][] readBatchedBytes(DataInputStream inputStream, int batches, int bytesPerBatch) throws IOException {
        byte[][] entries = new byte[batches][bytesPerBatch];
        for (int i = 0; i < batches; i++) {
            inputStream.readFully(entries[i]);
        }
        return entries;
    }


    private static float[][] readImageBuffer(DataInputStream inputStream, int numImages, int imageSize) throws IOException {
        byte[][] entries = readBatchedBytes(inputStream, numImages, imageSize);
        float[][] unsignedEntries = new float[numImages][imageSize];
        for (int i = 0; i < unsignedEntries.length; i++) {
            for(int j = 0; j < unsignedEntries[0].length; j++) {
                unsignedEntries[i][j] = (float) (entries[i][j] & 0xFF) / 255.0f;
            }
        }

        return unsignedEntries;
    }

    private static float[][] readLabelBuffer(DataInputStream inputStream, int numLabels) throws IOException {
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
}
