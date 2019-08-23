// package io.gitlab.keras.datasets;
//
// import io.gitlab.keras.utils.DataUtils;
// import io.gitlab.keras.utils.Keras;
// import org.nd4j.linalg.api.ndarray.INDArray;
// import org.nd4j.linalg.factory.Nd4j;
//
// import java.io.BufferedReader;
// import java.io.File;
// import java.io.FileReader;
// import java.io.IOException;
// import java.util.Arrays;
// import java.util.LinkedList;
// import java.util.List;
//
// public class BostonHousing {
//    private static final String ORIGIN_PATH =
// "https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz";
//
//    public static void download()throws IOException {
//
//
//
//    }
//
//    public static Dataset loadData(float valSplit) throws IOException {
//        File train = Keras.kerasPath("datasets/boston-housing", "train.csv").toFile();
//        File test = Keras.kerasPath("datasets/boston-housing", "test.csv").toFile();
//
//
//        float[][] X = new float[506][13];
//        float[] y = new float[506];
//
//
//        try (BufferedReader br = new BufferedReader(new FileReader(train))) {
//            String line;
//            br.readLine();
//            int count = 0;
//            while ((line = br.readLine()) != null) {
//                String[] values = line.split(",");
//                for (int i = 1; i < values.length - 1; i++) {
//                    X[count][i - 1] = Float.parseFloat(values[i]);
//                    //System.out.print(values[i] + ", ");
//                }
//
//                y[count] = Float.parseFloat(values[values.length - 1]);
//                count++;
//            }
//        }
//
//        int valNum = (int) (valSplit * 506);
//        int valId = 505 - valNum;
//
//        float[][] XTrain = new float[valId + 1][13];
//        for (int i = 0; i < XTrain.length; i++) {
//            XTrain[i] = Arrays.copyOf(X[i], X[i].length);
//        }
//
//        float[] yTrain = new float[valId + 1];
//        for (int i = 0; i < XTrain.length; i++) {
//            yTrain[i] = y[i];
//        }
//
//        float[][] XTest = new float[valNum][13];
//        float[] yTest = new float[valNum];
//        for (int i = valId + 1; i < 506; i++) {
//            XTest[i - (valId + 1)] = Arrays.copyOf(X[i], X[i].length);
//            yTest[i - (valId + 1)] = y[i];
//        }
//
//
//        return new Dataset(
//                toBatches(XTrain, 3),
//                toBatches(yTrain, 3),
//                toBatches(XTest, 3),
//                toBatches(yTest, 3)
//        );
//    }
//
//    public static List<float[][]> toBatches(float[][] target, int batchSize) {
//        List<float[][]> batches = new LinkedList<>();
//        for (int i = 0; i < target.length - batchSize; i += batchSize) {
//            float[][] batch = new float[batchSize][target[0].length];
//            for (int j = 0; j < batchSize; j++) {
//                for (int k =0; k < target[0].length; k++) {
//                    batch[j][k] = target[i + j][k];
//                }
//            }
//
//            batches.add(batch);
//        }
//
//        return batches;
//    }
//
//    public static List<float[]> toBatches(float[] target, int batchSize) {
//        List<float[]> batches = new LinkedList<>();
//        for (int i = 0; i < target.length - batchSize; i += batchSize) {
//            float[] batch = new float[batchSize];
//            for (int j = 0; j < batchSize; j++) {
//
//                batch[j]= target[i + j];
//            }
//
//            batches.add(batch);
//        }
//
//        return batches;
//    }
//
// }
