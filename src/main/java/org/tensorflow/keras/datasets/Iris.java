 package org.tensorflow.keras.datasets;

 import org.tensorflow.Tensors;
 import org.tensorflow.data.GraphLoader;
 import org.tensorflow.data.GraphModeTensorFrame;
 import org.tensorflow.keras.utils.DataUtils;
 import org.tensorflow.keras.utils.Keras;
 import org.tensorflow.utils.Pair;

 import java.io.BufferedReader;
 import java.io.FileReader;
 import java.io.IOException;
 import java.util.Arrays;

 public class Iris {
    private static final String IRIS_ORIGIN =
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
    private static final int NUM_EXAMPLES = 151;
    private static final int INPUT_LENGTH = 4;
    private static final int OUTPUT_LENGTH = 3;
    private static final String LOCAL_PREFIX = "datasets/iris/";
    private static final String LOCAL_FILE = "iris.data";

    private enum COLOR {
        setosa(0), versicolor(1), virginica(2);
        private final int value;

        COLOR(int value) { this.value = value; }
        int getValue() { return this.value; }
    }


    public static void main(String[] args) throws IOException {

    }


    public static void download() throws IOException {
        DataUtils.getFile(LOCAL_PREFIX + LOCAL_FILE, IRIS_ORIGIN);
    }

    public static Pair<GraphLoader<Float>, GraphLoader<Float>> loadData(double val_split) throws IOException {
        Iris.download();
        try (BufferedReader br = new BufferedReader(new FileReader(
                Keras.kerasPath(LOCAL_PREFIX + LOCAL_FILE).toFile()))) {
            float[][] X = new float[NUM_EXAMPLES][INPUT_LENGTH];
            float[][] y = new float[NUM_EXAMPLES][OUTPUT_LENGTH];

            assert X.length == y.length;
            int trainSize = (int) (NUM_EXAMPLES * (1 - val_split));

            float[][] XTrain = new float[trainSize][INPUT_LENGTH];
            float[][] yTrain = new float[trainSize][OUTPUT_LENGTH];

            float[][] XVal = new float[NUM_EXAMPLES - trainSize][INPUT_LENGTH];
            float[][] yVal = new float[NUM_EXAMPLES - trainSize][OUTPUT_LENGTH];

            String line;
            int count = 0;
            while((line = br.readLine()) != null && count < trainSize) {
                if (line.equals("")) break;

                String[] values = line.split(",Iris-");

                String[] xstring = values[0].split(",");
                float[] xvector = new float[xstring.length];
                for (int i = 0; i < xstring.length; i++) {
                    xvector[i] = Float.parseFloat(xstring[i]);
                }

                float[] yvector = oneHot(COLOR.valueOf(values[1]).getValue(),
 COLOR.values().length);

                XTrain[count] = xvector;
                yTrain[count] = yvector;
                count ++;
            }

            while((line = br.readLine()) != null && count < NUM_EXAMPLES) {
                if (line.equals("")) break;

                String[] values = line.split(",Iris-");

                String[] xstring = values[0].split(",");
                float[] xvector = new float[xstring.length];
                for (int i = 0; i < xstring.length; i++) {
                    xvector[i] = Float.parseFloat(xstring[i]);
                }

                float[] yvector = oneHot(COLOR.valueOf(values[1]).getValue(),
 COLOR.values().length);

                XVal[count - trainSize] = xvector;
                yVal[count - trainSize] = yvector;
                count ++;
            }



            return new Pair<>(
                    new GraphModeTensorFrame<>(Float.class, Tensors.create(XTrain), Tensors.create(yTrain)),
                    new GraphModeTensorFrame<>(Float.class, Tensors.create(XVal), Tensors.create(yVal))
            );
        }
    }

    private static float[] oneHot(int label, int total) {
        if (label >= total) {
            throw new IllegalArgumentException("Invalid Index for One-Hot Vector");
        }

        float[] oneHot = new float[total];
        Arrays.fill(oneHot, 0);
        oneHot[label] = 1.0f;
        return oneHot;
    }
 }
