package io.gitlab.keras.utils;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class Keras {

    //
    // Manage .keras configuration location
    //

    private static final String DEFAULT_KERAS_HOME =
            Paths.get(System.getProperty("user.home"), ".keras").toString();
    private static final String SYSTEM_KERAS_HOME_VAR = "KERAS_HOME";

    public static void main(String[] args) {
        System.out.println(DEFAULT_KERAS_HOME);
    }

    public static String kerasHome() {
        String systemHome = System.getenv(SYSTEM_KERAS_HOME_VAR);
        if (systemHome != null) return systemHome;
        return DEFAULT_KERAS_HOME;
    }

    public static Path kerasPath(String... path) {
        return Paths.get(kerasHome(), path);
    }

    public static String datasetsDirectory() {
        return Paths.get(kerasHome(), "datasets").toString();
    }

    //
    // Keras backend utilties
    //

    public static <T>Operand<Integer> constArray(Ops tf, int... i) {
        return tf.constant(i);
    }

    public static Operand<Long> shapeOperand(Ops tf, Shape shape) {
        long[] shapeArray = new long[shape.numDimensions()];
        for (int i = 0; i < shapeArray.length; i++) {
            shapeArray[i] = shape.size(i);
        }

        return tf.constant(shapeArray);
    }

    public static long head(long... dims) {
        return dims[0];
    }

    public static long[] tail(long... dims) {
        return Arrays.copyOfRange(dims, 1, dims.length);
    }

    public static long[] concatenate(long[] first, long last) {
        long[] dims = new long[first.length + 1];
        System.arraycopy(first, 0, dims, 0, first.length);
        dims[dims.length - 1] = last;
        return dims;
    }

    public static long[] dimsFromShape(Shape shape) {
        long[] dims = new long[shape.numDimensions()];
        for (int i = 0; i < shape.numDimensions(); i++) {
            dims[i] = shape.size(i);
        }
        return dims;
    }
}
