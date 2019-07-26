package io.gitlab.keras.utils;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

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

    public static Operand<Integer> constArray(Ops tf, int... i) {
        return tf.constant(i);
    }
}
