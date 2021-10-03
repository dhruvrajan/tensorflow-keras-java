package org.tensorflow.keras.utils;

import org.tensorflow.Operand;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;

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

  public static Operand<TInt32> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }

  public static Operand<TInt32> shapeOperand(Ops tf, Shape shape) {
    int[] shapeArray = new int[shape.numDimensions()];
    for (int i = 0; i < shapeArray.length; i++) {
      shapeArray[i] = (int) shape.size(i);
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

  public static long[] concatenate(long first, long... remaining) {
    long[] dims = new long[remaining.length + 1];
    System.arraycopy(remaining, 0, dims, 1, remaining.length);
    dims[0] = first;
    return dims;
  }

  public static long[] dimsFromShape(Shape shape) {
    long[] dims = new long[shape.numDimensions()];
    for (int i = 0; i < shape.numDimensions(); i++) {
      dims[i] = shape.size(i);
    }
    return dims;
  }

  public static Shape shapeFromDims(long... dims) {
    return Shape.of(dims); // Shape.make(head(dims), tail(dims));
  }
}
