package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public class Zeros<T> extends Initializer<T> {
  @Override
  public Operand<T> call(Ops tf, Operand<Integer> shape) {
    return tf.zeros(shape, dtype);
  }
}
