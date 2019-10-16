 package org.tensorflow.keras.initializers;

 import org.tensorflow.Operand;
 import org.tensorflow.keras.utils.Keras;
 import org.tensorflow.op.Ops;

 public class Ones<T> extends Initializer<T> {
  @Override
  public Operand<T> call(Ops tf, Operand<Integer> shape) {
    return tf.fill(shape, tf.constant(1.0f, dtype));
  }
 }
