package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public class Zeros<T> extends Initializer<T> {

  @Override
  public Operand<T> build(Ops tf, Operand<T> in, Class<T> dtype) {
    this.initializerOp = tf.assign(in, tf.zeros(Keras.shapeOperand(tf, in.asOutput().shape()), dtype));
    this.built = true;
    return this.initializerOp;
  }
}
