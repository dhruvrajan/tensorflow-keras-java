package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class SoftmaxCrossEntropyLoss<T extends Number> extends Loss<T> {
  public Operand<T> loss;
  public Operand lossPrintOp;

  @Override
  protected Operand<T> call(Ops tf, Operand<T> actual, Operand<T> labels) {
    loss = tf.mean(tf.neg(tf.reduceSum(tf.mul(actual, tf.log(labels)), Keras.constArray(tf, 1))), Keras.constArray(tf, 0));
//    loss = tf.mean(tf.softmaxCrossEntropyWithLogits(actual, labels).loss(), tf.constant(0));
    return loss;
  }
}
