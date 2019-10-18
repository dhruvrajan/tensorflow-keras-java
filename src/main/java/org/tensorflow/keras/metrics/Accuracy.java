package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {
  @Override
  public <T extends Number> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> output, Operand<T> label) {
    Operand<Long> predicted = tf.argMax(output, tf.constant(1));
    Operand<Long> expected = tf.argMax(label, tf.constant(1));

    return tf.mean(tf.cast(tf.equal(predicted, expected), dtype), tf.constant(0));
  }
}
