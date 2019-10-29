package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {
  @Override
  public <T extends Number> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> output, Operand<T> label) {
    Operand<Long> predicted = tf.math.argMax(output, tf.constant(1));
    Operand<Long> expected = tf.math.argMax(label, tf.constant(1));

    return tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), dtype), tf.constant(0));
  }
}
