package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {
  public void build(Ops tf) {}

  @Override
  public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
    Operand<Long> predicted = tf.argMax(output, tf.constant(1));
    Operand<Long> expected = tf.argMax(label, tf.constant(1));

    return tf.mean(tf.cast(tf.equal(predicted, expected), Float.class), tf.constant(0));
  }

  @Override
  public void build(Ops tf, Shape inputShape) {}

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return Shape.unknown();
  }
}
