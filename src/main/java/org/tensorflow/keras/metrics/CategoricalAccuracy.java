package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class CategoricalAccuracy extends Metric {

  @Override
  public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
    Operand<Long> trueLabel = tf.argMax(label, tf.constant(-1));
    Operand<Long> predLabel = tf.argMax(output, tf.constant(-1));

    return tf.cast(tf.equal(trueLabel, predLabel).asOutput(), Float.class);
  }

  @Override
  public void build(Ops tf, Shape inputShape) {}

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return Shape.unknown();
  }
}
