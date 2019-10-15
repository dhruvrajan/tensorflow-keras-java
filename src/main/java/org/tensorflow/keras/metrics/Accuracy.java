package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy<T extends Number> extends Metric<T> {
  Class<T> dtype;

  public void build(Ops tf) {}

  @Override
  public Operand<T> call(Ops tf, Operand<T> output, Operand<T> label) {
    Operand<Long> predicted = tf.argMax(output, tf.constant(1));
    Operand<Long> expected = tf.argMax(label, tf.constant(1));

    return tf.mean(tf.cast(tf.equal(predicted, expected), dtype), tf.constant(0));
  }

  @Override
  public void build(Ops tf, Shape inputShape, Class<T> dtype) {
    System.out.println("FOUND >>> " + dtype);
    this.dtype = dtype;
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return Shape.unknown();
  }
}
