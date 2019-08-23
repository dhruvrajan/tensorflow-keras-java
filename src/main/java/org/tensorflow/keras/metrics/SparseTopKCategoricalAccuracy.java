package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class SparseTopKCategoricalAccuracy extends Metric {
  @Override
  public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
    throw new UnsupportedOperationException(
        "SparseTopKCategoricalAccuracy is not yet implemented.");
  }

  @Override
  public void build(Ops tf, Shape inputShape) {}

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return Shape.unknown();
  }
}
