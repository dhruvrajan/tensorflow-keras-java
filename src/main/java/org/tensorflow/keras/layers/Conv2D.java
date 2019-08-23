package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;

public class Conv2D extends Layer<Float> {

  org.tensorflow.op.core.Conv2D<Float> convOp;

  private Conv2D(long filters, long[] kernel, long[] strides) {
    super(1);
  }

  @Override
  public void build(Ops tf, Shape inputShape) {}

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    return null;
  }

  @Override
  protected Operand<Float> call(Ops tf, Operand<Float>... inputs) {
    return null;
  }
}
