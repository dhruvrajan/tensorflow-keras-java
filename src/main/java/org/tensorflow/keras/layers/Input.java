package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Input extends Layer<Float> {
  public Placeholder<Float> input;
  private long[] dims;

  public Input(long... otherDims) {
    super(0);
    this.dims = otherDims;
  }


  @Override
  public void build(Ops tf, Shape inputShape) {
    throw new UnsupportedOperationException(
        "Cannot call create(Ops, Shape) on input layer with an input shape. Use create(Ops).");
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    throw new UnsupportedOperationException(
        "Cannot call create(Ops, Shape) on input layer with an input shape. Use create(Ops).");
  }

  public Shape computeOutputShape() {
    return input.asOutput().shape();
  }

  public void build(Ops tf) {
    this.input = tf.placeholder(Float.class, Placeholder.shape(Keras.shapeFromDims(Keras.concatenate(-1, this.dims))));
    this.built = true;
  }

  @SafeVarargs
  public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
    return input;
  }
}
