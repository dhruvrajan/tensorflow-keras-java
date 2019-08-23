package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class InputLayer extends Layer<Float> {
  public Placeholder<Float> input;
  private int length;

  public InputLayer(int length) {
    super(0);
    this.length = length;
  }

  public static InputLayer create(int length) {
    return new InputLayer(length);
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
    this.input = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, length)));
    this.built = true;
  }

  @SafeVarargs
  public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
    return input;
  }
}
