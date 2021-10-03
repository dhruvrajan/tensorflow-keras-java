package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.family.TNumber;

public class Input<T extends TNumber> extends Layer<T> {
  private Placeholder<T> input;
  private final long[] dims;

  public Input(long... otherDims) {
    super(0);
    this.dims = otherDims;
  }

  @Override
  public void build(Ops tf, Shape inputShape) {
    throw new UnsupportedOperationException(
        "Cannot build an input layer with an input shape; it doesn't take any inputs. Use Input.build(Ops tf, Class<T> dtype)");
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    throw new UnsupportedOperationException(
        "Cannot call computeOutputShape on");
  }

  public Shape computeOutputShape() {
    return input.asOutput().shape();
  }

  public void build(Ops tf, Class<T> dtype) {
    this.dtype = dtype;
    // System.out.println(dtype);
    this.input = tf.placeholder(dtype, Placeholder.shape(Keras.shapeFromDims(Keras.concatenate(-1, this.dims))));
    this.built = true;
  }

  @SafeVarargs
  public final Operand<T> call(Ops tf, Operand<T>... inputs) {
    return input;
  }
}