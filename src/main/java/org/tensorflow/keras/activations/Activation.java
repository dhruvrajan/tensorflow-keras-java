package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.op.Ops;

/**
 * Base activation function class.
 */
public abstract class Activation<T extends Number> extends Layer<T> {

  public Activation() {
    super(1);
  }

  @Override
  public void build(Ops tf, Shape inputShape) {
    // Activations don't need state to be built and added to a graph. Does nothing.
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    // Activation functions should not change the shape of the input.
    return inputShape;
  }

  @Override
  public Operand<T> call(Ops tf, Operand<T>... inputs) {
    return call(tf, inputs[0]);
  }

  /**
   * Calls the activation function. Override this when defining an activation function.
   */
  protected abstract Operand<T> call(Ops tf, Operand<T> inputs);
}
