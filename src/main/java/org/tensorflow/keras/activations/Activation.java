package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.op.Ops;

/**
 * Layer which applies an activation function to an output operand.
 */
public class Activation<T extends Number> extends Layer<T> {

  private ActivationFunction<T> activation;


  /**
   * Creates an Activation function.
   * @param activation An activation function.
   */
  public Activation(ActivationFunction<T> activation) {
    super(1);
    this.activation = activation;
    this.built = true;
  }

  public Activation<T> create(Activations activation) {
    return options().create(activation);
  }

  public static Options options() {
    return new Options();
  }

  public static class Options {
    public Options() {}
    public <T extends Number> Activation<T> create(Activations activation) {
      return Activations.select(activation);
    }
  }

  @Override
  public void build(Ops tf, Shape in) {
    // Activations don't need state to be built and added to a graph. Does nothing.
  }

  @Override
  public Shape computeOutputShape(Shape inputShape) {
    // Activation functions should not change the shape of the input.
    return inputShape;
  }

  private Operand<T> call(Ops tf, Operand<T> inputs) {
    return this.activation.apply(tf, inputs);
  }

  @Override
  public Operand<T> call(Ops tf, Operand<T>... inputs) {
    return call(tf, inputs[0]);
  }
}
