package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.mixin.KerasType;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class Dense extends Layer<Float> implements KerasType<Float> {
  private static int DENSE_INPUT_LENGTH = 1;
  private int units;

  private static String KERNEL = "kernel";
  private static String KERNEL_INIT = "kernelInit";
  private static String BIAS = "bias";
  private static String BIAS_INIT = "biasInit";

  // weight tensors
  private Variable<Float> kernel;
  private Variable<Float> bias;

  // initializers
  private Initializer<Float> kernelInitializer;
  private Initializer<Float> biasInitializer;

  // activation function
  private Activation<Float> activation;

  public Dense(int units, Dense.Options options) {
    super(DENSE_INPUT_LENGTH);
    this.units = units;

    this.activation = Activations.select(options.activation);
    this.kernelInitializer = Initializers.select(options.kernelInitializer);
    this.biasInitializer = Initializers.select(options.biasInitializer);
  }

  private Dense(
      int units,
      Activation<Float> activation,
      Initializer<Float> kernelInitializer,
      Initializer<Float> biasInitializer) {
    super(DENSE_INPUT_LENGTH);
    this.units = units;

    this.activation = activation;
    this.kernelInitializer = kernelInitializer;
    this.biasInitializer = biasInitializer;
  }

  // Dense Builders
  public static Dense create(int units) {
    return options().create(units);
  }

  public static Dense create(int units, Activations activation) {
    return options().setActivation(activation).create(units);
  }

  public static Options options() {
    return new Dense.Options();
  }

  public static class Options {
    // Default parameters
    private Activations activation = Activations.linear;
    private Initializers kernelInitializer = Initializers.zeros;
    private Initializers biasInitializer = Initializers.zeros;

    public Options() {}

    public Options setActivation(Activations activation) {
      this.activation = activation;
      return this;
    }

    public Options setKernelInitializer(Initializers kernelInitializer) {
      this.kernelInitializer = kernelInitializer;
      return this;
    }

    public Options setBiasInitializer(Initializers biasInitializer) {
      this.biasInitializer = biasInitializer;
      return this;
    }

    public Dense create(int units) {
      return new Dense(
          units,
          Activations.select(activation),
          Initializers.select(kernelInitializer),
          Initializers.select(biasInitializer));
    }
  }

  public void build(Ops tf, Shape inputShape) {
    TensorShape tensorShape = new TensorShape(inputShape);
    tensorShape.assertKnown(tensorShape.numDimensions() - 1);

    Shape kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), this.units);

    Shape biasShape = Shape.make(this.units);

    // Create dense kernel tensor
    this.kernel = this.addWeight(KERNEL, tf.variable(kernelShape, Float.class));
    this.addInitializer(KERNEL_INIT, this.kernelInitializer);
    this.kernelInitializer.build(tf, this.kernel);

    // Create bias tensor
    this.bias = this.addWeight(BIAS, tf.variable(biasShape, Float.class));
    addInitializer(BIAS_INIT, this.biasInitializer);
    this.biasInitializer.build(tf, this.bias);

    this.built = true;
  }

  public Shape computeOutputShape(Shape inputShape) {
    // leaves unknown dimensions unknown
    return new TensorShape(inputShape).replaceLast(this.units).toShape();
  }

  @SafeVarargs
  public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
    return this.call(tf, inputs[0]);
  }

  private Operand<Float> call(Ops tf, Operand<Float> input) {
    Operand<Float> signal = tf.add(tf.matMul(input, this.kernel), this.bias);
    return this.activation.apply(tf, signal);
  }
}
