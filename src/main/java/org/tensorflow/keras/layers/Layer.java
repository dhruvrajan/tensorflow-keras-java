package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.mixin.LayerFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Base layer class.
 *
 * <p>A layer implements common neural network operations, such as convolution, batch norm, etc.
 * These operations require managing weights, losses, updates, and inter-layer connectivity.
 *
 * @param <T> Numeric type of the output (Float, Double)
 */
public abstract class Layer<T extends Number> implements LayerFunction<T> {
  private int INPUTS_LENGTH;
  protected Class<T> dtype;
  protected boolean built;

  private Map<String, Variable<T>> weights;
  private Map<String, Initializer<T>> initializers;

  public Layer(int numInputs) {
    this.INPUTS_LENGTH = numInputs;

    this.built = false;
    this.weights = new HashMap<>();
    this.initializers = new HashMap<>();
  }

  /**
   * Override create(Ops) to add variables (weight tensors) to the layer.
   *
   * @param tf Tensorflow Ops accessor
   * @param inputShape Shape of the layer's input tensor
   */
  protected abstract void build(Ops tf, Shape inputShape, Class<T> dtype);

  /**
   * Computes the output shape of the tensor returned by a Layer from the input tensor's shape
   *
   * @param inputShape Shape of an input tensor to this layer
   * @return Shape of the tensor that would be returned by `apply`.
   */
  public abstract Shape computeOutputShape(Shape inputShape);

  /**
   * Defines the layer's logic, in terms of input operands, and variables.
   *
   * @param tf Tensorflow Ops accessor.
   * @param inputs A sequence of TF Operands
   * @return The transformed input tensors, according to the layer's logic.
   */
  @SuppressWarnings("unchecked")
  protected abstract Operand<T> call(Ops tf, Operand<T>... inputs);


  /**
   * Internal wrapper for Layer.build()
   * @param tf
   * @param inputShape
   * @param dtype
   * @return
   */
  public void doBuild(Ops tf, Shape inputShape, Class<T> dtype) {
    this.dtype = dtype;
    this.build(tf, inputShape, dtype);
    this.built = true;
  }

  /**
   * Internal wrapper for Layer.call
   * @param tf
   * @param inputs
   * @return
   */
  @SafeVarargs
  public final Operand<T> apply(Ops tf, Operand<T>... inputs) {
    if (!this.built) {
      throw new IllegalStateException(
          "Layer.call() cannot be called before the layer is built (Layer.build())");
    }

    if (inputs.length != INPUTS_LENGTH) {
      throw new IllegalArgumentException(
          "Layer call() expected " + INPUTS_LENGTH + "inputs; received " + inputs.length + ".");
    }

    return this.call(tf, inputs);
  }

  /**
   * Adds a new weight tensor to the layer
   *
   * @param name variable name
   * @param variable variable to add
   * @return the created variable.
   */
  protected Variable<T> addWeight(String name, Variable<T> variable) {
    this.weights.put(name, variable);
    return variable;
  }

  /**
   * Adds a new initializer op to the layer
   *
   * @param name initializer name
   * @param initializer initializer op to add
   * @return the initializer
   */
  protected Initializer<T> addInitializer(String name, Initializer<T> initializer) {
    this.initializers.put(name, initializer);
    return initializer;
  }

  public Collection<Operand<T>> initializerOps() {
    return this.initializers.values().stream()
        .filter(Initializer::isBuilt)
        .map(Initializer::getInitializerOp)
        .collect(Collectors.toList());
  }


  public List<Variable<T>> trainableWeights() {
    return new ArrayList<>(this.weights.values());
  }

  public boolean isBuilt() {
    return this.built;
  }

  public Class<T> getDtype() {
    return this.dtype;
  }
}
