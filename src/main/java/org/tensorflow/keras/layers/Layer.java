package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.mixin.LayerFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;

import java.util.*;

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

    // Input() layer needs to access dtype and built.
    protected Class<T> dtype;
    protected boolean built;

    private Map<String, Variable<T>> weights;
    private Map<String, Assign<T>> initializerOps;

    public Layer(int numInputs) {
        this.INPUTS_LENGTH = numInputs;

        this.built = false;
        this.weights = new HashMap<>();
        this.initializerOps = new HashMap<>();
    }

    /**
     * Override create(Ops) to add variables (weight tensors) to the layer.
     *
     * The addWeight function and some tf ops require passing a Class<T> "dtype" object
     *
     * To get the dtype of this layer in the build function, use Layer.getDtype()
     *
     * @param tf         Tensorflow Ops accessor
     * @param inputShape Shape of the layer's input tensor
     *
     */
    protected abstract void build(Ops tf, Shape inputShape);

    public final void build(Ops tf, Shape inputShape, Class<T> dtype) {
      this.dtype = dtype;
      build(tf, inputShape);
      this.built = true;
    }


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
     * @param tf     Tensorflow Ops accessor.
     * @param inputs A sequence of TF Operands
     * @return The transformed input tensors, according to the layer's logic.
     */
    @SuppressWarnings("unchecked")
    protected abstract Operand<T> call(Ops tf, Operand<T>... inputs);


    /**
     * Internal wrapper for Layer.call
     *
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
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected final Variable<T> addWeight(String name, Variable<T> variable) {
        this.weights.put(name, variable);
        return variable;
    }

    /**
     * Adds a new weight tensor to the layer
     *
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected final Variable<T> addWeight(Ops tf, String name, Variable<T> variable, String initializerName, Initializers initializer) {
        return addWeight(tf, name, variable, initializerName, Initializers.select(initializer));
    }

    /**
     * Adds a new weight tensor to the layer
     *
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected final Variable<T> addWeight(Ops tf, String name, Variable<T> variable, String initializerName, Initializer initializer) {
        this.weights.put(name, variable);
        this.initializerOps.put(initializerName, initializer.apply(tf, variable, dtype));
        return variable;
    }

    public Collection<Operand<T>> initializerOps() {
        return new ArrayList<>(this.initializerOps.values());
    }


    public List<Variable<T>> trainableWeights() {
        return new ArrayList<>(this.weights.values());
    }

    public boolean isBuilt() {
        return this.built;
    }

    public boolean hasDtype() {
      return this.dtype != null;
    }

    public Class<T> getDtype() {
        return this.dtype;
    }
}
