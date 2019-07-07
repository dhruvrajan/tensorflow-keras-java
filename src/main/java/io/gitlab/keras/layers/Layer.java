package io.gitlab.keras.layers;

import io.gitlab.keras.initializers.Initializer;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Base layer class.
 *
 * A layer implements common neural network operations, such as
 * convolution, batch norm, etc. These operations require managing weights,
 * losses, updates, and inter-layer connectivity.
 *
 * @param <T> Numeric type of the output (Float, Double)
 */
public abstract class Layer<T> {
    private static int ID_COUNTER = 0;
    protected boolean built = false;
    int id;

    public Map<String, Variable<T>> weights;
    public Map<String, Initializer<T>> initializers;

    public Layer() {
        this.id = ID_COUNTER++;
        weights = new HashMap<>();
        initializers = new HashMap<>();
    }

    /**
     * Adds a new weight tensor to the layer
     * @param name variable name
     * @param variable variable to add
     * @return the created variable.
     */
    public Variable<T> addWeight(String name, Variable<T> variable) {
       this.weights.put(name, variable);
       return variable;
    }

    /**
     * Adds a new initializer op to the layer
     * @param name initializer name
     * @param initializer initializer op to add
     * @return
     */
    public Initializer<T> addInitializer(String name, Initializer<T> initializer) {
        this.initializers.put(name, initializer);
        return initializer;
    }

    public Collection<Operand<T>> initializerOps() {
        return this.initializers.values().stream()
                .filter(Initializer::isBuilt)
                .map(Initializer::getInitializerOp)
                .collect(Collectors.toList());
    }

    public Collection<Variable<T>> trainableWeights() {
        return this.weights.values();
    }

    public abstract void build(Ops tf);
    public abstract Operand<T> call(Ops tf, Operand<T> in);
    public boolean isBuilt() {
        return this.built;
    }
}