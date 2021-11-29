package org.tensorflow.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.family.TNumber;

import java.util.List;

public abstract class Optimizer<T extends TNumber> {
    private Class<T> dtype;
    private boolean built;

    public List<Operand<T>> minimize(Ops tf, Operand<T> loss, List<Variable<T>> weights) {
        if (!isBuilt()) {
            throw new IllegalStateException("Must call optimizer.build(Ops, Class<T>) before calling optimizer.minimize()");
        }

        Gradients gradients = computeGradients(tf, loss, weights);
        return applyGradients(tf, weights, gradients);
    }

    private Gradients computeGradients(Ops tf, Operand<T> loss, List<Variable<T>> weights) {
        return tf.gradients(loss, weights);
    }

    public void build(Ops tf, Class<T> dtype) {
        this.dtype = dtype;
        this.build(tf);
        this.built = true;
    }

    protected abstract void build(Ops tf);

    protected abstract List<Operand<T>> applyGradients(Ops tf, List<Variable<T>> weights, Gradients gradients);

    public Class<T> getDtype() {
        return dtype;
    }

    public boolean isBuilt() {
        return built;
    }
}
