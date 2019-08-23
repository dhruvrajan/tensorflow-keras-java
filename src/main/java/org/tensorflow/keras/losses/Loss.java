package org.tensorflow.keras.losses;

import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public abstract class Loss extends Layer<Float> implements MetricFunction {

    public Loss() {
        super(2);
    }

    /**
     * Subclasses should override this method.
     */
    protected abstract Operand<Float> call(Ops tf, Operand<Float> actual, Placeholder<Float> labels);

    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        if (!(inputs[1] instanceof Placeholder)) {
            throw new IllegalArgumentException("Second input to loss must be a placeholder");
        }
        return this.call(tf, inputs[0], (Placeholder<Float>) inputs[1]);
    }

    @Override
    public Operand<Float> apply(Ops tf, Operand<Float> output, Operand<Float> label) {
        // Call Layer.apply
        return this.apply(tf, output, (Operand<Float>) label);
    }
}
