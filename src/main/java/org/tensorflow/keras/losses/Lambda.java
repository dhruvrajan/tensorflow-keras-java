package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.LossOrMetric;
import org.tensorflow.op.Ops;

public class Lambda extends Loss {
    private LossOrMetric metricFunction;

    public Lambda(LossOrMetric metricFunction) {
        this.metricFunction = metricFunction;
    }

    @Override
    public <T extends Number> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> yTrue, Operand<T> yPred) {
        return this.metricFunction.apply(tf, dtype, yTrue, yPred);
    }
}
