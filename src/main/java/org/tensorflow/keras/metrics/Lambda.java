package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.LossOrMetric;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

public class Lambda extends Metric {
    private LossOrMetric metricFunction;

    public Lambda(LossOrMetric metricFunction) {
        this.metricFunction = metricFunction;
    }

    @Override
    public <T extends TNumber> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> yTrue, Operand<T> yPred) {
        return this.metricFunction.apply(tf, dtype, yTrue, yPred);
    }
}
