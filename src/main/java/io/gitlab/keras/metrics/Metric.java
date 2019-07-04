package io.gitlab.keras.metrics;

import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.mixin.MetricFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.Collections;
import java.util.List;

public abstract class Metric extends Layer<Float> implements MetricFunction {
    Operand<Float> outputOp;

    public abstract Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception;

    public static Metric select(String s) { return select(Metrics.valueOf(s)); }

    private static Metric select(Metrics metricType) {
        return Metrics.select(metricType);
    }

    @Override
    public Operand<Float> apply(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        outputOp =  build(tf, output, label);
        return outputOp;
    }

    @Override
    public List<Operand<Float>> metricOps() {
        return Collections.singletonList(outputOp);
    }
}
