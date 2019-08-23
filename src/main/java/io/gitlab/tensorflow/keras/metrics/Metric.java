package io.gitlab.tensorflow.keras.metrics;

import io.gitlab.tensorflow.keras.layers.Layer;
import io.gitlab.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public abstract class Metric extends Layer<Float> implements MetricFunction {
    private Operand<Float> outputOp;

    public Metric() {
        super(2);
    }

    public static Metric select(String s) { return select(Metrics.valueOf(s)); }

    private static Metric select(Metrics metricType) {
        return Metrics.select(metricType);
    }

    @Override
    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... ops) {
        if (ops.length != 2) {
            throw new IllegalArgumentException("Metric type " + "'" + this.getClass().getName() + "'" +
                    " call() takes 2 Operand<Float> objects as input. " + "Recevied " + ops.length + ".");
        }

        return call(tf, ops[0], ops[1]);
    }

    public abstract Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label);

    @Override
    public Operand<Float> apply(Ops tf, Operand<Float> output, Operand<Float> label)  {
        return this.apply(tf, output, (Operand<Float>) label);
    }

//    @Override
//    public List<Operand<Float>> metricOps() {
//        return Collections.singletonList(outputOp);
//    }
}
