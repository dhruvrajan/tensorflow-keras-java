package io.gitlab.keras.losses;

import io.gitlab.keras.mixin.MetricFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.Collections;
import java.util.List;

public abstract class Loss implements MetricFunction {
    Operand<Float> outputOp;

    public abstract Operand<Float> build(Ops tf, Operand<Float> actual, Operand<Float> labels);

    @Override
    public Operand<Float> apply(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        outputOp = build(tf, output, label);
        return outputOp;
    }

    @Override
    public List<Operand<Float>> metricOps() {
        return Collections.singletonList(outputOp);
    }

    public static Loss select(String lossName) {
        return LossType.select(LossType.valueOf(lossName));
    }

    enum LossType {
        mean_squared_error, softmax_crossentropy;

        static Loss select(LossType lossType) {
            switch (lossType) {
                case mean_squared_error:
                    return new MeanSquaredError();
                case softmax_crossentropy:
                    return new SoftmaxCrossEntropyLoss();
                default:
                    throw new IllegalArgumentException("Invalid loss type.");
            }
        }
    }
}
