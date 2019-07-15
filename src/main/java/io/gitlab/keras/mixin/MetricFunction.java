package io.gitlab.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.List;

@FunctionalInterface
public interface MetricFunction {
    Operand<Float> apply(Ops tf, Operand<Float> output, Operand<Float> label);
}
