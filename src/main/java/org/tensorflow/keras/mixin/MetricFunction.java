package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface MetricFunction {
  Operand<Float> apply(Ops tf, Operand<Float> yTrue, Operand<Float> yPred);
}
