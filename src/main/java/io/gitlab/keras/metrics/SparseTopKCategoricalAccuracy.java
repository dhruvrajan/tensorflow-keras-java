package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class SparseTopKCategoricalAccuracy extends Metric {
    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        throw new Exception("SparseTopKCategoricalAccuracy is not yet implemented.");
    }
}

