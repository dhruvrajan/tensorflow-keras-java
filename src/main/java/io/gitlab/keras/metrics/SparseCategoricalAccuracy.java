package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class SparseCategoricalAccuracy extends Metric {
    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        throw new Exception("Sparse Categorical Accuracy is not yet implemented.");
    }
}

