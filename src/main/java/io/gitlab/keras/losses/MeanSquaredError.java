package io.gitlab.keras.losses;

import io.gitlab.keras.metrics.Metric;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class MeanSquaredError extends Loss {
    public Operand<Float> build(Ops tf, Operand<Float> output, Operand<Float> label) {
        return tf.mean(tf.squaredDifference(output, label), tf.constant(-1));
    }
}
