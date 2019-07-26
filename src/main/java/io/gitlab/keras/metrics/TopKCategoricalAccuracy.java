package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class TopKCategoricalAccuracy extends Metric {
    private static final long DEFAULT_K = 5;
    private long k = DEFAULT_K;

    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) throws Exception {
        Operand<Boolean> inTopK = tf.inTopK(output, tf.argMax(output, tf.constant(-1)), k);
        return tf.mean(tf.cast(inTopK, Float.class), tf.constant(-1));
    }
}

