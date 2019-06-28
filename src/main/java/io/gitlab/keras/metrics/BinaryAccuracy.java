package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class BinaryAccuracy extends Metric {
    private static final float DEFAULT_THRESHOLD = 0.5f;
    private float threshold = DEFAULT_THRESHOLD;

    BinaryAccuracy setThreshold(float threshold) {
        this.threshold = threshold;
        return this;
    }

    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) {
        Operand<Float> yPred = tf.cast(tf.greater(output, tf.constant(threshold)).asOutput(), Float.class);
        Operand<Float> equal = tf.cast(tf.equal(label, yPred).asOutput(), Float.class);
        return tf.mean(equal, tf.constant(-1));

    }
}
