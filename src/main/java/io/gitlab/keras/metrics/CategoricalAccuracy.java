package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class CategoricalAccuracy extends Metric {

    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) {
        Operand<Long> trueLabel = tf.argMax(label, tf.constant(-1));
        Operand<Long> predLabel = tf.argMax(output, tf.constant(-1));

        return tf.cast(tf.equal(trueLabel, predLabel).asOutput(), Float.class);
    }
}
