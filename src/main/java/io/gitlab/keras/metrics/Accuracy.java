package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {
    @Override
    public void build(Ops tf) {}

    @Override
    public Operand<Float> call(Ops tf, Operand<Float> output, Operand<Float> label) {
        // Compute Accuracy
        Operand<Long> predicted = tf.argMax(output, tf.constant(1));
        Operand<Long> expected = tf.argMax(label, tf.constant(1));

        return tf.mean(tf.cast(tf.equal(predicted, expected), Float.class), tf.constant(0));
    }

}
