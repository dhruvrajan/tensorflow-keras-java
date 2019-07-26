package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {


    public Operand<Float> build(Ops tf, Operand<Float> output, Placeholder<Float> label) {
        // Compute Accuracy
        Operand<Long> predicted = tf.argMax(output, tf.constant(1));
        Operand<Long> expected = tf.argMax(label, tf.constant(1));
        Operand<Float> accuracy = tf.mean(tf.cast(tf.equal(predicted, expected), Float.class), tf.constant(0));

        return accuracy;
    }

}
