package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class Accuracy extends Metric {
    public void build(Ops tf) {}

    @Override
    public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
        Operand<Long> predicted = tf.math.argMax(output, tf.constant(1));
        Operand<Long> expected = tf.math.argMax(label, tf.constant(1));

        return tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), Float.class), tf.constant(0));
    }

    @Override
    public void build(Ops tf, Shape inputShape) {

    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        return Shape.unknown();
    }
}
