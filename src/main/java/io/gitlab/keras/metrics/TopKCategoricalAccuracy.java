package io.gitlab.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

//public class TopKCategoricalAccuracy extends Metric {
//    private static final long DEFAULT_K = 5;
//    private long k = DEFAULT_K;
//
//    public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
//        Operand<Boolean> inTopK = tf.nn.inTopK(output, (Operand<Long>) tf.math.argMax(output, tf.constant(-1)), k);
//        return tf.math.mean(tf.dtypes.cast(inTopK, Float.class), tf.constant(-1));
//    }
//}

