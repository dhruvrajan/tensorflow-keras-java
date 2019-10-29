package org.tensorflow.data;

import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

import java.util.Arrays;

public class Utils {

    /**
     * Create a `startSelector` for use with tf.slice( ... )
     * @return A constant array with `dims` dimensions, with first entry =`start` and
     *          all other entries = 0.
     */
    public static long[] batchStartSelector(Ops tf, long start, int dims) {
        long[] selector = new long[dims];
        Arrays.fill(selector, 0);
        selector[0] = start;
        return selector;
    }

    /**
     * Create a `sizeSelector` to select a batch of a given size using tf.slice( ... ).
     * (Assumes that the first dimension is the batch dimension).
     *
     * @return A constant array with `dims` dimensions, with first entry = size and
     *          all other entries = -1 (unbounded size).
     */
    public static long[] batchSizeSelector(Ops tf, long size, int dims) {
        long[] selector = new long[dims];
        Arrays.fill(selector, -1);
        selector[0] = size;
        return selector;
    }
}
