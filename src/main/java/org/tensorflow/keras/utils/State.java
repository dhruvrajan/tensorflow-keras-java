package org.tensorflow.keras.utils;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility to keep track of unique IDs for tensors.
 */
public class State {
    private static long UNIQUE_TENSOR_ID = 0L;
    private static Map<String, Long> UNIQUE_PREFIX_ID = new HashMap<>();

    public static long getUniqueTensorId() {
        return UNIQUE_TENSOR_ID++;
    }

    public static long getUniquePrefixId(String prefix) {
        UNIQUE_PREFIX_ID.putIfAbsent(prefix, 0L);
        long current = UNIQUE_PREFIX_ID.get(prefix);
        UNIQUE_PREFIX_ID.replace(prefix, current + 1);
        return current;
    }
}
