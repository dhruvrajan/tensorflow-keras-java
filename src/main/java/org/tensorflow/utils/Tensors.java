package org.tensorflow.utils;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

public class Tensors {
    public static Tensor create(int[][] values) {
        return TInt32.tensorOf(StdArrays.ndCopyOf(values));
    }

    public static Tensor create(float[] values) {
        return TFloat32.vectorOf(values);
    }

    public static Tensor create(float[][] values) {
        // return Tensor.of(TFloat32.class, Shape.of(values.length, values[0].length), data -> StdArrays.copyTo(values, data));
        return TFloat32.tensorOf(StdArrays.ndCopyOf(values));
    }

    public static Tensor create(float[][][] values) {
        return TFloat32.tensorOf(StdArrays.ndCopyOf(values));
    }
}
