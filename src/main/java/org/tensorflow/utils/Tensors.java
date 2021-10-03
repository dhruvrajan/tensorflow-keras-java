package org.tensorflow.utils;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

public class Tensors {
    public static Tensor create(float[][] values) {
        return Tensor.of(TFloat32.class, Shape.of(values.length, values[0].length), data -> StdArrays.copyTo(values, data));
    }

    public static Tensor create(float[][][] values) {
        return Tensor.of(TFloat32.class, Shape.of(values.length, values[0].length, values[1].length), data -> StdArrays.copyTo(values, data));
    }
}
