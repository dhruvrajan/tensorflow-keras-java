package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public class Constant<T> extends Initializer<T> {
    T val;

    public Constant(T val) {
        super();
        this.val = val;
    }

    @Override
    public Operand<T> call(Ops tf, Operand<Integer> shape) {
        return tf.fill(shape, tf.constant(0.1f, dtype));
    }
}
