package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public class Constant extends Initializer {
    private Object val;

    public  Constant(Object val) {
        this.val = val;
    }

    @Override
    public <T extends Number> Operand<T> initialize(Ops tf, Operand<Integer> shape, Class<T> dtype) {
        return tf.fill(shape, tf.constant(val, dtype));
    }
}
