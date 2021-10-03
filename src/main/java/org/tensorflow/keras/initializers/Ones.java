package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TNumber;

public class Ones extends Initializer {
    @Override
    public <T extends TNumber> Operand<T> initialize(Ops tf, Operand<TInt32> shape, Class<T> dtype) {
//        return tf.fill(shape, tf.constant(1.0f, dtype));
        return tf.fill(shape, tf.dtypes.cast(tf.constant(1.0f), dtype));
    }
}
