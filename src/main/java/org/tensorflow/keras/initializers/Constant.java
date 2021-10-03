package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TNumber;

public class Constant extends Initializer {
    private final int    tpe;
    private final int    intValue;
    private final long   longValue;
    private final float  floatValue;
    private final double doubleValue;

    public Constant(int value) {
        this.tpe            = 0;
        this.intValue       = value;
        this.longValue      = 0L;
        this.floatValue     = 0f;
        this.doubleValue    = 0.0;
    }

    public Constant(long value) {
        this.tpe            = 1;
        this.intValue       = 0;
        this.longValue      = value;
        this.floatValue     = 0f;
        this.doubleValue    = 0.0;
    }

    public Constant(float value) {
        this.tpe            = 2;
        this.intValue       = 0;
        this.longValue      = 0L;
        this.floatValue     = value;
        this.doubleValue    = 0.0;
    }

    public Constant(double value) {
        this.tpe            = 3;
        this.intValue       = 0;
        this.longValue      = 0L;
        this.floatValue     = 0f;
        this.doubleValue    = value;
    }

    @Override
    public <T extends TNumber> Operand<T> initialize(Ops tf, Operand<TInt32> shape, Class<T> dtype) {
        // return tf.fill(shape, tf.constant(val, dtype));
        final Operand<?> c;
        switch(tpe) {
            case 0: c = tf.constant(intValue); break;
            case 1: c = tf.constant(longValue); break;
            case 2: c = tf.constant(floatValue); break;
            case 3: c = tf.constant(doubleValue); break;
            default: throw new RuntimeException(String.valueOf(tpe));
        }
        return tf.fill(shape, tf.dtypes.cast(c, dtype));
    }
}
