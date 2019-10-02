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


    public Operand<T> build(Ops tf, Operand<T> in) {
        this.initializerOp = tf.assign(in, tf.fill(Keras.shapeOperand(tf, in.asOutput().shape()), tf.constant(0.1f, dtype)));
        this.built = true;
        return this.initializerOp;
    }
}
