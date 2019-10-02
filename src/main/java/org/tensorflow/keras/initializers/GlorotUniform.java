package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public class GlorotUniform<T extends Number > extends Initializer<T> {
    @Override
    public Operand<T> build(Ops tf, Operand<T> in) {
        this.initializerOp = tf.assign(in, tf.randomNormal(Keras.shapeOperand(tf, in.asOutput().shape()), dtype));
        this.built = true;
        return this.initializerOp;
    }
}
