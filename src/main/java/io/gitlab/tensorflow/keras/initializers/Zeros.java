package io.gitlab.tensorflow.keras.initializers;

import io.gitlab.tensorflow.keras.utils.Keras;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;


public class Zeros<T> extends Initializer<T> {


    public Operand<T> build(Ops tf, Operand<T> in) {
        this.initializerOp = tf.assign(in, tf.zeros(Keras.shapeOperand(tf, in.asOutput().shape()), this.dtype));
        this.built = true;
        return this.initializerOp;
    }
}
