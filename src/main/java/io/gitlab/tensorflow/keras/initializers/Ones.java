package io.gitlab.tensorflow.keras.initializers;

import io.gitlab.tensorflow.keras.utils.Keras;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

public class Ones<T> extends Initializer<T> {
    @Override
    public Operand<T> build(Ops tf, Operand<T> in) {
        this.initializerOp = tf.assign(in, tf.fill(Keras.shapeOperand(tf, in.asOutput().shape()), tf.constant(1)));
        this.built = true;
        return this.initializerOp;
    }
}
