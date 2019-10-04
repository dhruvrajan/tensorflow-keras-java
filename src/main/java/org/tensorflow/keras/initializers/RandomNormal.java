package org.tensorflow.keras.initializers;


import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;
import org.tensorflow.DataType;

public class RandomNormal<T extends Number > extends Initializer<T> {
    private T mean;
    private T stdev;
    private T p1;
    private T p2;

    public RandomNormal(T mean, T stdev, T p1, T p2) {
        super();
        this.mean = mean;
        this.stdev = stdev;
        this.p1 = p1;
        this.p2 = p2;
    }

    @Override
    public Operand<T> build(Ops tf, Operand<T> in) {

        this.initializerOp = tf.assign(in, tf.parameterizedTruncatedNormal(
                Keras.shapeOperand(tf, in.asOutput().shape()),
                tf.constant(this.mean, dtype),
                tf.constant(this.stdev, dtype),
                tf.constant(this.p1, dtype),
                tf.constant(this.p2, dtype)
        ));

//        this.initializerOp = tf.assign(in, tf.randomNormal(
//                Keras.shapeOperand(tf, in.asOutput().shape()),
//                dtype
//        ));

        this.built = true;
        return this.initializerOp;
    }
}
