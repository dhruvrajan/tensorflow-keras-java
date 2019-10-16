package org.tensorflow.keras.initializers;


import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

public class RandomNormal<T extends Number> extends Initializer<T> {
    private float mean;
    private float stdev;
    private float p1;
    private float p2;

    public RandomNormal(float mean, float stdev, float p1, float p2) {
        super();
        this.mean = mean;
        this.stdev = stdev;
        this.p1 = p1;
        this.p2 = p2;
    }

    @Override
    public Operand<T> call(Ops tf, Operand<Integer> shape) {
        return tf.parameterizedTruncatedNormal(
                shape,
                tf.constant(this.mean, dtype),
                tf.constant(this.stdev, dtype),
                tf.constant(this.p1, dtype),
                tf.constant(this.p2, dtype)
        );
    }
}
