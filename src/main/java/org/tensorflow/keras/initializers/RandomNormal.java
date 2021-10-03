package org.tensorflow.keras.initializers;


import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TNumber;

public class RandomNormal extends Initializer {
    private final float mean;
    private final float stdev;
    private final float p1;
    private final float p2;

    public RandomNormal(float mean, float stdev, float p1, float p2) {
        super();
        this.mean = mean;
        this.stdev = stdev;
        this.p1 = p1;
        this.p2 = p2;
    }

    @Override
    public <T extends TNumber> Operand<T> initialize(Ops tf, Operand<TInt32> shape, Class<T> dtype) {
        return tf.random.parameterizedTruncatedNormal(
                shape,
                tf.constant(this.mean, dtype),
                tf.constant(this.stdev, dtype),
                tf.constant(this.p1, dtype),
                tf.constant(this.p2, dtype)
        );
    }
}
