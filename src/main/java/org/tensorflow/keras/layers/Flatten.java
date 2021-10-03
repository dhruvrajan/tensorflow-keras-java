package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TNumber;

public class Flatten<T extends TNumber> extends Layer<T> {
    private static final int FLATTEN_INPUT_LENGTH = 1;
    private Constant<TInt32> units;

    public Flatten() {
        super(FLATTEN_INPUT_LENGTH);
    }

    @Override
    public void build(Ops tf, Shape inputShape) {
        TensorShape tensorShape = new TensorShape(inputShape);
        this.units = tf.constant(new int[] {-1, (int) (tensorShape.numElements() / Math.abs(tensorShape.size(0)))});
    }

    public Shape computeOutputShape(Shape inputShape) {
        // leaves unknown dimensions unknown
        return Shape.of(new TensorShape(inputShape).numElements());
    }

    @SafeVarargs
    public final Operand<T> call(Ops tf, Operand<T>... inputs) {
        return this.call(tf, inputs[0]);
    }

    private Operand<T> call(Ops tf, Operand<T> input) {
        return tf.reshape(input, this.units);
    }

}
