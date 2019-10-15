package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

public class Flatten<T extends Number> extends Layer<T> {
    private static int FLATTEN_INPUT_LENGTH = 1;
    private Constant<Integer> units;

    public Flatten() {
        super(FLATTEN_INPUT_LENGTH);
    }

    @Override
    public void build(Ops tf, Shape inputShape, Class<T> dtype) {
        TensorShape tensorShape = new TensorShape(inputShape);
        this.units = tf.constant(new int[] {-1, (int) (tensorShape.numElements() / Math.abs(tensorShape.size(0)))});
        this.built = true;
    }

    public Shape computeOutputShape(Shape inputShape) {
        // leaves unknown dimensions unknown
        return Shape.make(new TensorShape(inputShape).numElements());
    }

    @SafeVarargs
    public final Operand<T> call(Ops tf, Operand<T>... inputs) {
        return this.call(tf, inputs[0]);
    }

    private Operand<T> call(Ops tf, Operand<T> input) {
        return tf.reshape(input, this.units);
    }

}
