package org.tensorflow.keras.utils;

import org.tensorflow.Shape;
import static org.tensorflow.keras.utils.Keras.*;

public class TensorShape {
    private long[] dims;

    public TensorShape(long head, long... tail) {
        this.dims = new long[tail.length + 1];
        this.dims[0] = head;
        System.arraycopy(tail, 0, this.dims, 1, tail.length);
    }

    public TensorShape(Shape shape) {
        this.dims = dimsFromShape(shape);
    }

    public boolean isKnown(int i) {
        return dims[i] != -1;
    }

    public void assertKnown(int i) {
        if (!isKnown(i)) {
            throw new IllegalStateException("Dimension " + i + " in shape needs to be known.");
        }
    }

    public TensorShape replaceLast(long dim) {
        return replace(this.dims.length - 1, dim);
    }

    public long size(int i) {
        return this.dims[i];
    }

    public int numDimensions() {
        return this.dims.length;
    }

    public TensorShape replace(int i, long dim) {
        dims[i] = dim;
        return this;
    }

    public TensorShape concatenate(long dim) {
        this.dims = Keras.concatenate(this.dims, dim);
        return this;
    }

    public Shape toShape() {
        return Shape.make(head(dims), tail(dims));
    }
}
