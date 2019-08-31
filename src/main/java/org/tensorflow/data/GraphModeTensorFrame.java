package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Collectors;

public class GraphModeTensorFrame<T> extends TensorFrame<T> {
    private Class<T> dtype;
    private Tensor<T>[] tensors;

    public GraphModeTensorFrame(Class<T> dtype, Tensor<T> firstTensor, Tensor<T>... tensors) {
        this.dtype = dtype;

        // Check first dimension matches
        long matchDim = firstTensor.shape()[0];
        for (Tensor<T> t: tensors) {
            if (t.shape()[0] != matchDim) {
                throw new IllegalArgumentException("All tensors in a tensor frame must have equal first dimension.");
            }
        }

        this.tensors = new Tensor[tensors.length + 1];
        this.tensors[0] = firstTensor;
        System.arraycopy(tensors, 0, this.tensors, 1, tensors.length);
    }

    public long size() {
        return this.tensors[0].shape()[0];
    }

    /** Utility to construct a Shape from a long[] */
    private static Shape getShape(long... dims) {
        assert dims.length > 0;

        long head = dims[0];
        long[] tail = new long[dims.length - 1];
        System.arraycopy(dims, 1, tail, 0, dims.length - 1);

        return Shape.make(head, tail);
    }

    private Operand<T>[] getBatch(Ops tf, long i) {
        Operand<T>[] ops = new Operand[this.tensors.length];
        return Arrays.stream(tensors)
                .map(tensor -> {
                    Operand<T> variable = tf.variable(getShape(tensor.shape()), dtype);
                    var begin = tf.constant(i * batchSize);
                    var size = tf.constant(batchSize);
                    return tf.slice(variable, begin, size);
                }).collect(Collectors.toList()).toArray(ops);
    }

    public Iterator<Operand<T>[]> iterator(Ops tf) {
        return new Iterator<Operand<T>[]>() {
            int index = 0;

            @Override
            public boolean hasNext() {
                return index < size() / batchSize;
            }

            @Override
            public Operand<T>[] next() {
                Operand<T>[] batch = getBatch(tf, index);
                index++;
                return batch;
            }
        };
    }
}
