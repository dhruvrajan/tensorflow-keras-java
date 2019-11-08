package org.tensorflow.data;

import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.index.Indices;

import java.util.Iterator;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class NioTensorFrame<T> extends TensorFrame<T> {
    private NdArray<T>[] batchTensors;

    @SafeVarargs
    public NioTensorFrame(NdArray<T> firstTensor, NdArray<T>... tensors) {
        long batchDim = firstTensor.shape().size(0);
        for (NdArray<T> tensor : tensors) {
            if (tensor.shape().size(0) != batchDim) {
                throw new IllegalArgumentException("All tensors in a tensor frame must have" +
                        "equal batch (first) dimension");
            }
        }

        // Record Tensor Objects
        this.batchTensors = (NdArray<T>[]) new NdArray[tensors.length + 1];
        this.batchTensors[0] = firstTensor;
        System.arraycopy(tensors, 0, this.batchTensors, 1, tensors.length);
    }

    @Override
    public long numTensors() {
        return this.batchTensors.length;
    }

    @Override
    public long numElementsPerTensor() {
        return this.batchTensors[0].shape().size(0);
    }

    public NdArray<T>[] getBatch(long batchIndex) {
        NdArray<T>[] batches = (NdArray<T>[]) new NdArray[(int) numTensors()];

        long batchSize = getBatchSize();

        for (int i = 0; i < batches.length; i++) {
            long batchStart = batchIndex * batchSize;
            long batchEnd = batchStart + batchSize;
            batches[i] = batchTensors[i].slice(Indices.range(batchStart, batchEnd));
        }

        return batches;
    }

    public Iterator<NdArray<T>[]> batchIterator() {
        return new Iterator<>() {
            int currentBatch = 0;
            @Override
            public boolean hasNext() {
                return currentBatch < numBatches();
            }

            @Override
            public NdArray<T>[] next() {
                return getBatch(currentBatch++);
            }
        };
    }

    public void forEachBatch(Consumer<NdArray<T>[]> consumer) {
        for (int i = 0; i < numBatches(); i++) {
            consumer.accept(getBatch(i));
        }
    }

    public void forEachBatchIdx(BiConsumer<Long, NdArray<T>[]> consumer) {
        for (long i = 0; i < numBatches(); i++) {
            consumer.accept(i, getBatch(i));
        }
    }
}
