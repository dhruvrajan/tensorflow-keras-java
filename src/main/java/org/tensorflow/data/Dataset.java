package org.tensorflow.data;

import org.tensorflow.Operand;

import java.util.function.Function;

/**
 * Represents a potentially large list of independent elements (samples),
 * and allows iteration and transformations to be performed across
 * these elements.
 */
public interface Dataset<T> {
    /**
     * Groups elements of this dataset into batches.
     *
     * @param batchSize     The number of desired elements per batch
     * @param dropLastBatch Whether to leave out the final batch if
     *                      it has fewer than `batchSize` elements.
     * @return A batched Dataset
     */
    Dataset<T> batch(long batchSize, boolean dropLastBatch);

    default Dataset<T> batch(long batchSize) {

        return batch(batchSize, true);
    }

    /**
     * Concatenates this `Dataset` with another.
     *
     * @param dataset A `Dataset` to be concatenated with this one.
     * @return A `Dataset`.
     */
    Dataset<T> concatenate(Dataset<T> dataset);

    Dataset<T> filter(Function<Operand<T>, Operand<T>> filterFunc);
}