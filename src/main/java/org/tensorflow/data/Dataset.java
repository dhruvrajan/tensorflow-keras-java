package org.tensorflow.data;

import java.util.Iterator;
import java.util.stream.Stream;

/**
 * Represents a potentially large list of independent elements (samples),
 * and allows iteration and transformations to be performed across
 * these elements.
 */
public interface Dataset<T> {

    /**
     * Creates a new stream of dataset elements (data batches).
     */
//    Stream<T> stream();

    /**
     * Creates a new iterator over dataset elements (data batches).
//     */
//    Iterator<T> iterator();

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


    long getBatchSize();
    long numBatches();

    long numTensors();
    long numElementsPerTensor();
}