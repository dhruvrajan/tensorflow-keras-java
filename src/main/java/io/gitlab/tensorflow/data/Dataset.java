package io.gitlab.tensorflow.data;

import io.gitlab.tensorflow.keras.utils.Pair;
import org.tensorflow.Operand;

import java.util.Collection;
import java.util.Iterator;
import java.util.function.Function;

public abstract class Dataset<T> implements Iterable<Collection<Operand<T>>> {

    public abstract void apply(Function<Dataset<T>, Dataset<T>> transformation);

    public Dataset<T> batch(int batchSize) {
        return this.batch(batchSize, false);
    }
    public abstract Dataset<T> batch(int batchSize, boolean dropRemainder);

    public abstract Dataset<T> concatenate(Dataset<T> other);
    public abstract Dataset<T> filter();


    public abstract Iterator<Pair<Integer, Collection<T>>> enumerate(int start);

}
