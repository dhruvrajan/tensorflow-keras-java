package io.gitlab.keras.data;


/**
 * Represents a trainable dataset.
 * @param <T> dtype for data values
 */
public class TensorDataset<T extends Number> {
    private TensorSplit<T> train;
    private TensorSplit<T> val;

    public TensorDataset(TensorSplit<T> train, TensorSplit<T> val) {
        this.train = train;
        this.val = val;
    }

    /**
     * Retrieve the train split of this dataset.
     */
    public TensorSplit<T> getTrain() {
        return train;
    }

    /**
     * Retrieve the validation split of this dataset
     */
    public TensorSplit<T> getVal() {
        return val;
    }
}
