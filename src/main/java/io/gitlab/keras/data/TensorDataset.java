package io.gitlab.keras.data;


/**
 * Represents a trainable dataset.
 * @param <T> dtype for data values
 */
public class TensorDataset<T extends Number> {
    private CompactTensorSplit<T> train;
    private CompactTensorSplit<T> val;

    public TensorDataset(CompactTensorSplit<T> train, CompactTensorSplit<T> val) {
        this.train = train;
        this.val = val;
    }

    /**
     * Retrieve the train split of this dataset.
     */
    public CompactTensorSplit<T> getTrain() {
        return train;
    }

    /**
     * Retrieve the validation split of this dataset
     */
    public CompactTensorSplit<T> getVal() {
        return val;
    }
}
