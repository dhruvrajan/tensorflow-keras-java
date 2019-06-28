package io.gitlab.keras.data;


public class TensorDataset<T extends Number> {
    private TensorSplit<T> train;
    private TensorSplit<T> val;

    public TensorDataset(TensorSplit<T> train, TensorSplit<T> val) {
        this.train = train;
        this.val = val;
    }

    public TensorSplit<T> getTrain() {
        return train;
    }

    public TensorSplit<T> getVal() {
        return val;
    }
}
