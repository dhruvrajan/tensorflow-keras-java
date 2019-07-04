package io.gitlab.keras.losses;

public enum Losses {
    mean_squared_error, softmax_crossentropy;

    public static Loss select(Losses lossType) {
        switch (lossType) {
            case mean_squared_error:
                return new MeanSquaredError();
            case softmax_crossentropy:
                return new SoftmaxCrossEntropyLoss();
            default:
                throw new IllegalArgumentException("Invalid loss type.");
        }
    }
}