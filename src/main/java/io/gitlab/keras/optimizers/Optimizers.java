package io.gitlab.keras.optimizers;

public enum Optimizers {
    sgd,
    adam,
    adagrad,
    adadelta;

    public static Optimizer<Float> select(Optimizers optimizerType) {
        switch (optimizerType) {
            case sgd:
                return new GradientDescentOptimizer(0.01f);
            default:
                throw new IllegalArgumentException("Invalid Optimizer Type.");
        }
    }
}