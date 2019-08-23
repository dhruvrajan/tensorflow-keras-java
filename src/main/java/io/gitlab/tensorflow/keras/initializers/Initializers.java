package io.gitlab.tensorflow.keras.initializers;

public enum Initializers {
    zeros;

    public static Initializer<Float> select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros<Float>().dtype(Float.class);
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
