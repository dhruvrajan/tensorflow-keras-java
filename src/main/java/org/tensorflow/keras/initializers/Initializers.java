package org.tensorflow.keras.initializers;

public enum Initializers {
    zeros, ones, randomNormal, glorotUniform;

    public static <T extends Number> Initializer<T> select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros<T>();
            case ones:
                return new Ones<T>();
            case randomNormal:
                return new RandomNormal<T>(0.0f, 0.1f, -0.2f, 0.2f);
            case glorotUniform:
                throw new UnsupportedOperationException("Glorot Uniform does not exist");
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
