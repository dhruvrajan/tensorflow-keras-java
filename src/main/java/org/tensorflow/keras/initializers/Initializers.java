package org.tensorflow.keras.initializers;

public enum Initializers {
    zeros, ones, randomNormal, glorotUniform;

    public static Initializer select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros();
            case ones:
                return new Ones();
            case randomNormal:
                return new RandomNormal(0.0f, 0.1f, -0.2f, 0.2f);
            case glorotUniform:
                throw new UnsupportedOperationException("Glorot Uniform does not yet exist");
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
