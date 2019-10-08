package org.tensorflow.keras.initializers;

public enum Initializers {
    zeros, ones, randomNormal, glorotUniform;

    public static Initializer<Float> select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros<Float>().dtype(Float.class);
            case ones:
                return new Ones<Float>().dtype(Float.class);
            case randomNormal:
                return new RandomNormal<Float>(0.0f, 0.1f, -0.2f, 0.2f).dtype(Float.class);
            case glorotUniform:
                throw new UnsupportedOperationException("Glorot Uniform does not exist");
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
