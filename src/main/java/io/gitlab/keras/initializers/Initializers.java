package io.gitlab.keras.initializers;

public enum Initializers {
    zeros;

    public static <T> Initializer<T> select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros<>();
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
