 package org.tensorflow.keras.layers;

 import org.tensorflow.keras.activations.Activation;
 import org.tensorflow.keras.activations.Activations;
 import org.tensorflow.keras.initializers.Initializer;
 import org.tensorflow.keras.initializers.Initializers;
 import org.tensorflow.keras.types.DataFormat;
 import org.tensorflow.keras.types.PaddingMode;

 import java.util.Arrays;

 public class ConvOptions {
     private long[] strides;
     private PaddingMode padding;

     DataFormat dataFormat;
     private long[] dilationRate;

     private boolean useBias;
     private Activation activation;
     private Initializer kernelInitializer;
     private Initializer biasInitializer;


     public long[] getStrides() {
         return strides;
     }

     public PaddingMode getPadding() {
         return padding;
     }

     public DataFormat getDataFormat() {
         return dataFormat;
     }

     public long[] getDilationRate() {
         return dilationRate;
     }

     public boolean isUseBias() {
         return useBias;
     }

     public Activation getActivation() {
         return activation;
     }

     public Initializer getKernelInitializer() {
         return kernelInitializer;
     }

     public Initializer getBiasInitializer() {
         return biasInitializer;
     }

     /**
      * kernelConstraint
      * biasConstraint
      * kernelRegularizer
      * biasRegularizer
      * activityRegularizer
      */

     public static ConvOptions defaults(int rank, long filters, long[] kernelSize) {
         long[] strides = new long[rank];
         Arrays.fill(strides, 1);
         long[] dilations = new long[rank];
         Arrays.fill(dilations, 1);

         return new Builder()
                 .setActivation(Activations.linear)
                 .setStrides(strides)
                 .setPadding(PaddingMode.valid)
                 .setDataFormat(DataFormat.channelsLast)
                 .setDilationRate(dilations)
                 .setKernelInitializer(Initializers.glorotUniform)
                 .setBiasInitializer(Initializers.zeros)
                 .setUseBias(true)
                 .build();
     }

     public static class Builder {

         private ConvOptions options;

         public Builder() {
             options = new ConvOptions();
         }

         public Builder(ConvOptions options) {
             this.options = options;
         }

         public Builder setStrides(long[] strides) {
             this.options.strides = strides;
             return this;

         }

         public Builder setPadding(PaddingMode padding) {
             this.options.padding = padding;
             return this;

         }

         public Builder setDataFormat(DataFormat dataFormat) {
             this.options.dataFormat = dataFormat;
             return this;

         }

         public Builder setDilationRate(long[] dilationRate) {
             this.options.dilationRate = dilationRate;
             return this;

         }

         public Builder setUseBias(boolean useBias) {
             this.options.useBias = useBias;
             return this;
         }

         public Builder setActivation(Activation activation) {
             this.options.activation = activation;
             return this;

         }

         public Builder setActivation(Activations activation) {
             this.options.activation = Activations.select(activation);
             return this;
         }

         public Builder setKernelInitializer(Initializer kernelInitializer) {
             this.options.kernelInitializer = kernelInitializer;
             return this;

         }

         public Builder setKernelInitializer(Initializers kernelInitializer) {
             this.options.kernelInitializer = Initializers.select(kernelInitializer);
             return this;

         }

         public Builder setBiasInitializer(Initializer biasInitializer) {
             this.options.biasInitializer = biasInitializer;
             return this;

         }

         public Builder setBiasInitializer(Initializers biasInitializer) {
             this.options.biasInitializer = Initializers.select(biasInitializer);
             return this;

         }

         public ConvOptions build() {
             return this.options;
         }

     }
 }
