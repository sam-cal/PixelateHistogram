# PixelateHistogram
Contains code for an activation layer that ease the training of NN on histograms

In layers.py
PixelateLayer: an activation layer, that by 'pixelizing' an histogram, allows a better training of NN on histograms
PoissonianNoise: regularisation layer, that add Poissonian noise on top of input during the training phase

In losses.py:
Significance: a loss function that account for the Poisson statistic to compute the deviation between the output and the target

More details can be found in <link to article>. 
Examples can be found in https://github.com/sam-cal/PixelateHistogramExample
