# Frequently Asked Questions

1. __Do the solution need to run on Lava?__
    Solutions must run inference on the test set in the Lava framework. Our intention here is to increase neuromorphic software convergence, reproducibility, and extensibility, which has been lacking in the neuromorphic community. Lava provides convenient abstractions for mapping algorithms efficiently to neuromorphic hardware—which we certainly encourage challenge participants to take advantage of — however, importantly, Lava is also very flexible and arbitrary code can be wrapped into a Lava process. Also please note that there is no requirement to use Lava for model training.

2. __Can we use additional/custom data for training?__
    Yes, we encourage participants to utilize the Microsoft DNS dataloaders for training data and focus on neuromorphic algorithm innovation; however, we do not restrict participants to only use the Microsoft DNS Dataset. External data and data augmentation are allowed. The models will eventually be evaluated on test sets (which will be released later). Test sets cannot be changed.

3. __Is preprocessing of the input audio allowed before the encoder?__
    Preprocessing of the input audio waveforms before the encoder is not allowed. All preprocessing is considered part of the encoder. Likewise for postprocessing and the decoder.

4. __[Can we submit results under multiple hyperparameter configurations to better showcase our models](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/issues/3)?__
    Yes, we welcome this. However, please present all results in a single [writeup](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#solution-writeup).
