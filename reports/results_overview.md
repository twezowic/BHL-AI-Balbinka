### Predicting Solar Flares through a regression model.

## Data
the model takes satellite images as a data, the images were taken using different method in 4 different time periods before the eclipse, so to choose the data, one needs to edit the lookingFor variable with the name of the set of pictures to add to the model.

## Input
After testing, the magnetogram images were chosen as the input for the model.

## Output
The output of a model is a prediction of the maximum energy of a solar flare.
The output is normalized, so the energy is 10^(output - 9)

### Data tests
Most tests were run on a dataset of reduced scale

A couple batches of images were tested - magnetogram, continuum, __94 + __131 + __171
From those, the best result were had while using magnetogram dataset.
The model reached positive hit ratio of 22%, with the epsilon as 1/10, so the random generator would hit a positive ration of ~16%
