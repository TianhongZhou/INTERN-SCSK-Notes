# CNN

## What is CNN?

- Short for: Convolutional Neural Networks
- Comprised of node layers, containing an input layer, one or more hidden layers, and an output layer
- Each node connects to another and has an associated weight and threshold



## How does CNN work?

- Three main types of layers
  - Convolutional layer
  - Pooling layer
  - Fully-connected layer
- With each layer, CNN increases in its complexity, identifying greater portions of the image
- Earlier layers focus on simple features
- As image data progresses through the layers of the CNN, it starts to recognize larger elements or shapes of the object until it finally identifies the intended object
- Convolutional layer
  - Core building block of CNN
  - Where majority of computation occurs
  - Requires input data, a filter, and a feature map
  - Input will have three dimensions
    - Height
    - Width
    - Depth
  - Feature detector, also known as a kernel or a filter
    - Move across the receptive fields of the image
    - Check if the feature is present
    - 2D array of weights
    - Filter size is typically a 3x3 matrix
  - Dot product is calculated between input pixels and filter
  - Filter shifts by a stride, repeating the process until the kernel has swept across the entire image
  - Final output from series of dot products from the input and filter is known as a feature map, activation map, or a convolved feature
  - Output value in feature map only need to connect to receptive field
  - Convolutional layers are commonly referred to as "partially connected" layers
  - Weights in the feature detector remain fixed as it moves across the image, also known as parameter sharing
  - Three hyperparameters which affect the volume size of the output that need to be set before the training of the neural network begins
    - Number of filters
      - Affects the depth of the output
    - Stride
      - Is the distance, or number of pixels, that the kernel moves over the input matrix
      - Strides values of two or greater is rare
      - A larger stride yields a smaller output
    - Zero-padding
      - Used when the filters do not fit the input image
      - Sets all elements that fall outside of the input matrix to zero
      - Produces a larger or equally sized output
      - Three types of padding
        - Valid padding
          - Also known as no padding
          - Last convolution is dropped if dimensions do not align
        - Same padding
          - Ensures that the output layer has the same size as the input layer
        - Full padding
          - Increases the size of the output by adding zeros to the border of the input

- Pooling layer
  - Also known as downsampling
  - Reduces the number of parameters in the input
  - Pooling operation sweeps a filter across the entire input
  - This filter does not have any weights
  - Two main types of pooling
    - Max pooling
      - As the filter moves across the input, it selects the pixel with the maximum value to send to the output array
      - Tends to be used more often compared to average pooling
    - Average pooling
      - As the filter moves across the input, it calculates the average value within the receptive field to send to the output array
  - Help to reduce complexity, improve efficiency, and limit risk of overfitting
- Fully-connected layer
  - Each node in the output layer connects directly to a node in the previous layer
  - Performs the task of classification based on the features extracted through the previous layers and their different filters
  - Usually leverage a softmax activation function to classify inputs appropriately, producing a probability from 0 to 1



## Different types of CNNs

- 1D CNN
  - CNN kernel moves in one direction
  - Used on time-series data
- 2D CNN
  - CNN kernel moves in two direction
  - Used with image labelling and processing
- 3D CNN
  - CNN kernel moves in three direction
  - Used on 3D images like CT scans and MRIs