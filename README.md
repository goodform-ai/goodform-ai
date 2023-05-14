## Model Overview and Design Choices

The goal of this model is to classify poses based on the input pose landmarks data. The input data consists of 17 pose landmarks, and each landmark has an x-coordinate, y-coordinate, and a confidence score.

### Preprocessing: LandmarksToEmbedding Layer

Before feeding the pose landmarks data into the model, a custom preprocessing layer called `LandmarksToEmbedding` is used. This layer is responsible for the following operations:

1. **Reshaping**: The input tensor is reshaped from a 1D tensor with shape `[51]` to a 2D tensor with shape `[17, 3]`. This is done to separate the x, y, and score values for each of the 17 landmarks.

2. **Removing Confidence Score**: The score values are removed by slicing the tensor, leaving only the x and y coordinates. This is because the model analyzes the pose in a 2D space, and the score values do not contribute to the pose's spatial information.

3. **Normalizing Landmarks**: The landmarks are normalized by translating the pose center to `(0, 0)` and scaling the landmarks to a constant pose size. This normalization step ensures that the model is invariant to translation and scale, making it more robust to different poses and input data variations.

4. **Flattening and Reshaping**: The normalized landmarks tensor is flattened into a 1D tensor (embedding) and then reshaped to keep the batch size information. Flattening is done to convert the 2D landmarks tensor into a format that can be fed into the subsequent dense layers for classification. Reshaping with a shape of `[-1, 34]` ensures that TensorFlow retains the batch size information during runtime.

The `LandmarksToEmbedding` layer is a custom layer that preprocesses the input landmarks data to create an embedding that can be fed into the subsequent dense layers for classification.

#### computeOutputShape()

The `computeOutputShape` function is a method of the custom layer that is responsible for defining the shape of the output tensor for a given input shape. This information is necessary for TensorFlow to correctly allocate memory for the output tensor and perform shape inference during the model compilation.

In the `LandmarksToEmbedding` layer, the `computeOutputShape` function takes the input shape `[51]` (17 landmarks with 3 items each - x, y, and score) and returns the output shape `[34]`. The output shape has 34 elements because the custom layer removes the score from each landmark, leaving only the x and y coordinates for the 17 landmarks.

#### call()

The `call` function is the core method of the custom layer that defines the forward pass of the layer. It takes an input tensor (or an array of input tensors) and returns the output tensor after applying the layer's operations.

In the `LandmarksToEmbedding` layer, the `call` function performs the following operations:

1. Reshape the input tensor to a shape of `[17, 3]`. This is done to facilitate the removal of the score from each landmark.

2. Slice the tensor to remove the score, keeping only the x and y coordinates. The score is removed because the analysis focuses on the pose in a 2D space (x and y coordinates only).

3. Normalize the landmarks using the `normalizePoseLandmarks` function. This step translates the pose center to `(0, 0)` and scales the landmarks to a constant pose size, making the model invariant to translation and scale.

4. Flatten the normalized landmarks into an embedding. Flattening is done to convert the normalized 2D landmarks tensor into a 1D tensor (embedding), which can be fed into the subsequent dense layers for classification.

5. Reshape the embedding to keep the batch size information. This step ensures that the output tensor has the correct shape for processing by the subsequent layers.

### Normalizing Landmarks

Normalizing the landmarks is an essential preprocessing step that improves the model's ability to learn from the input data. It involves translating the pose center to `(0, 0)` and scaling the landmarks to a constant pose size. The purpose of this normalization is to make the model invariant to translation and scale, making it more robust to different poses and input data variations.

#### Why Normalize Landmarks?

1. **Translation Invariance**: By translating the pose center to `(0, 0)`, the model can learn features that are independent of the position of the pose in the input space. This invariance is important because the model should be able to recognize a pose regardless of its location in the input data.

2. **Scale Invariance**: By scaling the landmarks to a constant pose size, the model can learn features that are independent of the size of the pose in the input data. This invariance is crucial because the model should be able to recognize a pose regardless of its size (e.g., if the person is closer or farther away from the camera).

Here's the explanation of the normalization process and utility functions arranged in the order they are executed:

#### How to Normalize Landmarks?

The normalization of landmarks involves two main steps: translation and scaling. This process is implemented using a series of utility functions, which are explained below in the order they are executed:

1. **normalizePoseLandmarks**: This function normalizes the pose landmarks based on the pose center and pose size. It takes `landmarks` as input. First, it calculates the pose center using the `getCenterPoint` function and expands its dimensions to make it compatible for broadcasting. It broadcasts the pose center to match the shape of the landmarks tensor and subtracts the pose center from each landmark to center the pose, creating the `centeredLandmarks`. Next, it calculates the pose size using the `getPoseSize` function and divides the `centeredLandmarks` by the pose size to normalize the pose, creating the `normalizedLandmarks`. The `normalizedLandmarks` tensor is returned.

2. **getCenterPoint**: This function calculates the center point between two body parts (left and right) by taking the average of their coordinates. It takes `landmarks`, `leftBodyPart`, and `rightBodyPart` as input arguments. Using `tf.gather`, it extracts the coordinates of the left and right body parts. Then, it calculates the center point by averaging the coordinates of the left and right body parts and returns the center point tensor.

3. **getPoseSize**: This function calculates the pose size based on the torso size and the maximum distance from the pose center to any body part. It takes `landmarks` and an optional `torsoSizeMultiplier` (default: 2.5) as input arguments. First, it calculates the center points of the hips and shoulders using the `getCenterPoint` function. Then, it calculates the torso size as the distance between the hips center and the shoulders center. Next, it calculates the pose center and expands its dimensions to make it compatible for broadcasting. It broadcasts the pose center to match the shape of the landmarks tensor and calculates the distance from each body part to the pose center. Finally, it finds the maximum distance from the pose center to any body part and calculates the pose size as the maximum of the torso size multiplied by the `torsoSizeMultiplier` and the maximum distance to the pose center. The pose size tensor is returned.

By normalizing the landmarks, the model can focus on learning the spatial relationships between the landmarks and the pose itself, rather than being influenced by the position or size of the pose in the input data. This preprocessing step enhances the model's ability to generalize to unseen data and recognize a wider range of poses.

### Model Layers

In this model, several layers are used to process the input landmarks data and perform classification. Here, we'll provide a definition for each layer type used, explaining what it is, why it's used, and how it works.

1. **LandmarksToEmbedding layer (custom layer)**: This custom layer preprocesses the input landmarks data to create an embedding. It removes the score from each landmark, normalizes the landmarks, and flattens the result into a 1D tensor. This layer is crucial for preparing the raw pose landmarks data for processing by the subsequent dense layers. By creating an embedding, it simplifies the input data and ensures that the model focuses only on the x and y coordinates of the landmarks.

2. **Dense (fully connected) layers**: Dense layers, also known as fully connected layers, connect each neuron in the layer to every neuron in the previous layer. They are used for classification because they can learn complex patterns in the input data. The dense layers in this model use ReLU6 as an activation function, a variation of the popular ReLU activation function with a maximum value of 6. This helps control the output range and avoid potential issues with exploding gradients.

3. **Dropout layers**: Dropout layers are used for regularization during the training process. They randomly "drop out" or deactivate a proportion of neurons during each training iteration, helping the model generalize better to unseen data. This reduces the risk of overfitting, a common problem in deep learning models where the model becomes too specialized to the training data and performs poorly on new, unseen data.

###

### Model Architecture

The model architecture consists of a series of dense (fully connected) layers, dropout layers, and the custom LandmarksToEmbedding layer for preprocessing, as well as an output layer for classification.

1. **LandmarksToEmbedding layer**: This custom layer preprocesses the input landmarks data to create an embedding. The layer removes the score from each landmark, normalizes the landmarks, and flattens the result into a 1D tensor. This step is crucial for preparing the raw pose landmarks data for processing by the subsequent dense layers.

2. **Dense layer 1 (128 units, ReLU6 activation)**: This layer is a fully connected dense layer with 128 units and ReLU6 activation. The choice of 128 units is somewhat arbitrary and can be adjusted based on the complexity of the classification task. ReLU6 is used as an activation function because it is a variation of the popular ReLU activation function with a maximum value of 6. This layer learns non-linear features from the input embedding and serves as the first step in the classification process, which also makes it the first hidden layer.

3. **Dropout layer 1 (0.5 rate)**: This layer applies dropout with a rate of 0.5 to prevent overfitting. Dropout is a regularization technique that randomly "drops out" or deactivates a proportion of neurons during training, which helps the model generalize better to unseen data. The rate of 0.5 means that approximately 50% of the neurons will be deactivated during each training iteration. This is the second hidden layer.

4. **Dense layer 2 (64 units, ReLU6 activation)**: This layer is another fully connected dense layer with 64 units and ReLU6 activation. The choice of 64 units is also somewhat arbitrary and can be adjusted based on the complexity of the classification task. This layer further processes the features learned by the first dense layer, adding more complexity to the model's decision-making process. This is the third hidden layer.

5. **Dropout layer 2 (0.5 rate)**: This layer applies dropout with a rate of 0.5 to prevent overfitting, just like the first dropout layer. Adding multiple dropout layers throughout the model helps improve the model's ability to generalize to new data. This is the last hidden layer.

6. **Output layer**: The output layer has a number of units equal to the number of classes and uses a softmax activation for classification. Softmax activation is used in the final layer because it's a common activation function for multi-class classification problems. Softmax converts the output of the layer into probabilities that sum up to 1, making it easier to interpret the model's predictions. This layer takes the processed features from the previous layers and makes the final decision on which class each input example belongs to.

Hidden layers refer to layers that are neither input nor output layers. In this model, the hidden layers include the dense and dropout layers. These layers learn complex patterns and features from the input data, which are then used for classification.

### Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. The Adam optimizer is a popular choice because it is an adaptive learning rate optimization algorithm that works well in practice and requires little tuning. Categorical cross-entropy loss is a common choice for multi-class classification problems, as it measures the dissimilarity between the predicted class probabilities and the true class labels. The model also tracks the accuracy metric during training to monitor its performance.
