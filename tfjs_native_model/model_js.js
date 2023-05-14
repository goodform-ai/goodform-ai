// Import required libraries
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { parse } = require('csv-parse');

// Define the custom layer LandmarksToEmbedding that extends the base layer
// inheritance is being performed from the parent class AKA tf.layers.Layer
class LandmarksToEmbedding extends tf.layers.Layer {
    constructor(numKeyPoints, config) {
        super(config); // Call the parent class constructor
        this.numKeyPoints = numKeyPoints;
    }

    // METHOD OVERRIDING from inherited methods
    // Override computeOutputShape to calculate the output shape of the custom layer given an input shape.
    // It is necessary for TensorFlow.js to know the shape of the output tensor after passing through the layer.
    computeOutputShape(inputShape) {
        return [inputShape[0], this.numKeyPoints * 2]; // Output shape after processing (34 because of 17 landmarks in 2D space: 17 * 2 = 34)
    }

    // Override call() to define the forward pass of the custom layer.
    // It takes an input tensor (or an array of input tensors) and returns the output tensor after applying the layer's operations.
    call(inputs, kwargs) {
        let input = inputs;
        // Handle the case where inputs are in an array or not
        if (Array.isArray(inputs)) {
            input = inputs[0];
        } else {
            input = inputs;
        }

        // Reshape the input 1D tensor to a shape of a 2D tensor with [this.numKeyPoints, 3] (this.numKeyPoints landmarks with 3 items each)
        // The input shape is [51] because the pose landmarks input data has this.numKeyPoints points, and each point has 3 items (x, y, and score).
        // This reshaping step makes it easier to work with the landmarks data and perform further processing,
        // like removing the confidence scores, normalizing the landmarks, and flattening them into an embedding.
        const reshapedInputs = tf.layers.reshape({ targetShape: [this.numKeyPoints, 3] }).apply(input);
        // Slice the tensor to remove the score, keeping only the x and y coordinates.
        // The score is removed because we want to analyze the pose in a 2D space (x and y coordinates only).
        const landmarks = reshapedInputs.slice([0, 0, 0], [-1, -1, 2]);
        // Normalize the landmarks
        const normalizedLandmarks = normalizePoseLandmarks(landmarks, numKeyPoints);
        // Flatten the normalized landmarks into an embedding
        // Flattening is done to convert the normalized 2D landmarks tensor into a 1D tensor (embedding),
        // which can be fed into the subsequent dense layers for classification.
        const embedding = normalizedLandmarks.flatten();

        // Reshape the embedding to keep the batch size information which is 2d
        // Reshape to 2D tensor. '-1' automatically calculates the first dimension based on the total size of the tensor. Basically a row
        // The second dimension size 'this.numKeyPoints * 2' are basically the columns to a row.

        // the new output size would be a multiple of numKeyPoints so [2,34], 34 being how many landmark / keypoints be have because we removed the score item
        // 2 being how many rows we have (see csv training data for basic eample)

        // if you have a 1D tensor with 100 elements and you reshape it to [-1, 10], 
        // you will get a 2D tensor with shape [10, 10]. If you reshape it to [-1, 20], 
        // you will get a 2D tensor with shape [5, 20]

        // The -1 tells TensorFlow to calculate the size of that dimension such that the total size 
        // of the tensor (i.e., the total number of elements) remains the same.
        return embedding.reshape([-1, this.numKeyPoints * 2]);
    }

    // Define the layer's class name (used for serialization)
    static get className() {
        return 'LandmarksToEmbedding';
    }
}

// Register the custom layer for serialization
tf.serialization.registerClass(LandmarksToEmbedding);

tf.serialization.registerClass(LandmarksToEmbedding);

const BodyPart = {
    NOSE: 0,
    LEFT_EYE: 1,
    RIGHT_EYE: 2,
    LEFT_EAR: 3,
    RIGHT_EAR: 4,
    LEFT_SHOULDER: 5,
    RIGHT_SHOULDER: 6,
    LEFT_ELBOW: 7,
    RIGHT_ELBOW: 8,
    LEFT_WRIST: 9,
    RIGHT_WRIST: 10,
    LEFT_HIP: 11,
    RIGHT_HIP: 12,
    LEFT_KNEE: 13,
    RIGHT_KNEE: 14,
    LEFT_ANKLE: 15,
    RIGHT_ANKLE: 16,
};

// Utility function to find the center point between two body parts (left and right)
function getCenterPoint(landmarks, leftBodyPart, rightBodyPart) {
    // Extract the coordinates of the left and right body parts using tf.gather
    // tf.gather extract slices or elements from a tensor based on specified indices
    // in this case we are extracting the cords of leftBodyPart and leftBodyPart from the landmarks tensor
    const left = tf.gather(landmarks, leftBodyPart, 1);
    const right = tf.gather(landmarks, rightBodyPart, 1);
    // Calculate the center point by averaging the coordinates of the left and right body parts
    const center = left.mul(0.5).add(right.mul(0.5));
    return center;
}

// Utility function to calculate the pose size based on the torso size and the maximum distance from the pose center to any body part
function getPoseSize(landmarks, numKeyPoints, torsoSizeMultiplier = 2.5) {
    // Calculate the center points of the hips and shoulders
    const hipsCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    const shouldersCenter = getCenterPoint(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER);
    // Calculate the torso size as the distance between the hips center and the shoulders center
    const torsoSize = tf.norm(shouldersCenter.sub(hipsCenter));

    // Calculate the pose center as the center point between the left and right hips
    const poseCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    // Expand the dimensions of the pose center to make it compatible for broadcasting
    const poseCenterExpanded = poseCenter.expandDims(1);
    // Broadcast the pose center to match the shape of the landmarks tensor
    const poseCenterBroadcasted = tf.broadcastTo(poseCenterExpanded, [landmarks.shape[0], numKeyPoints, 2]);

    // Calculate the distance from each body part to the pose center
    const distToPoseCenter = tf.gather(landmarks.sub(poseCenterBroadcasted), 0, 1);
    // Find the maximum distance from the pose center to any body part
    const maxDist = tf.max(tf.norm(distToPoseCenter, 1));

    // Calculate the pose size as the maximum of the torso size multiplied by the torsoSizeMultiplier and the maximum distance to the pose center
    const poseSize = tf.maximum(torsoSize.mul(torsoSizeMultiplier), maxDist);

    return poseSize;
}

// Utility function to normalize the pose landmarks based on the pose center and pose size
function normalizePoseLandmarks(landmarks, numKeyPoints = 17) {
    // Calculate the pose center as the center point between the left and right hips
    const poseCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    // Expand the dimensions of the pose center to make it compatible for broadcasting
    const poseCenterExpanded = poseCenter.expandDims(1);
    // Broadcast the pose center to match the shape of the landmarks tensor
    const poseCenterBroadcasted = tf.broadcastTo(poseCenterExpanded, [landmarks.shape[0], numKeyPoints, 2]);
    // Subtract the pose center from each landmark to center the pose
    const centeredLandmarks = landmarks.sub(poseCenterBroadcasted);

    // Calculate the pose size
    const poseSize = getPoseSize(centeredLandmarks, numKeyPoints);
    // Divide the centered landmarks by the pose size to normalize the pose
    const normalizedLandmarks = centeredLandmarks.div(poseSize);

    return normalizedLandmarks;
}

async function loadPoseLandmarks(csvPath) {
    return new Promise((resolve, reject) => {
        const data = [];
        fs.createReadStream(csvPath)
            .pipe(parse({ columns: true, delimiter: ',' }))
            .on('data', (row) => {
                data.push(row);
            })
            .on('end', () => {
                const classSet = new Set();
                const landmarks = [];
                const labels = [];

                data.forEach((row) => {
                    classSet.add(row.class_name);
                    labels.push(Number(row.class_no));
                    const rowLandmarks = Object.values(row)
                        .slice(2, -1)
                        .map((val) => parseFloat(val));
                    landmarks.push(rowLandmarks);
                });

                const classes = Array.from(classSet);
                const X = tf.tensor2d(landmarks);
                const y = tf.oneHot(tf.tensor1d(labels, 'int32'), classes.length).cast('float32');

                console.log('X shape:', X.shape);
                console.log('y shape:', y.shape);

                resolve({ X, y, classes });
            });
    });
}

async function loadData() {
    const trainCsvPath = 'train_data.csv';
    const testCsvPath = 'test_data.csv';

    const { X: X_train_all, y: y_train_all, classes } = await loadPoseLandmarks(trainCsvPath);
    const { X: X_test, y: y_test } = await loadPoseLandmarks(testCsvPath);

    // Split training data (X_train_all, y_train_all) into (X_train, y_train) and (X_val, y_val)
    const splitIndex = Math.floor(0.85 * X_train_all.shape[0]);
    const X_train = X_train_all.slice([0, 0], [splitIndex, -1]);
    const y_train = y_train_all.slice([0, 0], [splitIndex, -1]);
    const X_val = X_train_all.slice([splitIndex, 0], [-1, -1]);
    const y_val = y_train_all.slice([splitIndex, 0], [-1, -1]);

    return { X_train, y_train, X_val, y_val, X_test, y_test, classes };
}

function earlyStoppingCallback(model, patience = 3) {
    let bestValAcc = -Number.MAX_VALUE;
    let count = 0;

    return {
        onEpochEnd: async (epoch, logs) => {
            if (logs.val_acc > bestValAcc) {
                bestValAcc = logs.val_acc;
                count = 0;
            } else {
                count += 1;
            }

            if (count >= patience) {
                console.log(`\nEarly stopping at epoch ${epoch + 1}`);
                model.stopTraining = true;
            }
        },
    };
}

(async () => {
    const { X_train, y_train, X_val, y_val, X_test, y_test, classes } = await loadData();
    const numClasses = classes.length;

    numKeyPoints = 17;
    // Each layer in the TensorFlow.js model returns a tensor

    // Define the input tensor shape ((51 or lower) values corresponding to (17 landmarks or numKeyPoints) with 3 dimensions each)
    // The input shape of 51 comes from the pose landmarks data that you're working with.
    // Each pose consists of (17 landmarks or numKeyPoints), and each landmark has 3 dimensions (x, y, and score).
    // Therefore, the total number of values in the input tensor for a single pose is (17 landmarks or numKeyPoints) * 3 = 51.
    const input = tf.input({ shape: [numKeyPoints * 3] });
    // Apply the custom LandmarksToEmbedding layer to the input
    // apply() method is used to apply a layer to the input tensor(s).
    // This method takes care of connecting the input tensor to the layer, and it returns the output tensor generated by the layer.
    const embedding = new LandmarksToEmbedding(numKeyPoints).apply(input);
    // Define the first dense layer with 128 units and ReLU6 activation

    // HIDDEN LAYERS
    // The choice of 128 units is somewhat arbitrary and can be adjusted based on the complexity of the classification task.
    // ReLU6 is used as an activation function because it is a variation of the popular ReLU activation function with a maximum value of 6.
    const denseLayer1 = tf.layers.dense({ units: 128, activation: 'relu6' }).apply(embedding);
    // Apply a dropout layer with a rate of 0.5 to prevent overfitting
    // Dropout is a regularization technique that randomly "drops out" or deactivates a proportion of neurons during training,
    // which helps the model generalize better to unseen data also known as overfitting. The rate of 0.5 means that approximately 50% of the neurons will be deactivated during each training iteration.
    const dropout1 = tf.layers.dropout({ rate: 0.5 }).apply(denseLayer1);
    // Define the second dense layer with 64 units and ReLU6 activation
    // The choice of 64 units is also somewhat arbitrary and can be adjusted based on the complexity of the classification task.
    const denseLayer2 = tf.layers.dense({ units: 64, activation: 'relu6' }).apply(dropout1);
    // Apply a dropout layer with a rate of 0.5 to prevent overfitting
    const dropout2 = tf.layers.dropout({ rate: 0.5 }).apply(denseLayer2);
    
    // Define the output layer with a number of units equal to the number of classes and a softmax activation for classification
    // Softmax activation is used in the final layer because it's a common activation function for multi-class classification problems.
    // Softmax converts the output of the layer into probabilities that sum up to 1, making it easier to interpret the model's predictions.
    const outputs = tf.layers.dense({ units: numClasses, activation: 'softmax' }).apply(dropout2);

    // Create a model with the specified input and output layers
    const model = tf.model({ inputs: input, outputs: outputs });

    // Compile the model with the specified optimizer, loss function, and metric
    // The optimizer 'adam' is a popular choice for training deep learning models because it adapts the learning rate for each parameter during training,
    // leading to faster convergence and improved performance.
    // The loss function 'categoricalCrossentropy' is used because it is suitable for multi-class classification tasks.
    // It measures the dissimilarity between the true labels (one-hot encoded) and the predicted probabilities.
    // The metric 'accuracy' is used to track the model's performance during training. It measures the proportion of correctly classified instances out of the total instances.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    console.log('Data loaded and split.');
    console.log({ classes });

    // // Train the model with training data and validate using validation data
    await model.fit(X_train, y_train, {
        epochs: 200, // Adjust the number of epochs as needed
        batchSize: 16, // Adjust the batch size as needed
        validationData: [X_val, y_val],
        callbacks: [earlyStoppingCallback(model, 20)],
    });
    await model.save('file://./squat_classifier_js');
})();
