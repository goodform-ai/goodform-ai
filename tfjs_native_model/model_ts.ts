import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { parse } from 'csv-parse';

class LandmarksToEmbedding extends tf.layers.Layer {
    numKeyPoints: number;
    constructor(numKeyPoints:number = 17, config?: any) {
        super(config);
        this.numKeyPoints = numKeyPoints;
    }

    computeOutputShape(inputShape: number[]): number[] {
        return [inputShape[0], this.numKeyPoints * 2];
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs?: Record<string, any>): tf.Tensor {
        let input: tf.Tensor;
        if (Array.isArray(inputs)) {
            input = inputs[0];
        } else {
            input = inputs as tf.Tensor;
        }

        const reshapedInputs = tf.layers.reshape({ targetShape: [this.numKeyPoints, 3] }).apply(input) as tf.Tensor;
        const landmarks = reshapedInputs.slice([0, 0, 0], [-1, -1, 2]);
        const normalizedLandmarks = normalizePoseLandmarks(landmarks, this.numKeyPoints);
        const embedding = normalizedLandmarks.flatten();
        return embedding.reshape([-1, this.numKeyPoints * 2]);
    }

    getClassName(): string {
        return 'LandmarksToEmbedding';
    }
}

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

function getCenterPoint(landmarks: tf.Tensor, leftBodyPart: number, rightBodyPart: number): tf.Tensor {
    const left = tf.gather(landmarks, leftBodyPart, 1);
    const right = tf.gather(landmarks, rightBodyPart, 1);
    const center = left.mul(0.5).add(right.mul(0.5));
    return center;
}

function getPoseSize(landmarks: tf.Tensor, numKeyPoints:number = 17, torsoSizeMultiplier = 2.5): tf.Tensor {
    const hipsCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    const shouldersCenter = getCenterPoint(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER);
    const torsoSize = tf.norm(shouldersCenter.sub(hipsCenter));

    const poseCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    const poseCenterExpanded = poseCenter.expandDims(1);
    const poseCenterBroadcasted = tf.broadcastTo(poseCenterExpanded, [landmarks.shape[0], numKeyPoints, 2]);

    const distToPoseCenter = tf.gather(landmarks.sub(poseCenterBroadcasted), 0, 1);
    const maxDist = tf.max(tf.norm(distToPoseCenter, 1));

    const poseSize = tf.maximum(torsoSize.mul(torsoSizeMultiplier), maxDist);

    return poseSize;
}

function normalizePoseLandmarks(landmarks: tf.Tensor, numKeyPoints:number = 17): tf.Tensor {
    const poseCenter = getCenterPoint(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP);
    const poseCenterExpanded = poseCenter.expandDims(1);
    const poseCenterBroadcasted = tf.broadcastTo(poseCenterExpanded, [landmarks.shape[0], numKeyPoints, 2]);
    const centeredLandmarks = landmarks.sub(poseCenterBroadcasted);

    const poseSize = getPoseSize(centeredLandmarks, numKeyPoints);
    const normalizedLandmarks = centeredLandmarks.div(poseSize);

    return normalizedLandmarks;
}

async function loadPoseLandmarks(csvPath: string): Promise<{ X: tf.Tensor; y: tf.Tensor; classes: string[]; dataframe: any[] }> {
    return new Promise(async (resolve, reject) => {
        const dataframe: any[] = [];

        fs.createReadStream(csvPath)
            .pipe(
                parse({
                    delimiter: ',',
                    columns: true,
                })
            )
            .on('data', (row) => {
                dataframe.push(row);
            })
            .on('end', () => {
                const fileNameColumn = 'file_name';
                const classNameColumn = 'class_name';
                const classNoColumn = 'class_no';

                // Remove the file_name column
                dataframe.forEach((row) => {
                    delete row[fileNameColumn];
                });

                // Extract the list of class names
                const classNamesSet = new Set<string>();
                dataframe.forEach((row) => {
                    classNamesSet.add(row[classNameColumn]);
                });
                const classes = Array.from(classNamesSet);

                // Extract the labels
                const labels: number[] = [];
                dataframe.forEach((row) => {
                    labels.push(row[classNoColumn]);
                    delete row[classNoColumn];
                    delete row[classNameColumn];
                });

                // Convert the input features and labels into the correct format for training
                const X = tf.tensor(dataframe.map((row) => Object.values(row).map((value: any) => parseFloat(value))));
                const y = tf.oneHot(tf.tensor1d(labels, 'int32'), classes.length);

                resolve({ X, y, classes, dataframe });
            })
            .on('error', (error: any) => {
                reject(error);
            });
    });
}

interface Dataset {
    X_train: tf.Tensor;
    y_train: tf.Tensor;
    X_val: tf.Tensor;
    y_val: tf.Tensor;
    X_test: tf.Tensor;
    y_test: tf.Tensor;
    classes: string[];
}

async function loadData(): Promise<Dataset> {
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

function earlyStoppingCallback(model: tf.LayersModel, patience = 3) {
    let bestValAcc = -Number.MAX_VALUE;
    let count = 0;

    return {
        onEpochEnd: async (epoch: number, logs: { val_acc: number; }) => {
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

    const numKeyPoints = 17;
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
    const outputs = tf.layers.dense({ units: numClasses, activation: 'softmax' }).apply(dropout2 as tf.SymbolicTensor);

    // Create a model with the specified input and output layers
    const model = tf.model({ inputs: input, outputs: outputs as tf.SymbolicTensor });

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
    await model.save('file://./squat_classifier_ts');
})();
