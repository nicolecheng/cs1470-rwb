import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocessing import get_data, get_deep_learners_data
import sys
import argparse
from matplotlib import pyplot as plt

global dog_breeds

class RWB(tf.keras.Model):

    def __init__(self):
        super(RWB, self).__init__()    

        # hyperparameters 
        self.batch_size = 10
        self.regularizer = tf.keras.regularizers.l2(5e-4)
        self.epochs = 25
        self.model = tf.keras.Sequential()
        self.num_classes = 120


        # dimensions of below should be 500 but we ran oom:
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv2D(400, (5,5), strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=self.regularizer))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME"))
        self.model.add(tf.keras.layers.Reshape((self.batch_size,  -1)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1000, activation="tanh"))
        self.model.add(tf.keras.layers.Dense(100, activation="tanh")) 
        self.model.add(tf.keras.layers.Dense(120, activation="softmax"))

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9)



    def call(self, inputs):
        logits = self.model(inputs)
        return logits

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(labels,axis=1), logits))
        return loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    num_inputs = train_inputs.shape[0]
    start = 0
    end = model.batch_size
    i = 0

    while (end < num_inputs):
        cur_input_batch = train_inputs[start:end]
        cur_label_batch = train_labels[start:end]
        cur_input_batch = tf.image.random_flip_left_right(cur_input_batch)

        start = end
        end = end + model.batch_size
        i += 1


        with tf.GradientTape() as tape:
            predictions = model.call(cur_input_batch)
            loss = model.loss(predictions, cur_label_batch)

        if (i % 100 == 0):
            print("Loss: ", loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        

def test(model, test_inputs, test_labels):

    global dog_breeds

    num_inputs = test_inputs.shape[0]
    start = 0
    end = model.batch_size
    i = 0
    total_accuracy = 0
    all_predictions = []

    while (end < num_inputs):
        cur_input_batch = test_inputs[start:end]
        cur_label_batch = test_labels[start:end]

        start = end
        end = end + model.batch_size
        i += 1

        predictions = model.call(cur_input_batch)
        batch_accuracy = model.accuracy(predictions, cur_label_batch)
        total_accuracy += batch_accuracy

        all_predictions += list(tf.argmax(predictions, axis=1).numpy())


    test_labels = list(tf.argmax(test_labels, axis=1).numpy())

    f = open(os.path.join("output","test_results.txt"), "w")
    for i in range(len(all_predictions)):
        predicted = str(dog_breeds[all_predictions[i]])
        actual = str(dog_breeds[test_labels[i]])
        spaces = (35 - len(predicted)) * " "
        f.write("Predicted: "+predicted+spaces+"Actual: "+actual+"\n")
    f.close()

    avg_accuracy = total_accuracy / len(test_labels)

    return avg_accuracy


def main():

    parser = argparse.ArgumentParser(description='Team RWB: Dog Breed Classifier')
    parser.add_argument('--restore-checkpoint', action='store_true',
                        help='Use this flag if you want to resuming training from a previously-saved checkpoint')
    parser.add_argument('--deep-learners', default=False, action='store_true',
                        help='Use in conjunction w checkpt if want to run the model on photos of DL prof and HTAs')
    args = parser.parse_args()


    global dog_breeds

    # create Model
    m = RWB()
    restore = args.restore_checkpoint
    print("G:", tf.test.is_gpu_available())

    checkpoint_dir = "./checkpoints"
    checkpoint = tf.train.Checkpoint(m=m)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    train_inputs, train_labels, test_inputs, test_labels, dog_breeds = get_data(m.batch_size)

    if args.deep_learners and restore:
        deep_learner_names, deep_learner_images = get_deep_learners_data()
        f = open(os.path.join("output","deeplearner_predictions.txt"), "w")
        deep_learner_images = tf.convert_to_tensor(deep_learner_images)
        probs = m.call(deep_learner_images)
        preds = tf.argmax(probs,axis=1)
        f.write("preds\n" + str(preds))
        for i in range(len(preds)):
            predicted = str(dog_breeds[preds[i]])
            actual = deep_learner_names[i]
            spaces = (35 - len(predicted)) * " "
            f.write("Predicted: "+predicted+spaces+"Actual: "+actual+"\n")
        
        f.close()
        return

    if restore:
        checkpoint.restore(manager.latest_checkpoint)

    else:
        # load train and test data
        for i in range(m.epochs):
            print("************ EPOCH  %d *************" % i)
            train(m, train_inputs, train_labels)
            print("Saving checkpoint at epoch...")
            manager.save()

    print("Testing...")
    acc = test(m, test_inputs, test_labels)
    print("model received an accuracy of: %s" % str(acc))


if __name__ == '__main__':
    main()
