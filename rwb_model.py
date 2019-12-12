import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocessing import get_data
import sys
from matplotlib import pyplot as plt

global dog_breeds

class RWB(tf.keras.Model):
    #super(RWB, self).__init__()    
    def __init__(self):
        super(RWB, self).__init__()    
        #hyper paramters 
        self.batch_size = 10
        self.regularizer = tf.keras.regularizers.l2(5e-4)
        self.epochs = 25
        self.model = tf.keras.Sequential()
        self.num_classes = 120


        # dimensions of below should be 500:
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
        self.model.add(tf.keras.layers.Dense(1000, activation="tanh")) #need to make 1000
        self.model.add(tf.keras.layers.Dense(100, activation="tanh")) #need to make 1000
        self.model.add(tf.keras.layers.Dense(120, activation="softmax"))

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9)#5e-4, momentum=0.9)



    def call(self, inputs):

        logits = self.model(inputs)
        # print("final output shape", logits.shape)
        return logits

    def loss(self, logits, labels):
        #print(labels.shape)
        #print(logits.shape)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(labels,axis=1), logits))
        return loss

    def accuracy(self, logits, labels):
        #print("preds of first 10: ", tf.argmax(logits, 1)[:10])
        #print("labels of first 10:", tf.argmax(labels, 1)[:10])
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        #print(correct_predictions)
        return tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    #indices = tf.random.shuffle(range(len(train_inputs)))
    #train_inputs = tf.gather(train_inputs, indices)
    #train_labels = tf.gather(train_labels, indices)

    num_inputs = train_inputs.shape[0]
    #print("total number of training images:", num_inputs)
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
        #print("batch", cur_label_batch.shape)

        start = end
        end = end + model.batch_size
        i += 1

        predictions = model.call(cur_input_batch)
        batch_accuracy = model.accuracy(predictions, cur_label_batch)
        total_accuracy += batch_accuracy

        #viz_inputs, viz_labels = cur_input_batch, cur_label_batch#test_inputs[:10], test_labels[:10]
        #visualize_results(viz_inputs, predictions, viz_labels)#m.call(viz_inputs), viz_labels)

        all_predictions += list(tf.argmax(predictions, axis=1).numpy())

    test_labels = list(tf.argmax(test_labels, axis=1).numpy())
    #print(len(all_predictions),len(test_labels))
    
    f = open(os.path.join("output","test_results.txt"), "w")
    for i in range(len(all_predictions)):
        predicted = str(dog_breeds[all_predictions[i]])
        actual = str(dog_breeds[test_labels[i]])
        spaces = (35 - len(predicted)) * " "
        f.write("Predicted: "+predicted+spaces+"Actual: "+actual+"\n")
    f.close()


    avg_accuracy = total_accuracy / len(test_labels)

    return avg_accuracy


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        #plt.show()
        plt.savefig(os.path.join("output","result_"+str(cur_num)+".png"))
        
        
def main():

    global dog_breeds
    # create Model
    m = RWB()
    restore = False
    print("G:", tf.test.is_gpu_available())

    checkpoint_dir = "./checkpoints"
    checkpoint = tf.train.Checkpoint(m=m)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    train_inputs, train_labels, test_inputs, test_labels, dog_breeds = get_data(m.batch_size)
        
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
