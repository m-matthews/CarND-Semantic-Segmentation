import os.path
import glob
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from shutil import copyfile
import scipy
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    return (graph.get_tensor_by_name(vgg_input_tensor_name),
            graph.get_tensor_by_name(vgg_keep_prob_tensor_name),
            graph.get_tensor_by_name(vgg_layer3_out_tensor_name),
            graph.get_tensor_by_name(vgg_layer4_out_tensor_name),
            graph.get_tensor_by_name(vgg_layer7_out_tensor_name))
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Attempted to prevent re-training the early vgg layers.  This did not improve the result.
#    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    print('\nLayer shapes:')
    print('> vgg_layer3_out: ', str(vgg_layer3_out.get_shape()))
    print('> vgg_layer4_out: ', str(vgg_layer4_out.get_shape()))
    print('> vgg_layer7_out: ', str(vgg_layer7_out.get_shape()))

    layer3_size = vgg_layer3_out.get_shape()[-1]
    layer4_size = vgg_layer4_out.get_shape()[-1]
    layer7_size = vgg_layer7_out.get_shape()[-1]

    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, layer7_size, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_7 = tf.layers.conv2d_transpose(conv_1x1_7, layer4_size, 4, strides=(2, 2), padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print('> conv_1x1_7: ', str(conv_1x1_7.get_shape()))
    print('> output_7: ', str(output_7.get_shape()))

    skip_4 = tf.add(output_7, vgg_layer4_out)
    output_4 = tf.layers.conv2d_transpose(skip_4, layer3_size, 4, strides=(2, 2), padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print('> skip_4: ', str(skip_4.get_shape()))
    print('> output_4: ', str(output_4.get_shape()))

    skip_3 = tf.add(output_4, vgg_layer3_out)
    output_3 = tf.layers.conv2d_transpose(skip_3, num_classes, 16, strides=(8, 8), padding='same',
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print('> skip_3: ', str(skip_3.get_shape()))
    print('> output_3: ', str(output_3.get_shape()))

    return output_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, optimizer, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print('\nTraining:')
    for epoch in range(epochs):
        for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
            # Note that keep_prob and learning rate are float values not TF Placeholders (found with Slack scanning as OK).
            _, loss = sess.run([train_op, cross_entropy_loss],
                                feed_dict={input_image:image, correct_label:label,
                                           keep_prob:0.5, learning_rate:0.0005})
            print('> Epoch {}, Batch {}, Loss={:.6f}.'.format(epoch, batch, loss))
    return
tests.test_train_nn(train_nn)

def augment(inpath, outpath):
    """
    Train neural network and print out the loss during training.
    :param inpath: Input path
    :param outpath: Output path to contain augmented images and data
    """
    print('\nAugmenting data:')
    if os.path.isdir(os.path.join(outpath, 'calib')):
        print('> Augmentation data already exists.')
    else:
        os.makedirs(os.path.join(outpath, 'calib'))
        os.makedirs(os.path.join(outpath, 'image_2'))
        os.makedirs(os.path.join(outpath, 'gt_image_2'))

        files = list(glob.iglob(os.path.join(inpath, 'calib/*.txt'))) + \
                list(glob.iglob(os.path.join(inpath, 'image_2/*.png'))) + \
                list(glob.iglob(os.path.join(inpath, 'gt_image_2/*.png')))

        for path in files:
            file = path.split('/')[-1]
            ext = file.split('.')[-1]
            folder = path.split('/')[-2]

            # Copy the original file.
            copyfile(os.path.join(inpath, folder, file),
                     os.path.join(outpath, folder, file))

            if ext=='png':
                # Read the images.
                image = scipy.misc.imread(path)

            # Mirror (left/right) image augmentation.
            filem = file.split('.')[0] + 'm.' + ext
            if ext=='png':
                scipy.misc.imsave(os.path.join(outpath, folder, filem), np.fliplr(image))
            else:
                copyfile(os.path.join(inpath, folder, file),
                         os.path.join(outpath, folder, filem))

        print('> Augmentation complete.')

    return

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Epoch set based on reviewing loss from previous testing iterations.
        epochs = 20
        # Experimented with batch_size:
        #  10 = terrible, 5 = good, 6 = similar to 5, 7 = worse, 4 = good.
        batch_size = 5

        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        augment(os.path.join(data_dir, 'data_road/training'),
                os.path.join(data_dir, 'data_road/training_aug'))

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training_aug'), image_shape)

        # TODO: Build NN using load_vgg, layers, and optimize function
        t_input, t_keep_prob, t_layer3_out, t_layer4_out, t_layer7_out = load_vgg(sess, vgg_path)
        t_final = layers(t_layer3_out, t_layer4_out, t_layer7_out, num_classes)
        logits, optimizer, cross_entropy_loss = optimize(t_final, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss,
                 t_input, correct_label, t_keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, t_keep_prob, t_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
