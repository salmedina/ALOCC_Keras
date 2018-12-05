from __future__ import print_function, division

import logging

import keras.backend as K
import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.utils import shuffle

from kh_tools import *
from utils import *

# Tensorflow configurator
config = tf.ConfigProto()
# Allocate gpu memory on-demand
config.gpu_options.allow_growth = True
# Let it eat the whole gpu, just not since the beginning
config.gpu_options.per_process_gpu_memory_fraction = 1.0
# Set a session with the new configuration
K.tensorflow_backend.set_session(tf.Session(config=config))

class ALOCC_Model():
    def __init__(self,
                 input_height=28,input_width=28,
                 output_height=28, output_width=28,
                 attention_label=1,
                 is_training=True,
                 z_dim=100, gf_dim=16, df_dim=16, c_dim=3,
                 dataset_name=None, dataset_address=None, input_fname_pattern=None,
                 checkpoint_dir='checkpoint', log_dir='log', sample_dir='sample',
                 r_alpha = 0.2, kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1, n_fetch_data=10):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16]
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16]
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'mnist', 'virat', or custom defined name.
        :param dataset_address: path to dataset folder or file. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param log_dir: log directory for training, can be later viewed in TensorBoard.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset.
        """

        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.is_training = is_training

        self.r_alpha = r_alpha

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.dataset_address = dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label
        if self.is_training:
            logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)
            if self.dataset_name == 'mnist':
                (X_train, y_train), (_, _) = mnist.load_data()
                # Make the data range between 0~1.
                X_train = X_train / 255
                specific_idx = np.where(y_train == self.attention_label)[0]
                self.data = X_train[specific_idx].reshape(-1, 28, 28, 1)
                self.c_dim = 1
            elif self.dataset_name == 'virat':
                print('Loading VIRAT dataset', self.dataset_address)
                dataset = np.load(open(self.dataset_address, 'rb'))
                X_train, y_train = dataset['images'], dataset['labels']
                X_train = X_train / 255.0
                specific_idx = np.where(y_train == self.attention_label)[0]
                self.data = X_train[specific_idx].reshape(-1, self.input_width, self.input_height, 1)
                neg_idx = np.random.choice(np.where(y_train != self.attention_label)[0], len(X_train)//2, replace=False)
                self.neg_data = shuffle(X_train[neg_idx].reshape(-1, self.input_width, self.input_height, 1))
                self.c_dim = 1
            else:
                assert('Unknown dataset')

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_generator(self, input_shape):
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape, name='z')
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2, kernel_size = 5, strides=2, padding='same', name='g_encoder_h0_conv')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4, kernel_size = 5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8, kernel_size = 5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        x = Conv2D(self.gf_dim*1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim*1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim*2, kernel_size=3, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.c_dim, kernel_size=5, activation='sigmoid', padding='same')(x)
        return Model(image, x, name='R')

    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=self.df_dim, kernel_size = 5, strides=2, padding='same', name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        img = Input(shape=image_dims)

        rms_opt = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-6)

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)

        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)
        # Model to train D to discriminate real images.
        self.discriminator.compile(optimizer=rms_opt, loss='binary_crossentropy')

        # Adversarial model to train Generator/R to minimize reconstruction loss and trick D to see
        r_network = self.generator(img)
        self.discriminator.trainable = False
        d_network = self.discriminator(r_network)

        # generated images as real ones.
        self.adversarial_model = Model(img, [r_network, d_network])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                       loss_weights=[self.r_alpha, 1],
                                       optimizer=rms_opt)

        print('\n\r adversarial_model')
        self.adversarial_model.summary()

    
    def train(self, num_epochs, batch_size = 128, sample_interval=500):
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        if self.dataset_name in ['mnist', 'virat']:
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.sample_dir, exist_ok=True)
        scipy.misc.imsave('./{}/train_input_samples.jpg'.format(self.sample_dir), montage(sample_inputs[:,:,:,0]))

        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []

        # Load traning data, add random noise.
        if self.dataset_name in ['mnist', 'virat']:
            sample_w_noise = get_noisy_data(self.data)
            neg_sample_w_noise = get_noisy_data(self.neg_data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(num_epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch, num_epochs))
            if self.dataset_name in ['mnist', 'virat']:
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size
             
            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                if self.dataset_name in ['mnist', 'virat']:
                    batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                    batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                    batch_clean = self.data[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)
                if self.dataset_name in ['mnist', 'virat']:
                    batch_fake_images = self.generator.predict(batch_noise_images)
                    # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                    d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                    d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, zeros)

                    # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                    self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                    g_loss = self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])    
                    plot_epochs.append(epoch+idx/batch_idxs)
                    plot_g_recon_losses.append(g_loss[1])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{5:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    if self.dataset_name in ['mnist', 'virat']:
                        samples = self.generator.predict(sample_inputs)
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)

        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_g_recon_losses)
        plt.savefig('plot_g_recon_losses.png')

        self.adversarial_model.get_layer('R').trainable = False
        self.adversarial_model.get_layer('D').trainable = True
        plot_g_finetune_losses = []
        plot_epochs_finetune = []
        for epoch in range(num_epochs//3):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch, num_epochs))
            batch_idxs = len(self.data) // batch_size

            for idx in range(0, batch_idxs):
                pos_batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                neg_batch = self.neg_data[idx * batch_size:(idx + 1) * batch_size]
                pos_noise_batch = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                neg_noise_batch = neg_sample_w_noise[idx * batch_size:(idx + 1) * batch_size]

                pos_batch_images = np.array(pos_batch).astype(np.float32)
                neg_batch_images = np.array(neg_batch).astype(np.float32)
                pos_noise_images = np.array(pos_noise_batch).astype(np.float32)
                neg_noise_images = np.array(neg_noise_batch).astype(np.float32)

                self.adversarial_model.train_on_batch(pos_noise_images, [pos_batch_images, ones])
                g_loss = self.adversarial_model.train_on_batch(neg_noise_images, [neg_batch_images, zeros])
                plot_epochs_finetune.append(epoch + idx / batch_idxs)
                plot_g_finetune_losses.append(g_loss[1])

                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss:{3:>0.3f}'.format(
                    epoch, idx, batch_idxs, g_loss[2])
                print(msg)

        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network loss after finetune')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs_finetune, plot_g_finetune_losses)
        plt.savefig('plot_g_finetune_losses.png')

        self.save()

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset_name,
            self.output_height, self.output_width)

    def save(self, step=None):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_name = 'ALOCC_Model_{}.h5'.format(step) if step is not None else 'ALOCC_Model.h5'
        self.adversarial_model.save_weights(os.path.join(self.checkpoint_dir, model_name))

def train_mnist():
    model = ALOCC_Model(dataset_name='mnist', input_height=28,input_width=28)
    model.train(num_epochs=50, batch_size=128, sample_interval=500)

def train_virat(sid, num_epochs=20):
    model = ALOCC_Model(dataset_name='virat',
                         dataset_address='/home/zal/Devel/Vehice_Action_Classifier/output/alocc_data.npz',
                         input_height=64,input_width=64,
                         output_height=64, output_width=64,
                         attention_label=sid,
                         checkpoint_dir='./checkpoint/%02d/'%sid,
                         log_dir='./log/%02d/'%sid,
                         sample_dir='./sample/%02d/'%sid,
                         r_alpha=0.2)
    model.train(num_epochs=num_epochs, batch_size=128, sample_interval=500)

if __name__ == '__main__':
    train_virat(4, 20)
