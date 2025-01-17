import tensorflow as tf
from FaceAging import FaceAging
from os import environ
import argparse

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable notification, warning, error
environ['CUDA_VISIBLE_DEVICES'] = '0'



def str2bool(v):
    ## a function transform string to bool.
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  ## lower() method provides lower case of a string.
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


## parse arguments
parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='test', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--use_sn', type=str2bool, default=False, help='whether use spectral normalization on conv2d')
parser.add_argument('--encoder_use_sn', type=str2bool, default=False, help='whether use spectral normalization on conv2d')
parser.add_argument('--use_hinge_loss', type=str2bool, default=False, help='whether use hinge-loss on G-D pair losses')
parser.add_argument('--weigts', type=float, nargs='+', default=None, help='whether use l1loss on EG loss')
parser.add_argument('--egloss', type=str, default='l1', help='whether use l1loss on EG loss')


FLAGS = parser.parse_args()


def main(_):

    ## print settings
    from pprint import pprint
    pprint(FLAGS)  ## pprint is a more elegant version of print, which can print out long, complex structures in separate lines.

    ## add tensorflow configs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.90 

    ## create session
    with tf.Session(config=config) as session:
        model = FaceAging(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset,  # name of the dataset in the folder ./data
            use_hinge_loss=FLAGS.use_hinge_loss,
            use_sn=FLAGS.use_sn,
            encoder_use_sn=FLAGS.encoder_use_sn,
            eg_loss_type=FLAGS.egloss
        )
        if FLAGS.is_train:
            print('\n\tTraining Mode')
            if not FLAGS.use_trained_model:  ## pre-train, go for 10 epochs
                print('\n\tPre-train the network')
                model.train(
                    num_epochs=10,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model,
                    weigts=(0, 0, 0)  ## the weights of adversarial loss and TV loss 
                )
                print('\n\tPre-train is done! The training will start.')
            if FLAGS.weigts is not None:
                weigts = tuple(FLAGS.weigts)
            else:
                weigts=(0.0001, 0, 0)
            print(weigts)
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=FLAGS.use_init_model,
                weigts=weigts  ## the weights of adversarial loss and TV loss 
            )
        else:
            print('\n\tTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*'
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()

