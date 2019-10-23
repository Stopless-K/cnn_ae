import tensorflow as tf
import os
from src.Autoencoder import Autoencoder, ConvAutoencoder


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', './train/', 'training_data')
tf.app.flags.DEFINE_string('logdir', './experiments/first', 'log directory to write events')

tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('input_dim', 60, 'data dimension')
tf.app.flags.DEFINE_integer('input_channel', 1, 'channel dimension')
tf.app.flags.DEFINE_integer('hidden_dim', 10, 'middle layer units')
tf.app.flags.DEFINE_integer('num_iters', 1000, 'number of iterations')
tf.app.flags.DEFINE_integer('print_interval', 10, 'display frequency')
tf.app.flags.DEFINE_integer('save_interval', 1000, 'save model frequency')

tf.app.flags.DEFINE_float('corruption_level', 0.3, 'denoising level')
tf.app.flags.DEFINE_float('lr', 1e-3, 'initial learning rate')


def read(filelist):
    file_queue = tf.train.string_input_producer(filelist)
    reader = tf.TextLineReader()
    key,value = reader.read(file_queue)
    sample = tf.decode_csv(value, record_defaults=[[]]*FLAGS.input_dim)
    sample_batch = tf.train.batch([sample], batch_size=FLAGS.batch_size, num_threads=1, capacity=10)
    return sample_batch


def main(arg):
    files = os.listdir(FLAGS.train_path)
    filelist = [os.path.join(FLAGS.train_path, _file) for _file in files]
    batch = read(filelist)

    ae = ConvAutoencoder(batch, [FLAGS.input_channel, FLAGS.hidden_dim],kernel_size = 3, 
                     transfer_function= tf.nn.relu, 
                     optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr),
                     ae_para = [FLAGS.corruption_level, 0]# sparse regularization set to zero
                    )
    
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(FLAGS.num_iters):
            train_step  = ae.partial_fit()
            loss, _ = sess.run(train_step, feed_dict={ae.keep_prob: ae.in_keep_prob})

            if(step % FLAGS.print_interval==0 and step>0):
                print("Step: {}, Loss: {}".format(step, loss) )
            
            if((step+1) % FLAGS.save_interval==0 ):
                saver.save(sess, os.path.join(FLAGS.logdir, 'model_{}.ckpt'.format(step+1)))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

