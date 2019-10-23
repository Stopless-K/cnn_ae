import tensorflow as tf
import os
from src.Autoencoder import Autoencoder, ConvAutoencoder
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_path', './train/', 'training_data')
tf.app.flags.DEFINE_string('checkpoint','./', 'chekpoint path to restore model')
tf.app.flags.DEFINE_integer('input_dim', 60, 'data dimension')
tf.app.flags.DEFINE_integer('input_channel', 1, 'channel dimension')
tf.app.flags.DEFINE_integer('hidden_dim', 10, 'middle layer units')



def read(filelist):
    file_queue = tf.train.string_input_producer(filelist, num_epochs=1, shuffle=False)
    reader = tf.TextLineReader()
    key,value = reader.read(file_queue)
    sample = tf.decode_csv(value, record_defaults=[[]]*FLAGS.input_dim)
    return key,sample


def main(arg):
    files = os.listdir(FLAGS.test_path)
    filelist = [os.path.join(FLAGS.test_path, _file) for _file in files]
    key, sample = read(filelist)
    
    batch = tf.expand_dims(sample, 0)
    ae = ConvAutoencoder(batch, [FLAGS.input_channel, FLAGS.hidden_dim],kernel_size = 3, 
                     transfer_function= tf.nn.relu, 
                    )
    ae.val()
    saver = tf.train.Saver()
    f = open('predict.csv', 'w')
    writer = csv.writer(f)

    with tf.Session() as sess:
        
        sess.run(tf.local_variables_initializer())
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint)
        assert ckpt_state
        saver.restore(sess, ckpt_state.model_checkpoint_path)
            
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)
        #init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        
        try:
            while True:
                s_loss = sess.run(ae.cost,  feed_dict={ae.keep_prob: ae.in_keep_prob})
                writer.writerow([1 if s_loss<=0.01 else 0])
        except tf.errors.OutOfRangeError:
            pass
        
        f.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

