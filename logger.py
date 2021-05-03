"""Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514."""
import os

import tensorflow as tf


class Logger(object):
    """Tensorboard logger class."""
    def __init__(self, log_dir: str, name: str = None):
        """Create a summary writer logging to log_dir.

        Args:
            log_dir: Directory where tensorboard logs are to be saved.
            name: Name of the sub-folder.

        """
        # self.acc_var = tf.Variable(0, dtype=tf.float32)
        # self.acc_tag = "acc"
        # self.acc_summ = tf.compat.v1.summary.scalar(self.acc_tag, self.acc_var)
        self.sess = tf.compat.v1.Session()


        if name is None:
            name = "temp"
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            # self.writer = tf.summary.create_file_writer(os.path.join(
            self.writer = tf.compat.v1.summary.FileWriter(os.path.join(log_dir, name),filename_suffix=name)
        else:
            self.writer = tf.compat.v1.summary.FileWriter(log_dir,filename_suffix=name)

    def scalar_summary(self, tag: str, value: float, step: int):
        """Log a scalar variable.

        Args:
            tag: Tag for the variable being logged.
            value: Value of the variable.
            step: Iteration step number.

        """
        # with self.writer.as_default():
        #     tf.summary.scalar(name=tag, data=value, step=step)
        # self.sess.run( self.acc_var.assign(value) )
        # self.sess.run( self.acc_tag.assign(tag) 
        acc_summ = tf.compat.v1.summary.scalar( tag, value )

        self.writer.add_summary( self.sess.run( acc_summ ), step)
