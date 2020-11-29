import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        '''
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        '''
        self.writer.add_scalar(tag= tag, scalar_value=value, global_step=step)
        #with self.writer.as_default():
        #     tf.summary.scalar(tag, value, step=step)
        #     self.writer.flush()
             