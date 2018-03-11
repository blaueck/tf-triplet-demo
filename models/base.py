import tensorflow as tf


class BaseNet:
    '''Base class for Net. It uses template to manager
    variables.
    '''

    def __init__(self, name='BaseNet'):
        self.name = name

        # wrapping network into a template. It is useful
        # for paramters sharing.
        self.template = tf.make_template(name, self.call, True)

    def __getattr__(self, name):
        # make this class able to access template's method 
        return self.template.__getattribute__(name)

    def __call__(self, *args, **kwargs):
        # call template
        return self.template(*args, **kwargs)

    def call(self, images, is_training):
        raise NotImplementedError('this method is not implemented.')
