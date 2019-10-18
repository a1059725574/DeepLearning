import tensorflow as tf

batch_size = 1
img_height = 9
img_width = 9
img_channel = 32


kernel_size = 3
stride = 1
padding = "SAME"

# x = tf.placeholder(tf.float32, [1, 9, 9, 32])

mask = tf.ones(shape = [1, 9, 9, 2])
weight_maskup = tf.ones(shape = [kernel_size, kernel_size, 2, 1])
slide_winsize = kernel_size * kernel_size * 2
slide_winsize = 18

def PartialConv(x, mask, channels, kernel_size, stride  = 1, padding = "SAME", use_bias = True, use_sn = False, multi_channel = False, return_mask = True, scope = "partial_conv")

    ch_in = tf.get_shape()[-1]
    ch_out = channels

    with tf.variable_scope(scope):

        with tf.variable_scope("mask"):

            if multi_channel:
                weight_maskupdater = tf.ones(shape = [kernel_size, kernel_size, ch_in, ch_out])
                slide_winsize = kernel_size * kernel_size * ch_in 
            else:
                weight_maskupdater = tf.ones(shape = [kernel_size, kernel_size, 1, 1])
                slide_winsize = kernel_size * kernel_size
        
            update_mask = tf.nn.conv2d(mask, weight_maskup, [1, stride, stride, 1], padding = padding)
            mask_ratio = slide_winsize/(update_mask)
            update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
            mask_ratio = tf.multiply(mask_ratio, update_mask)

        with tf.variable_scope("x"):
            w = tf.get_variable("kernel", [kernl_size, kernel_size, ch_in, ch_out], initializer = tf.contrib.layers.xavier_initializer())
            x = tf.nn.conv2d(x*mask, w, [1, stride, stride, 1], padding = padding)
            if use_bias:
                bias = tf.get_variable("bias", [ch_out], initializer = tf.constant_initializer(0.0))
                output = (x - bias)*mask_ratio + bias
                output = x*update_mask
            else:
                output = x*mask_ratio

            if return_mask: 
                update_mask = tf.reduce_prod(update_mask, axis = -1, keep_dims = True)
                return output, update_mask
            else:
                return output

def GramMatrix(x, length, depth):
    
    # batch_size, height, width, channels = tf.get_shape().as_list()
    # feature()
    x = tf.reshape(x, (length, depth))
    x = tf.matmul(tf.transpose(x), x)
    return x

"""
with tf.Session() as sess:
    # sess.run(tf.gloabl_variables_initializer())
    # print(sess.run(mask_ratio))
    print(sess.run(update_mask))
"""

