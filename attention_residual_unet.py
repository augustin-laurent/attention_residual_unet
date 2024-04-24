import tensorflow as tf

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = tf.nn.conv2d(x, filters=size, kernel_size=(filter_size, filter_size), padding="SAME")
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    conv = tf.nn.relu(conv)

    conv = tf.nn.conv2d(conv, filters=size, kernel_size=(filter_size, filter_size), padding="SAME")
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    conv = tf.nn.relu(conv)

    if dropout > 0:
        conv = tf.nn.dropout(conv, rate=dropout)

    return conv

def repeat_elem(tensor, rep):
    return tf.repeat(tensor, rep, axis=3)

def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = tf.nn.conv2d(x, filters=size, kernel_size=(filter_size, filter_size), padding="SAME")
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    conv = tf.nn.relu(conv)

    conv = tf.nn.conv2d(conv, filters=size, kernel_size=(filter_size, filter_size), padding="SAME")
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    if dropout > 0:
        conv = tf.nn.dropout(conv, rate=dropout)

    shortcut = tf.nn.conv2d(x, filters=size, kernel_size=(1, 1), padding="SAME")
    if batch_norm:
        shortcut = tf.nn.batch_normalization(shortcut, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)

    res_path = tf.add(shortcut, conv)
    res_path = tf.nn.relu(res_path)

    return res_path

def gating_signal(input, out_size, batch_norm=False):
    x = tf.nn.conv2d(input, filters=out_size, kernel_size=(1, 1), padding="SAME")
    if batch_norm:
        x = tf.nn.batch_normalization(x, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    x = tf.nn.relu(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = tf.shape(x)
    shape_g = tf.shape(gating)

    theta_x = tf.nn.conv2d(x, filters=inter_shape, kernel_size=(2, 2), strides=(2, 2), padding="SAME")

    phi_g = tf.nn.conv2d(gating, filters=inter_shape, kernel_size=(1, 1), padding="SAME")
    upsample_g = tf.nn.conv2d_transpose(phi_g, filters=inter_shape, kernel_size=(3, 3), strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]),padding="SAME")

    concat_xg = tf.add(upsample_g, theta_x)
    act_xg = tf.nn.relu(concat_xg)
    psi = tf.nn.conv2d(act_xg, filters=1, kernel_size=(1, 1), padding="SAME")
    sigmoid_xg = tf.nn.sigmoid(psi)

    upsample_psi = tf.image.resize(sigmoid_xg, size=(shape_x[1], shape_x[2]))
    upsample_psi = tf.tile(upsample_psi, [1, 1, 1, shape_x[3]])

    y = tf.multiply(upsample_psi, x)

    result = tf.nn.conv2d(y, filters=shape_x[3], kernel_size=(1, 1), padding="SAME")
    result_bn = tf.nn.batch_normalization(result, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    return result_bn

def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True, FILTER_NUM = 64, FILTER_SIZE = 3, UP_SAMP_SIZE = 2):
    inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))
    axis = 3
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = tf.nn.max_pool(conv_128, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = tf.nn.max_pool(conv_64, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = tf.nn.max_pool(conv_32, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = tf.nn.max_pool(conv_16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = tf.image.resize(conv_8, [UP_SAMP_SIZE, UP_SAMP_SIZE])
    up_16 = tf.concat([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = tf.image.resize(up_conv_16, [UP_SAMP_SIZE, UP_SAMP_SIZE])
    up_32 = tf.concat([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = tf.image.resize(up_conv_32, [UP_SAMP_SIZE, UP_SAMP_SIZE])
    up_64 = tf.concat([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = tf.image.resize(up_conv_64, [UP_SAMP_SIZE, UP_SAMP_SIZE])
    up_128 = tf.concat([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    conv_final = tf.nn.conv2d(up_conv_128, filters=NUM_CLASSES, kernel_size=(1,1), padding='SAME')
    if batch_norm:
        conv_final = tf.nn.batch_normalization(conv_final, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-5)
    conv_final = tf.nn.sigmoid(conv_final)

    return conv_final
