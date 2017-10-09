# coding = UTF-8
"""
风格迁移简单的实现

使用方法：
    python main.py [content_image_path] [style_image_path]
e.g.
    python main.py images/content_image.jpg images/style_image.jpg

指定参数示例：
python main.py images/content_image.jpg images/style_image.jpg --content_weight 6e-2 --style_weight 300000 1000 15 3 --tv_weight 2e-2 --num_iter 200
"""

import tensorflow as tf
import numpy as np
from model.squeezenet import SqueezeNet
from scipy.misc import imread, imresize
import PIL.Image
import argparse

def content_loss(content_features, result_features):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - result_features: features of the current image, Tensor with shape [1, height, width, channels]
    - content_features: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - content loss
    """
    # return np.sum((content_current - content_target) ** 2)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    # return (tf.norm(content_current - content_target)) ** 2

    # return tf.square(tf.norm(content_features - result_features))

    # shapes = tf.shape(content_features)

    # F_l = tf.reshape(content_features, [shapes[1], shapes[2]*shapes[3]])
    # P_l = tf.reshape(result_features,[shapes[1], shapes[2]*shapes[3]])

    # loss = (tf.reduce_sum((content_features - result_features)**2))

    return tf.reduce_sum((content_features - result_features)**2)

def gram_matrix(features):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.

    Returns:
    - gram: Tensor of shape (C, C) giving the
      Gram matrices for the input image.
    """
    shape = tf.shape(features)
    # print("gram_matrix", shape)
    H, W, C = shape[1], shape[2], shape[3]
    # 这个地方不知道为啥是这样
    #     F = tf.reshape(features, (C, H * W))
    #     G = tf.matmul(F, tf.transpose(F))
    # 这边好像确实是这样的，想了下，reshape的操作不能改变原先的维度位置，
    # 比如（1,2,3,4）只能考虑reshape成（1, （2×3）,4 ）这种
    # 如果调换中间的顺序结果就会有些奇怪
    # 感觉执行下这个例子就知道了：
    # a = np.random.random((1, 2, 3, 4))
    # b = np.reshape(a, (6, 4))
    # c = np.reshape(a, (4, 6)) # 这个结果就很奇怪
    # 当然这个操作本身没有问题，只是后面会有矩阵的乘法
    # 得到的这个c无论是怎么矩阵相乘在结果上都说不通
    # 比如按照上面的这个做法来求G（两种方法中分别用bb和cc表示）
    # bb = np.matmul(b.T, b)
    # cc = np.matmul(c, c.T)
    # 这两种方式都能得到一个维度上正确的结果，但是后面这个cc就逻辑上说不通
    F = tf.reshape(features, (H*W, C))
    G = tf.matmul(F, F, transpose_a=True)
    # normalize 
    G /= tf.cast(H * W * C, tf.float32)
    return G

def style_loss(result_features, style_features):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - result_features: features of the current image, Tensor with shape [1, height, width, channels]
    - style_features: features of the style image, Tensor with shape [1, height, width, channels]

    Returns:
    - style_loss
    """
    A = gram_matrix(result_features)
    G = gram_matrix(style_features)
    return tf.square(tf.norm(A - G))

def tv_loss(img):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    
    Returns:
    - loss
    """
    # 这部分是为了减少每个图片相邻像素的差异，让图片更加平滑，减少噪点
    shape = tf.shape(img)
    H = shape[1]
    W = shape[2]
    diff0 = tf.slice(img, [0, 0, 1, 0], [1, H-1, W-1, 3]) - tf.slice(img, [0, 0, 0, 0], [1, H-1, W-1, 3])
    diff1 = tf.slice(img, [0, 1, 0, 0], [1, H-1, W-1, 3]) - tf.slice(img, [0, 0, 0, 0], [1, H-1, W-1, 3])
    loss = tf.square(tf.norm(diff0)) + tf.square(tf.norm(diff1))
    return loss

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_path):
    img = imread(image_path)

    # height, width = img.shape[:2]
    # # img = np.expand_dims(img, axis=0)
    # img = imresize(img, 400.0 / max(height, width))
    
    orig_shape = np.array(img.shape[:2])
    min_idx = np.argmin(orig_shape)
    scale_factor = float(192) / orig_shape[min_idx]
    new_shape = (orig_shape * scale_factor).astype(int)
    img = imresize(img, scale_factor)

    img = (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD
    # print(img.shape)
    return img

def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)

def to_image(np_array):
    if len(np_array.shape) == 4:
        np_array = np_array[0, :, :, :]
    # Ensure the pixel-values are between 0 and 255.
    np_array = np.clip(np_array, 0.0, 255.0)
    # Convert pixels to bytes.
    np_array = np_array.astype(np.uint8)
    return PIL.Image.fromarray(np_array, 'RGB')

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def style_transfer(content_image, style_image, content_weight, style_weight, tv_weight, num_iter):
    content_layer = 3
    style_layers = [1, 4, 6, 7]

    content_image = preprocess_image(content_image)
    # print(content_image.shape)
    style_image = preprocess_image(style_image)
    # model = vgg16.VGG16()

    # image = tf.placeholder('float',shape=[None,224,224,3],name='input_image')
    # model.create_feed_dict(image)
    # feats = model.layer_tensors
    # tensors = model.get_layer_tensors(range(13))
    # for i in [0, 1, 2, 5, 10]:
    #     print(feats[i].shape)
    #     print(tensors[i].shape)
    # print(model.input.shape)
    tf.reset_default_graph() # remove all existing variables in the graph 
    sess = get_session() # start a new Session
    # sess = tf.Session()
    model = SqueezeNet(save_path='model/squeezenet.ckpt', sess=sess)
    feats = model.extract_features(model.image)

    print("获取指定层应该的content_target")
    # 这里content_image[None]和np.array([content_image])的写法应该是一样的
    # print(content_image[None] == np.array([content_image]))
    all_content_features = sess.run(
        feats,
        feed_dict={
            model.image: content_image[None]
        })

    print("获取指定层的style_targets")
    all_style_features = sess.run(
        feats,
        feed_dict={
            model.image: style_image[None]
        })

    print("使用content_image生成初始化图像")
    # print(content_image.shape)
    # 生成一张随机的图像
    # img_var = tf.Variable(tf.random_uniform(
    #     dtype=tf.float32, shape=np.array([content_image]).shape, 
    #     minval=0, maxval=1), name="image")
    # 使用content_image初始化
    img_var = tf.Variable(content_image[None], name='image')

    # print(img_var.shape)
    # new_image_feats = sess.run(
    #     feats,
    #     feed_dict={model.image: [img_var]})
    new_image_feats = model.extract_features(img_var)

    c_loss = content_weight * content_loss(new_image_feats[content_layer], all_content_features[content_layer])
    # print(np.array(new_image_feats).shape)
    # print(np.array(style_layers).shape)
    # print(style_targets.shape)
    # 风格层不只有一层，计算style_loss总和
    s_loss = tf.Variable(0.0)
    for index, style_layer in zip(range(len(style_layers)), style_layers):
        s_loss = tf.add(s_loss, style_weight[index] * style_loss(new_image_feats[style_layer], all_style_features[style_layer]))

    t_loss = tv_weight * tv_loss(img_var)
    loss =  c_loss + s_loss + t_loss

    print("初始化参数")
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Create and initialize the Adam optimizer
    lr_var = tf.Variable(initial_lr, name="lr")
    # Create train_op that updates the generated image when run
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])

    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    # sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))

    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

    print("开始迭代")
    # Hardcoded handcrafted 
    for t in range(num_iter):
        # Take an optimization step to update img_var
        sess.run(train_op)
        if t < decay_lr_at:
            sess.run(clamp_image_op)
        if t == decay_lr_at:
            sess.run(tf.assign(lr_var, decayed_lr))
        if t % 100 == 0:
            print('Iteration {}'.format(t))
            img = sess.run(img_var)
            to_image(deprocess_image(img[0], rescale=True)).save("result/result_{}.bmp".format(t))
            # to_image(img[0]).show()
            # plt.imshow(to_image(img[0]))
            # plt.axis('off')
            # plt.show()

    img = sess.run(img_var)
    sess.close()
    to_image(deprocess_image(img[0], rescale=True)).save("result/result.bmp")

if __name__ == '__main__':
    # import sys
    # a = float(sys.argv[1])
    # b = float(sys.argv[2])
    # c = float(sys.argv[3])
    # d = int(sys.argv[4])

    parser = argparse.ArgumentParser(description='style transfer')
    parser.add_argument('content_image', metavar='base', type=str,
                        help='Path to the content image.')
    parser.add_argument('style_image', metavar='ref', type=str,
                        help='Path to the style image.')
    parser.add_argument('--num_iter', type=int, default=200, required=False,
                        help='Number of iterations to run.')
    parser.add_argument('--content_weight', type=float, default=6e-2, required=False,
                        help='Content weight.')
    parser.add_argument('--style_weight', type=float, nargs='+', default=[300000, 1000, 15, 3], required=False,
                        help="Style weight. It's a list of float. Usage: --style_weight 3000000 1000 15 3")
    parser.add_argument('--tv_weight', type=float, default=2e-2, required=False,
                        help='Total Variation weight.')
    args = parser.parse_args()

    # params = {
    #     'content_image': 'images/content_image.jpg',
    #     'style_image': 'images/style_image.jpg',
    #     'content_weight' : 6e-2, 
    #     'style_weight' : [300000, 1000, 15, 3],
    #     'tv_weight' : 2e-2,
    #     'num_iter': 200
    # }

    params = {
        'content_image': args.content_image,
        'style_image': args.style_image,
        'content_weight' : args.content_weight, 
        'style_weight' : args.style_weight,
        'tv_weight' : args.tv_weight,
        'num_iter': args.num_iter
    }

    style_transfer(**params)
    # print(params)