import tensorflow as tf


def matrix_multiply():
    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[1], [2]])
    product = tf.matmul(matrix1, matrix2)
    with tf.Session() as sess:
        res = sess.run([product])
        print(res)

