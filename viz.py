"""Visualization modules"""

import cv2
import mxnet as mx
import numpy as np
import os

def _fill_buf(buf, i, img, shape):
    import pdb; pdb.set_trace()
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0] # n*h/h

    sx = (i%m)*shape[0]
    sy = (i//m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img


def layout(X, flip=False):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1)) ##batch x h x w x c
    X = np.clip(X * 255.0, 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    import pdb; pdb.set_trace()
    print n
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        img = np.flipud(img) if flip else img
        _fill_buf(buff, i, img, X.shape[1:3])
    if buff.shape[-1] == 1:
        return buff.reshape(buff.shape[0], buff.shape[1])
    if X.shape[-1] != 1:
        buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    return buff


def imshow(title, X, waitsec=1, flip=False):
    """Show images in X and wait for wait sec.
    """
    buff = layout(X, flip=flip)
    cv2.imshow(title, buff)
    cv2.waitKey(waitsec)

def imsave(path, X, flip=False):
    """save images
    """
    import pdb; pdb.set_trace()
    buff = layout(X, flip=flip)
    cv2.imwrite(path, buff)
    print 'images are saving to disk %s'%path

# data_iter = mx.io.ImageRecordIter(
#     path_imgrec='hangzhougongan_train.rec',
#     data_shape=(3, 112, 96), # output data shape. An 227x227 region will be cropped from the original image.
#     batch_size=10, # number of samples per batch
#     )


# data_iter.reset()
# batch = data_iter.next()
# img = batch.data[0].asnumpy()
# # import pdb; pdb.set_trace()
# buff = np.zeros((112, 96*10, 3))
# for i in range(10):
#     img_cv = cv2.cvtColor(img[i].transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
#     buff[:, i*96:(i+1)*96, :] = img_cv
# buff[0:10, 0:30, :] = 255
# buff_score = cv2.putText(buff, '0.00', (0, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
# cv2.imwrite('viz_img.png', buff)

score_img = {}
for filename in os.listdir(r"tiny_1M"):
    img = cv2.imread('tiny_1M/'+filename).astype(np.uint8)
    score = float(filename[0:8])
    score_img[score] = img   # img (112, 96, 3)

# import pdb; pdb.set_trace()
score_sorted = sorted(score_img.items(), key = lambda x:x[0], reverse = False)
buff = np.zeros((img.shape[0], img.shape[1]*10, img.shape[2]))
for i in range(10):
    buff[:, i*img.shape[1]:(i+1)*img.shape[1], :] = score_sorted[15*i+1][1]
for i in range(10):
    buff[0:10, img.shape[1]*i:img.shape[1]*i+30, :] = 255
    buff_score = cv2.putText(buff, str(score_sorted[15*i+1][0])[0:5], (img.shape[1]*i, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
cv2.imwrite('viz_img_1.png', buff)

buff = np.zeros((img.shape[0], img.shape[1]*10, img.shape[2]))
for i in range(10):
    buff[:, i*img.shape[1]:(i+1)*img.shape[1], :] = score_sorted[15*(i+10)+1][1]
for i in range(10):
    buff[0:10, img.shape[1]*i:img.shape[1]*i+30, :] = 255
    buff_score = cv2.putText(buff, str(score_sorted[15*(i+10)+1][0])[0:5], (img.shape[1]*i, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
cv2.imwrite('viz_img_2.png', buff)
