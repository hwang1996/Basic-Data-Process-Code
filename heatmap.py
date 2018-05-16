import cv2 
import mxnet as mx
import os
import numpy as np
import custom_layer

model_path = 'hdfs://hobot-bigdata/user/hao01.wang/'

# json_file = model_path + 'models-7-symbol.json'
# model_file = model_path + 'models-7-0107.params'
model_file = 'model_tiny_set_2800_casia_val_YB_batch_004-0278.params'
json_file = 'model_tiny_set_2800_casia_val_YB_batch_004-symbol.json'
ctx = mx.gpu(3)

sym = mx.sym.load(json_file)
save_dict = mx.nd.load(model_file)

arg_params = {}
aux_params = {}
loaded_params = []
for k, v in save_dict.items():
	tp, name = k.split(':', 1)
	loaded_params.append(name)
	if tp == 'arg':
		arg_params[name] = v
	if tp == 'aux':
		aux_params[name] = v

hidden_layer = sym.get_internals()['conv5_s_output']
hidden_layer = mx.sym.Activation(data=hidden_layer, act_type='relu')
mod_hidden = mx.mod.Module(symbol=hidden_layer, label_names=None, context=ctx)
mod_hidden.bind(for_training=False, data_shapes=[('data', (1,3,112,96))])
# texec = hidden_layer.bind(ctx=ctx, )
mod_hidden.set_params(arg_params, aux_params)

for filename in os.listdir(r"casia_set_test"):
	img = cv2.imread('casia_set_test/'+filename).astype(np.float32)
	org_img = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.transpose((2, 0, 1))
	img[0,:] = img[0,:]-127.5
	img = img/127.5
	img = img[np.newaxis, :]
	data = mx.nd.array(img, ctx)
	batch = mx.io.DataBatch([data], [])
	mod_hidden.forward(batch)

	fea = mod_hidden.get_outputs()[0].asnumpy()
	# import pdb; pdb.set_trace()
	fea_ave = np.sum(fea[0], axis=0)/fea.shape[1]

	cam = fea_ave
	cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
	# cam = np.uint8(cam * 255)
	# img_out = cv2.resize(cv2.cvtColor(cam, cv2.COLOR_GRAY2BGR), (96, 112))
	hmap = 1 - cam
	hmap = (hmap * 255).astype(np.uint8)
	hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
	img_out = cv2.resize(hmap, (96, 112))
	img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

	img_with_heatmap = np.float32(img_out) + np.float32(org_img)
	img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
	cv2.imwrite('heatmap/'+filename, np.uint8(255 * img_with_heatmap))




# def heatmap_to_color_im(
#     hmap, normalize=False, min_max_val=None, resize=False, resize_w_h=None):
#   """
#   Args:
#     hmap: a numpy array with shape [h, w]
#     normalize: whether to normalize the value to range [0, 1]. If `False`, 
#       make sure that `hmap` has been in range [0, 1]
#   Return:
#     hmap: with shape [3, h, w], with value in range [0, 1]"""
#   if resize:
#     hmap = cv2.resize(hmap, tuple(resize_w_h), interpolation=cv2.INTER_LINEAR)
#   # normalize to interval [0, 1]
#   if normalize:
#     if min_max_val is None:
#       min_v, max_v = np.min(hmap), np.max(hmap)
#     else:
#       min_v, max_v = min_max_val
#     hmap = (hmap - min_v) / float(max_v - min_v)
#   # The `cv2.applyColorMap(gray_im, cv2.COLORMAP_JET)` maps 0 to RED and 1
#   # to BLUE, not normal. So rectify it.
#   hmap = 1 - hmap
#   hmap = (hmap * 255).astype(np.uint8)
#   # print(hmap.shape, hmap.dtype, np.min(hmap), np.max(hmap))
#   hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
#   hmap = hmap / 255.
#   hmap = np.transpose(hmap, [2, 0, 1])
#   return hmap
