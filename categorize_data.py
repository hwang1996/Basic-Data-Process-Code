import os
import shutil

path = '/data-sdc/hao01.wang/IJBA/IJBA/align_image_11/ijb_a_11_align_split10/frame/'
file_list = os.listdir(path)
for i in range(len(file_list)):
	old_file_path = os.path.join(path, file_list[i])
	if os.path.isdir(old_file_path):
		pass
	elif not os.path.exists(old_file_path):
		pass
	else:
		label = file_list[i].split('_')[0]
		new_folder_path = os.path.join('/data-sdc/hao01.wang/IJBA/dataset/',label)
		if not os.path.exists(new_folder_path):
			os.makedirs(new_folder_path)
		try:
			shutil.move(old_file_path, new_folder_path)
		except shutil.Error:
			pass

