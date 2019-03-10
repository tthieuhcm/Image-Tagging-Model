from scipy.io import loadmat
import os

if __name__ == "__main__":
    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    mat_data = loadmat(data_folder+'/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.mat')
    with open(data_folder+'/ImageSets/CLS-LOC/val.txt', 'r') as val_images_file, open(data_folder+'/devkit/data/map_clsloc.txt', 'r') as mapping_file:
        mat_data = mat_data.get('rec')[0]
        val_images_file_lines = val_images_file.readlines()
        mapping_file_lines = mapping_file.readlines()

        for i, image in enumerate(mat_data):
            image_info = val_images_file_lines[i].split()
            image_name = image_info[0]
            image_class = mat_data[i][0][0][0][0][0][0]

            class_info = mapping_file_lines[int(image_class)-1].split()
            class_folder_name = class_info[0]
            if not os.path.exists(data_folder+'/Data/CLS-LOC/val/'+class_folder_name):
                os.mkdir(data_folder+'/Data/CLS-LOC/val/'+class_folder_name)
            try:
                os.rename(data_folder+'/Data/CLS-LOC/val/'+image_name+'.JPEG', data_folder+'/Data/CLS-LOC/val/'+class_folder_name+'/'+image_name+'.JPEG')
            except:
                continue
            if i%1000==0:
                print('Processing ... {}/{}'.format(i, len(mat_data)))
