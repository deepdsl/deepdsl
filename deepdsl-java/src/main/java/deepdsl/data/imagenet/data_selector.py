import sys
import os
from optparse import OptionParser
import random
import shutil

class ImageNetDataSelector:
    def parse_args(self):
        parser = OptionParser(prog="data_selector", usage="%prog path_to_dataset target_path num_categories num_images_per_category valset_ratio testset_ratio\n\n"
                                                          + "The util to select images from the imagenet training dataset\n\n"
                                                          + "Example: python data_selector.py ~/data/ILSVRC2012_img_train ~/data/temp 5 50 0.3 0.2\n"
                                                          + "The above example select from 5 categories (50 images per catetory) from ~/data/ILSVRC2012_img_train folder and store\n"
                                                          +  "selected images to ~/data/temp folder, where 30% are stored as validation dataset and 20% are stored as test dataset")
        (opts, args) = parser.parse_args()
        if len(args) != 6:
            parser.print_help()
            sys.exit(1)
        (dataset_path, target_path, numCategories, numImagesPerCategory, valsetRatio, testsetRatio) = args
        #print "dataset_path=%s, numCategories=%s, numImagesPerCategory=%s" % (dataset_path, numCategories, numImagesPerCategory)
        self.pick_images(dataset_path, target_path, numCategories, numImagesPerCategory, float(valsetRatio), float(testsetRatio))


    def pick_images(self, dataset_path, target_path, numCategories, numImagesPerCategory, valsetRatio, testsetRatio):
        sub_path = dataset_path[dataset_path.rindex('/') + 1:]
        result_target_path = os.path.join(target_path, sub_path)
        val_path = os.path.join(target_path, 'val')
        test_path = os.path.join(target_path, 'test')
        os.mkdir(result_target_path)
        os.mkdir(val_path)
        os.mkdir(test_path)
        dirs = self.get_immediate_subdirs(dataset_path)
        random_array=random.sample(xrange(0, len(dirs)), int(numCategories))
        for index in random_array:
            files = self.get_immediate_files(os.path.join(dataset_path, dirs[index]))
            os.mkdir(os.path.join(result_target_path, dirs[index]))
            source_path = os.path.join(dataset_path, dirs[index])
            if (len(files) <= int(numImagesPerCategory)):
                train_files = files[0:int(float(1 - valsetRatio - testsetRatio) * int(numImagesPerCategory))]
                val_files = files[int(float(1 - valsetRatio - testsetRatio) * int(numImagesPerCategory)):int(float(1 - testsetRatio) * int(numImagesPerCategory))]
                test_files = files[int(float(1 - testsetRatio) * int(numImagesPerCategory)):]
                for train_file in train_files:
                    shutil.copy2(os.path.join(source_path, train_file), os.path.join(result_target_path, dirs[index]))
                for val_file in val_files:
                    shutil.copy2(os.path.join(source_path, val_file), val_path)
                for test_file in test_files:
                    shutil.copy2(os.path.join(source_path, test_file), test_path)
            else:
                random_file_array = random.sample(xrange(0, len(files)), int(numImagesPerCategory))
                train_index_array = random_file_array[0:int(float(1 - valsetRatio - testsetRatio) * int(numImagesPerCategory))]
                val_index_array = random_file_array[int(float(1 - valsetRatio - testsetRatio) * int(numImagesPerCategory)):int(float(1 - testsetRatio) * int(numImagesPerCategory))]
                test_index_array = random_file_array[int(float(1 - testsetRatio) * int(numImagesPerCategory)):]

                for file_index in train_index_array:
                    shutil.copy2(os.path.join(source_path, files[int(file_index)]), os.path.join(result_target_path, dirs[index]))
                for file_index in val_index_array:
                    shutil.copy2(os.path.join(source_path, files[int(file_index)]), val_path)
                for file_index in test_index_array:
                    shutil.copy2(os.path.join(source_path, files[int(file_index)]), test_path)

    def get_immediate_files(self, dir):
        return [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]

    def get_immediate_subdirs(self, dir):
        return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

if __name__ == "__main__":
    ImageNetDataSelector().parse_args();

