import sys
import os
import urllib2
import gzip


base_url = "http://yann.lecun.com/exdb/mnist/"
list_archives = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
DEFAULT_MNIST_DATASET_PATH= "dataset/mnist/"

#The below code assumes we are using 64 bit operating system
class MnistDataHandler:
    def download_util(self, url):
        file_name = url.split('/')[-1]
        u = urllib2.urlopen(url)
        f = open(DEFAULT_MNIST_DATASET_PATH + file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders('Content-Length')[0])
        block_sz = 8192 * 16

        print "Downloading: %s Bytes: %s" % (file_name, file_size)

        toolbar_width =  ((file_size / block_sz) + 1) if (file_size % block_sz) else (file_size / block_sz)
        sys.stdout.write('[%s]' % (' ' * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write('\b' * (toolbar_width + 1))  # return to start of line, after '['

        file_size_dl = 0

        for i in xrange(toolbar_width):
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            # update the bar
            sys.stdout.write('-')
            sys.stdout.flush()

        f.close()
        sys.stdout.write('\n')
        return file_name


    def ungzip_file(self, gzip_file_path):
        file = open(DEFAULT_MNIST_DATASET_PATH + (gzip_file_path.split('/')[-1]).split('.gz')[0], 'wb')
        with gzip.open(gzip_file_path, 'rb') as f:
            file.write(f.read())
        f.close()
        os.remove(gzip_file_path)
        file.close()


    def change_file_permission(self, path, isWin):
        os.chmod(path, 0755)


    def process_mnist_archives(self, base_url, list_archives):
        for archive in list_archives:
            file_name = self.download_util(base_url + archive)
            self.ungzip_file(DEFAULT_MNIST_DATASET_PATH + file_name)


    def handle(self):
        self.process_mnist_archives(base_url, list_archives)


if __name__ == "__main__":
    MnistDataHandler().handle()