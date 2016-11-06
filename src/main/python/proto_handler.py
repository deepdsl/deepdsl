import sys
import os
import urllib2
import zipfile
import subprocess


osx_sys_str = "osx-x86_64"
linux_sys_str = "linux-x86_64"
windows_sys_str = "win32"
base_url = "https://github.com/google/protobuf/releases/download/v3.0.0/"
osx_url = base_url + "protoc-3.0.0-" + osx_sys_str + ".zip"
linux_url = base_url + "protoc-3.0.0-" + linux_sys_str + ".zip"
windows_url = base_url + "protoc-3.0.0-" + windows_sys_str + ".zip"


#The below code assumes we are using 64 bit operating system
class ProtoHandler:
    def download_util(self, url):
        file_name = url.split('/')[-1]
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders('Content-Length')[0])
        block_sz = 8192

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


    def unzip_file(self, path):
        unzip_folder = path.split('.zip')[0]
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(unzip_folder)
        zip_ref.close()
        return unzip_folder + '/bin/protoc'


    def change_file_permission(self, path, isWin):
        os.chmod(path, 0755)


    def process_proto(self, url, sys_str, isWin=False):
        file_name = self.download_util(url)
        protoc = self.unzip_file(file_name)
        if not isWin:
            self.change_file_permission(protoc, isWin)
        subprocess.check_call([protoc, '-I=.', '--java_out=src/main/java', 'dataset/imagenet/imagenet-' + sys_str + '.proto'])


    def handle(self):
        os_name = sys.platform
        print os_name
        if os_name.startswith('darwin'):
            # OS X
            self.process_proto(osx_url, osx_sys_str)
        elif os_name.startswith('linux'):
            # Linux
            self.process_proto(linux_url, linux_sys_str)
        elif os_name.startswith('win'):
            # Windows
            self.process_proto(windows_url, windows_sys_str, True)
        else:
            print "The operating system %s is not supported" % os_name


if __name__ == "__main__":
    ProtoHandler().handle();