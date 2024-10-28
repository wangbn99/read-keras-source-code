# Keras源码分析(3)：数据集下载

文件：/keras/utils/data_utils.py

在examples目录下的许多例子都涉及到数据集下载，查看源码你会发现它们最终都是通过keras.utils.data_utils.get_file函数下载的，这是一个普适的数据集下载工具函数，所以有必要了解其功能，以便更好的应用。get_file函数签名如下：
```sh
get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```
fname指的是缓存到本地的文件名，origin其实就是数据集文件的下载地址，即URL，其它参数基本都是不言自明的。函数大致实现如下：

（1）依据cache_dir得到数据文件就存放的文件夹，即：datadir。正常情况下，应该是/.keras/，但如果上述目录不可存取，则是/tmp/.keras/'
```sh
if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
datadir_base = os.path.expanduser(cache_dir)
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
datadir = os.path.join(datadir_base, cache_subdir)
if not os.path.exists(datadir):
    os.makedirs(datadir)
```
（2）由datadir和fname得到下载到本地的文件名：fpath
```sh
if untar:
    untar_fpath = os.path.join(datadir, fname)
    fpath = untar_fpath + '.tar.gz'else:
    fpath = os.path.join(datadir, fname)
```

（3）如果文件存在，则不用下载,
```sh
download = False
if os.path.exists(fpath):
    ......
else:
    download = True
```
（4）否则，下载文件
```sh
if download:
    print('Downloading data from', origin)
    try:
        try:
            urlretrieve(origin, fpath, dl_progress)
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if os.path.exists(fpath):
            os.remove(fpath)
            raise
```
（5）确定是否解压，如需要用extract(untar已过时)，最后返回已下载的文件路径
```sh
if untar:
    if not os.path.exists(untar_fpath):
        _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

if extract:
    _extract_archive(fpath, datadir, archive_format)
return fpath
```