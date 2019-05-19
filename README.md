# Youtube Video Understanding 

When having issues with GitHub ipynb notebook viewing, please visit this [link](https://nbviewer.jupyter.org/github/lordsoffallen/youtube-video-understanding/blob/master/Youtube%20Data%20Analytics.ipynb) for Youtube Analytics, this [link](https://nbviewer.jupyter.org/github/lordsoffallen/youtube-video-understanding/blob/master/Simple%20Models.ipynb) for simple models.

## Finding steps_per_epoch parameter value
```
# Get the list of file names
train = glob.glob('video/train*.tfrecord')
val = glob.glob('video/val*.tfrecord')

file_lens = [len([1 for example in tf.python_io.tf_record_iterator(train[0])]), 
             len([1 for example in tf.python_io.tf_record_iterator(train[1])])]
print('Steps_per_epoch should be length of data / batch size.')
print('In this case, we can choose : {} / {} (default batch_size) =~ {}'\
      .format(min(file_lens), 32, round(min(file_lens)/32)))
```