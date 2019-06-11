# Youtube Video Understanding 

When having issues with GitHub ipynb notebook viewing, please visit this [link](https://nbviewer.jupyter.org/github/lordsoffallen/youtube-video-understanding/blob/master/Youtube%20Data%20Analytics.ipynb) for Youtube Analytics, this [link](https://nbviewer.jupyter.org/github/lordsoffallen/youtube-video-understanding/blob/master/Model%20Analysis%20v1.ipynb) for Model analysis v1 and this [link](https://nbviewer.jupyter.org/github/lordsoffallen/youtube-video-understanding/blob/master/Model%20Analysis%20v2.ipynb) for v2.

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

# Paper Links

[Google Original Paper](https://arxiv.org/pdf/1609.08675.pdf)
[1](http://cs231n.stanford.edu/reports/2017/pdfs/702.pdf)  
[2](http://cs231n.stanford.edu/reports/2017/pdfs/711.pdf)  
[3](http://cs231n.stanford.edu/reports/2017/pdfs/705.pdf)  
