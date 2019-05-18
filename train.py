import glob
from src import fine_tune_model

if __name__ == '__main__':
    train = glob.glob('video/train*.tfrecord')
    val = glob.glob('video/val*.tfrecord')
    fine_tune_model(train_records=train, val_records=val, sel='rgb',
                    repeats=None, choice='dnn', units=[[128, 256, 512, 1024]])
