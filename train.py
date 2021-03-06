"""
Retrain the YOLO model for your own dataset.
"""
import os

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

try:
    from .yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_rloss
    from .yolo3.utils import get_random_data
except:
    from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_rloss
    from yolo3.utils import get_random_data

from configparser import ConfigParser

'''
Model1 = |cos(x)-cos(y)|*10 + ArIoU
Model2 = ArIoU
Model3 = |sin(x-y)| + ArIoU
Model4 = |tan(x-y)| + ArIoU
Model5 = |tan(x-y)| + IoU
Model6 = |cos(x)-cos(y)|*10 + IoU
Model7 = |cos(x)-cos(y)| + IoU
'''


def _main():
    config = ConfigParser()
    config.read("keras_yolo3.cfg")

    annotation_path = config.get("Train", "TrainPath")
    log_dir = config.get("Train", "backup_folder")
    load_pretrained = config.getboolean("Train", "load")

    classes_path = config.get("Model", "LabelsPath")
    anchors_path = config.get("Model", "anchors")
    model_path = config.get("Train", "ModelPath")
    model_name = config.get("Train", "ModelName")
    verbose = config.getint("Train", "verbose")

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    last_epoch = config.getint("Train", "last_epoch")
    ## why multiple of 32?
    input_shape = (416, 416)  # multiple of 32, hw

    # is_tiny_version = len(anchors) == 6  # default setting
    # if is_tiny_version:
    #     model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=load_pretrained,
    #                               freeze_body=2, weights_path=model_path)
    # else:
    model = create_model(input_shape, anchors, num_classes, load_pretrained=load_pretrained,
                         freeze_body=2, weights_path=model_path)  # make sure you know what you freeze

    val_split = config.getfloat("Train", "val_ratio")
    val_print = '-val_loss{val_loss:.3f}' if val_split != 0 else ''
    save_best_only = True and config.getboolean("Train", "best_model_only") if val_split != 0 else False

    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, model_name + '_ep{epoch:05d}-loss{loss:.3f}' + val_print + '.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=save_best_only,
        period=config.getint("Train", "backup_period"))
    from os.path import join
    cb2 = [checkpoint, CSVLogger(join(config.get("Train", "log_folder"), model_name + "_e" + str(last_epoch) + ".csv"),
                                 append=True)]

    # logging = TensorBoard()

    if config.getboolean("Train", "reduce_lr"):
        cb2.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1))

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    batch_size = config.getint("Train", "batch")

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if config.getboolean("Train", "stage1"):
        lr = config.getfloat("Train", "stage1_lr")

        model.compile(optimizer=Adam(lr=lr), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=last_epoch + config.getint("Train", "stage1_epoch"),
                            initial_epoch=last_epoch,
                            callbacks=cb2,
                            verbose=verbose)
        model.save_weights(os.path.join(log_dir, model_name + '_stage1.h5'))
        last_epoch += config.getint("Train", "stage1_epoch")

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if config.getboolean("Train", "stage2"):
        # for i in range(len(model.layers)):
        #     model.layers[i].trainable = True
        # print('Unfreeze all of the layers.')

        lr = config.getfloat("Train", "stage2_lr")
        model.compile(optimizer=Adam(lr=lr),
                      loss={'yolo_rloss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Learning rate:', lr)
        # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes)
                            if val_split != 0.0 else None,
                            validation_steps=max(1, num_val // batch_size) if val_split != 0.0 else None,
                            epochs=last_epoch + config.getint("Train", "stage2_epoch"),
                            initial_epoch=last_epoch,
                            callbacks=cb2, verbose=verbose)
        model.save_weights(os.path.join(log_dir, model_name + '_final.h5'))

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 6)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # if freeze_body in [1, 2]:
        #     # Freeze darknet53 body or freeze all but 3 output layers.
        #     num = (185, len(model_body.layers) - 3)[freeze_body - 1]
        #     for i in range(num): model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    model_rloss = Lambda(yolo_rloss, output_shape=(1,), name='yolo_rloss',
                         arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], [model_loss, model_rloss])

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 6)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # if freeze_body in [1, 2]:
        #     # Freeze the darknet body or freeze all but 2 output layers.
        #     num = (20, len(model_body.layers) - 2)[freeze_body - 1]
        #     for i in range(num): model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
