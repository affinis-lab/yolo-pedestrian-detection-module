import numpy as np
from keras.callbacks import TensorBoard, Callback
import os

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



class MAP_evaluation(Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, annot_dir, model,
                 yolo,
                 generator,
                 iou_threshold=0.5,
                 score_threshold=0.3,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None ):

        self.yolo = yolo
        self.annot_dir = annot_dir
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.period = period
        self.save_best = save_best
        self.save_name = save_name
        self.tensorboard = tensorboard

        self.bestMap = 0

        self.model = model

        if not isinstance(self.tensorboard, TensorBoard) and self.tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):
        print(epoch)
        # % self.period == 0 and self.period != 0:
        mAP, average_precisions = self.evaluate_mAP()
        print('\n')
        for label, average_precision in average_precisions.items():
            print(self.yolo.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(mAP))

        if self.save_best and self.save_name is not None and mAP > self.bestMap:
            print(
                "mAP improved from {} to {}, saving model to {}.".format(self.bestMap, mAP, self.save_name))
            self.bestMap = mAP
            print(self.save_name)
            self.model.save(self.save_name)
            self.model.save_weights('checkpoints\\best-mAP.h5')
        else:
            print("mAP did not improve from {}.".format(self.bestMap))

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = mAP
            summary_value.tag = "val_mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

    def evaluate_mAP(self):
        average_precisions = self._calc_avg_precisions()
        mAP = sum(average_precisions.values()) / len(average_precisions)

        return mAP, average_precisions

    def _calc_avg_precisions(self):
        # gather all detections and annotations
        all_detections = [[None for i in range(self.generator.num_classes())] for j in
                          range(self.generator.size())]
        all_annotations = [[None for i in range(self.generator.num_classes())] for j in
                           range(self.generator.size())]

        for i in range(self.generator.size()):
            raw_image = self.generator.load_image(self.generator.dataset[i])
            raw_height, raw_width, _ = raw_image.shape
            # make the boxes and the labels
            pred_boxes = self.yolo.predict(os.path.join(self.annot_dir, 'images', self.generator.dataset[i]['image_path']))

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(self.generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = self.generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(self.generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(self.generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self.generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= self.iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions