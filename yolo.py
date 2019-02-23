from PIL import Image
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
from BatchGenerator import BatchGenerator
import json
import cv2
from utils import compute_overlap, compute_ap, decode_netout

# def draw_rec(image, positions):
#     im = np.array(Image.open(image), dtype=np.uint8)
#     im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
#     #im  = im * 256
#     dy = int(np.floor(IMAGE_WIDTH / HORIZONTAL_GRIDS))
#     dx = int(np.floor(IMAGE_HEIGHT / VERTICAL_GRIDS))
#
#     # Custom (rgb) grid color
#     grid_color = [0, 0, 255]
#
#     # Modify the image to include the grid
#     im[:, ::dy, :] = grid_color
#     im[::dx, :, :] = grid_color
#     # Create figure and axes
#     fig, ax = plt.subplots()
#
#     # Display the image
#     ax.imshow(im)
#
#
#     # Create a Rectangle patch
#     for pos in [positions]:
#
#         rect = patches.Rectangle((pos[0], pos[1]), pos[2], pos[3], linewidth=3, edgecolor='r', facecolor='none')
#
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#     plt.grid(b=True, which='major', axis='both')
#     plt.show()

class Yolo:

    def __init__(self, config):

        self.config = config
        self.train_data_dir = self.config["train"]["train_image_folder"]
        self.validation_data_dir = self.config["valid"]["valid_image_folder"]

        self.NUM_CLASSES = 1
        self.epochs = self.config["train"]["epochs"]
        self.class_wt = np.ones(self.NUM_CLASSES, dtype='float32')
        self.true_boxes = 0
        self.grid_w = self.config["model"]["horizontal_grids"]
        self.grid_h = self.config["model"]["vertical_grids"]
        self.batch_size = self.config["train"]["batch_size"]
        self.nb_box = self.config["model"]["nb_box"]
        self.anchors = self.config["model"]["anchors"]
        self.coord_scale = self.config["train"]["coord_scale"]
        self.object_scale = self.config["train"]["object_scale"]
        self.no_object_scale = self.config["train"]["no_object_scale"]
        self.class_scale = self.config["train"]["class_scale"]
        self.warmup_batches = self.config["train"]["warmup"]
        self.debug = self.config["train"]["debug"]
        self.image_height = self.config["model"]["image_size"]
        self.image_width = self.config["model"]["image_size"]
        self.box = self.config["model"]["box"]
        self.number_of_grids = self.config["model"]["horizontal_grids"]

        self.model = self.get_full_yolo_model()

    def get_yolo_model(self):
        input_image = Input(shape=(self.image_width, self.image_height, 3))
        boxes = Input(shape=(1, 1, 1, self.box, 4))
        self.true_boxes = boxes

        # Layer 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 4
        for i in range(0, 3):
            x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2),
                       use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i + 2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 5
        x = Conv2D(32 * 8, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(5),
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='qweconv_' + str(6), use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # Layer 7
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='qweconv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='qweconv_' + str(8), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(8))(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(self.box * (4 + 1 + self.NUM_CLASSES), (1, 1), strides=(1, 1), padding='same', name='not', activation='linear',
                   kernel_initializer='lecun_normal')(x)
        output = Reshape((self.number_of_grids, self.number_of_grids, self.box, 4 + 1 + self.NUM_CLASSES))(x)

        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        return Model([input_image, self.true_boxes], output)

    def get_full_yolo_model(self):
        input_image = Input(shape=(self.image_width, self.image_height, 3))
        boxes = Input(shape=(1, 1, 1, self.nb_box, 4))
        self.true_boxes = boxes
        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)

        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        # # Layer 1
        # x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_0', use_bias=False)(input_image)
        # x = BatchNormalization(name='norm_1')(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
            skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(self.box * (4 + 1 + self.NUM_CLASSES), (1, 1), strides=(1, 1), padding='same',
                   name='not', kernel_initializer='lecun_normal')(x)
        output = Reshape((self.number_of_grids, self.number_of_grids, self.box, 4 + 1 + self.NUM_CLASSES))(x)

        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        return Model([input_image, self.true_boxes], output)

    # def custom_loss(self, y_true, y_pred):
    #     mask_shape = tf.shape(y_true)[:4]
    #
    #     cell_x = tf.to_float(
    #         tf.reshape(tf.tile(tf.range(HORIZONTAL_GRIDS), [VERTICAL_GRIDS]),
    #                    (1, VERTICAL_GRIDS, HORIZONTAL_GRIDS, 1, 1)))
    #     cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    #
    #     cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, BOX, 1])
    #
    #     coord_mask = tf.zeros(mask_shape)
    #     conf_mask = tf.zeros(mask_shape)
    #     class_mask = tf.zeros(mask_shape)
    #
    #     seen = tf.Variable(0.)
    #     total_loss = tf.Variable(0.)
    #     total_recall = tf.Variable(0.)
    #     total_boxes = tf.Variable(VERTICAL_GRIDS * HORIZONTAL_GRIDS * BOX * BATCH_SIZE)
    #
    #     """
    #     Adjust prediction
    #     """
    #     ### adjust x and y
    #     pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    #
    #     ### adjust w and h tf.exp(
    #     pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    #
    #     ### adjust confidence
    #     pred_box_conf = tf.sigmoid(y_pred[..., 4])
    #
    #     ### adjust class probabilities
    #     pred_box_class = y_pred[..., 5:]
    #
    #     """
    #     Adjust ground truth
    #     """
    #     ### adjust x and y
    #     true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
    #
    #     ### adjust w and h
    #     true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically
    #
    #     ### adjust confidence
    #     true_wh_half = true_box_wh / 2.
    #     true_mins = true_box_xy - true_wh_half
    #     true_maxes = true_box_xy + true_wh_half
    #
    #     pred_wh_half = pred_box_wh / 2.
    #     pred_mins = pred_box_xy - pred_wh_half
    #     pred_maxes = pred_box_xy + pred_wh_half
    #
    #     intersect_mins = tf.maximum(pred_mins, true_mins)
    #     intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    #     intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    #     intersect_areas2 = intersect_wh[..., 0] * intersect_wh[..., 1]
    #
    #     true_areas2 = true_box_wh[..., 0] * true_box_wh[..., 1]
    #     pred_areas2 = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    #
    #     union_areas2 = pred_areas2 + true_areas2 - intersect_areas2
    #     iou_scores2 = tf.truediv(intersect_areas2, union_areas2)
    #
    #     true_box_conf = iou_scores2 * y_true[..., 4]
    #
    #     ### adjust class probabilities
    #     true_box_class = tf.argmax(y_true[..., 5:], -1)
    #
    #     """
    #     Determine the masks
    #     """
    #     ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    #     coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    #
    #     ### confidence mask: penelize predictors + penalize boxes with low IOU
    #     # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    #     true_xy = self.true_boxes[..., 0:2]
    #     true_wh = self.true_boxes[..., 2:4]
    #
    #     true_wh_half = true_wh / 2.
    #     true_mins = true_xy - true_wh_half
    #     true_maxes = true_xy + true_wh_half
    #
    #     pred_xy = tf.expand_dims(pred_box_xy, 4)
    #     pred_wh = tf.expand_dims(pred_box_wh, 4)
    #
    #     pred_wh_half = pred_wh / 2.
    #     pred_mins = pred_xy - pred_wh_half
    #     pred_maxes = pred_xy + pred_wh_half
    #
    #     intersect_mins = tf.maximum(pred_mins, true_mins)
    #     intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    #     intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    #     intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    #
    #     true_areas = true_wh[..., 0] * true_wh[..., 1]
    #     pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    #
    #     union_areas = pred_areas + true_areas - intersect_areas
    #     iou_scores = tf.truediv(intersect_areas, union_areas)
    #
    #     best_ious = tf.reduce_max(iou_scores, axis=4)
    #     # conf_mask = conf_mask + tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * self.no_object_scale
    #
    #     # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    #     # conf_mask = conf_mask + y_true[..., 4] * self.object_scale
    #
    #     conf_mask_neg = tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    #     conf_mask_pos = y_true[..., 4] * OBJECT_SCALE
    #
    #     ### class mask: simply the position of the ground truth boxes (the predictors)
    #     class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * CLASS_SCALE
    #
    #     """
    #     Warm-up training
    #     """
    #     no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    #     seen = tf.assign_add(seen, 1.)
    #
    #     true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP + 1),
    #                                                    lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
    #                                                             true_box_wh + tf.ones_like(true_box_wh) * \
    #                                                             np.reshape(ANCHORS, [1, 1, 1, BOX, 2]) * \
    #                                                             no_boxes_mask,
    #                                                             tf.ones_like(coord_mask)],
    #                                                    lambda: [true_box_xy,
    #                                                             true_box_wh,
    #                                                             coord_mask])
    #
    #     """
    #     Finalize the loss
    #     """
    #     nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    #     # nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    #     nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
    #     nb_conf_box_pos = tf.subtract(tf.to_float(total_boxes),
    #                                   nb_conf_box_neg)  # tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))
    #     nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    #
    #     true_box_wh = tf.sqrt(true_box_wh)
    #     pred_box_wh = tf.sqrt(pred_box_wh)
    #
    #     loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    #     loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    #     loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (
    #     nb_conf_box_neg + 1e-6) / 2.
    #     loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (
    #     nb_conf_box_pos + 1e-6) / 2
    #     loss_conf = loss_conf_neg + loss_conf_pos
    #     # loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    #     loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    #     loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    #
    #     loss = tf.cond(tf.less(seen, WARM_UP + 1),
    #                    lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
    #                    lambda: loss_xy + loss_wh + loss_conf + loss_class)
    #                    # lambda: loss_xy + loss_wh + loss_conf + 10,
    #                    # lambda: loss_xy + loss_wh + loss_conf)
    #
    #     if True:
    #         nb_true_box = tf.reduce_sum(y_true[..., 4])
    #         nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.1) * tf.to_float(pred_box_conf > 0.25))
    #
    #         current_recall = nb_pred_box / (nb_true_box + 1e-6)
    #         total_recall = tf.assign_add(total_recall, current_recall)
    #
    #         total_loss = tf.assign_add(total_loss, loss)
    #
    #         #m1 = tf.reduce_max(iou_scores, axis = -1)
    #         #m2 = tf.reduce_max(pred_box_conf, axis = -1)
    #
    #
    #
    #         loss = tf.Print(loss, [iou_scores2[0][5][8]], message='\nIOU\t', summarize=1000)
    #         loss = tf.Print(loss, [pred_areas2[0][5][8]], message='\npred are \t', summarize=1000)
    #         loss = tf.Print(loss, [true_areas2[0][5][8]], message='\ntrue are \t', summarize=1000)
    #         loss = tf.Print(loss, [intersect_areas2[0][5][8]], message='\nint \t', summarize=1000)
    #         loss = tf.Print(loss, [y_true[0][5][8]], message='\nTrue XY \t', summarize=1000)
    #         loss = tf.Print(loss, [y_pred[0][5][8]], message='\nPred XY 8 \t', summarize=1000)
    #         loss = tf.Print(loss, [y_true[0][5][10]], message='\nTrue XY 10\t', summarize=1000)
    #         loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
    #         loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    #         loss = tf.Print(loss, [nb_conf_box_neg], message='Nb Conf Box Negative \t', summarize=1000)
    #         loss = tf.Print(loss, [nb_conf_box_pos], message='Nb Conf Box Positive \t', summarize=1000)
    #         loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf Negative \t', summarize=1000)
    #         loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf Positive \t', summarize=1000)
    #         loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    #         loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    #         loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    #         loss = tf.Print(loss, [total_loss / seen], message='Average Loss \t', summarize=1000)
    #         #loss = tf.Print(loss, [y_true[..., 5:]], message='\nYtrue \t', summarize=1000)
    #         #loss = tf.Print(loss, [true_box_class], message='True box class \t', summarize=1000)
    #         #loss = tf.Print(loss, [pred_box_class], message=' Pred box class \t', summarize=1000)
    #         loss = tf.Print(loss, [nb_pred_box], message='Number of pred boxes \t', summarize=1000)
    #         loss = tf.Print(loss, [nb_true_box], message='Number of true boxes \t', summarize=1000)
    #         loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    #         loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
    #
    #
    #     return loss

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_loss = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_boxes = tf.Variable(self.number_of_grids*self.number_of_grids*self.box*self.batch_size)

        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h tf.exp(
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        #conf_mask = conf_mask + tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        #conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        conf_mask_neg = tf.to_float(best_ious < 0.50) * (1 - y_true[..., 4]) * self.no_object_scale
        conf_mask_pos = y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches + 1),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * \
                                                                np.reshape(self.anchors, [1, 1, 1, self.box, 2]) * \
                                                                no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        #nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.subtract(tf.to_float(total_boxes), nb_conf_box_neg) #tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        true_box_wh = tf.sqrt(true_box_wh)
        pred_box_wh = tf.sqrt(pred_box_wh)

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (nb_conf_box_pos + 1e-6) / 2
        loss_conf = loss_conf_neg + loss_conf_pos
        #loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.32) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            total_loss = tf.assign_add(total_loss, loss)

            #m1 = tf.reduce_max(iou_scores, axis = -1)
            #m2 = tf.reduce_max(pred_box_conf, axis = -1)

            #loss = tf.Print(loss, [m2], message='\nPred box conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_neg], message='Nb Conf Box Negative \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_pos], message='Nb Conf Box Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf Negative \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [total_loss / seen], message='Average Loss \t', summarize=1000)
            # loss = tf.Print(loss, [y_true[..., 5:]], message='\nYtrue \t', summarize=1000)
            # loss = tf.Print(loss, [true_box_class], message='True box class \t', summarize=1000)
            # loss = tf.Print(loss, [pred_box_class], message=' Pred box class \t', summarize=1000)
            loss = tf.Print(loss, [nb_pred_box], message='Number of pred boxes \t', summarize=1000)
            loss = tf.Print(loss, [nb_true_box], message='Number of true boxes \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss

    def evaluate(self, generator, iou_threshold=0.3, score_threshold=0.3, max_detections=100, save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image_with_index(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            # pred_boxes = aug_image(raw_image, True)
            pred_boxes = self.predict(raw_image)

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
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
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

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
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

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (416, 416))
        image = self.normalize(image)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.box, 4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes = decode_netout(netout, self.anchors, self.NUM_CLASSES)

        return boxes

    def start_training(self):

        #model_name = "models/yolov2-tiny-coco.h5"
        model_name = "models/fullyolov2-coco.h5"
        pretrained_model = load_model(model_name, custom_objects={"tf": tf})

        indx = 0
        for layer in self.model.layers:

            name = layer.name
            if name.startswith("conv") or name.startswith("norm"):
                layer.trainable = False

            if not("not" in name):
                reshaped_weights = pretrained_model.get_layer(index=indx).get_weights()
                layer.set_weights(reshaped_weights)
            else:
                break

            indx += 1
        self.model.summary()

        f = open('annotations/carla annotations.json', 'r')
        anot = json.load(f)

        train_images = [name for name in os.listdir(self.train_data_dir)
                            if os.path.isfile(os.path.join(self.train_data_dir, name))]
        train_batch = BatchGenerator(self.config, self.train_data_dir, train_images, anot, jitter = True)

        val_images = [name for name in os.listdir(self.validation_data_dir)
                            if os.path.isfile(os.path.join(self.validation_data_dir, name))]
        val_batch = BatchGenerator(self.config, self.validation_data_dir, val_images, anot, jitter = True)

        f.close()

        # Save the model according to the conditions
        checkpoint = ModelCheckpoint("novi_pr.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)

        # optimizer = Adam(lr=0.5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        optimizer = optimizers.SGD(lr=0.00001, momentum=0.9, decay=0.0005)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer, metrics=["accuracy"])

        # map_evaluator_cb = mAp(self, val_batch,
        #                                        save_best=True,
        #                                        save_name='checkpoints\\best-mAP.h5',
        #                                        tensorboard=None,
        #                                        iou_threshold=0.4)

        self.model.fit_generator(generator=train_batch,
                                 steps_per_epoch=len(train_batch) * 5,
                                 epochs=self.epochs,
                                 validation_data=val_batch,
                                 validation_steps=len(val_batch),
                                 callbacks=[checkpoint])
