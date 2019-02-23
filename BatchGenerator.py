import os
import cv2
import copy
import numpy as np
import json
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, bbox_iou


# def draw_rec(image, positions):
#     im = np.array(Image.open(image), dtype=np.uint8)
#     im = cv2.resize(im, (consts.IMAGE_WIDTH, consts.IMAGE_HEIGHT))
#     dy = int(np.floor(consts.IMAGE_WIDTH / consts.HORIZONTAL_GRIDS))
#     dx = int(np.floor(consts.IMAGE_HEIGHT / consts.VERTICAL_GRIDS))
#
#     # Custom (rgb) grid color
#     grid_color = [0, 0, 255]
#
#     # Modify the image to include the grid
#     im[:, ::dy, :] = grid_color
#     im[::dx, :, :] = grid_color
#
#     # plt.imshow(im)
#     # plt.show()
#
#
#     #Create figure and axes
#     fig, ax = plt.subplots()
#
#     # Display the image
#     ax.imshow(im)
#
#     # Create a Rectangle patch
#     for pos in positions:
#         rect = patches.Rectangle((pos[0], pos[1]), pos[2], pos[3], linewidth=3, edgecolor='r', facecolor='none')
#
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#     plt.grid(b=True, which='major', axis='both')
#     plt.show()

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    f = open(img_dir + 'annotations.json', 'r')
    annotations = json.load(f)
    f.close()

    for set in annotations:
        for sub_set in set:
            for frame in sub_set["frames"]:
                img = {'object': []}
                img['filename'] = img_dir + set + "_" + sub_set + "_" + frame
                img['width'] = 640
                img['height'] = 480
                for objects in frame:
                    obj = {}
                    persons_pos = objects["pos"];
                    obj['name'] = objects["lbl"]
                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] += 1
                    else:
                        seen_labels[obj['name']] = 1
                    if len(labels) > 0 and obj['name'] not in labels:
                        break
                    else:
                        img['object'] += [obj]

                    obj['xmin'] = int(round(float(persons_pos[0])))
                    obj['ymin'] = int(round(float(persons_pos[1])))
                    obj['xmax'] = int(round(float(persons_pos[2])))
                    obj['ymax'] = int(round(float(persons_pos[3])))

    if len(img['object']) > 0:
        all_imgs += [img]

    # return all_imgs, seen_labels

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, config, images_dir, images, annotations,
                 shuffle=True,
                 jitter=False,
                 norm=None):

        self.config = config
        self.images = images
        self.images_dir = images_dir
        self.annotations = annotations
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.image_height = self.config["model"]["image_size"]
        self.image_width = self.config["model"]["image_size"]
        self.true_box_buffer = self.config["model"]["nb_box"]
        self.number_of_grids = self.config["model"]["horizontal_grids"]
        self.box = self.config["model"]["box"]
        self.input_image_width = self.config["model"]["input_image_width"]
        self.input_image_height = self.config["model"]["input_image_height"]
        self.anchors = [BoundBox(0, 0, config["model"]["anchors"][2 * i], config["model"]["anchors"][2 * i + 1]) for i
                        in
                        range(int(len(config["model"]["anchors"]) // 2))]

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config["train"]["batch_size"]))

    def num_classes(self):
        return len(self.config["model"]["labels"])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],
                     self.config["main"]["labels"].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    # def load_image(self, i):
    #     return cv2.imread(self.images[i])

    def __getitem__(self, idx):
        l_bound = idx * self.config["train"]["batch_size"]
        r_bound = (idx + 1) * self.config["train"]["batch_size"]

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config["train"]["batch_size"]

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.image_height, self.image_width, 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.true_box_buffer,
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.number_of_grids, self.number_of_grids, self.box,
                            4 + 1 + self.num_classes()))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # print(train_instance)
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.image_width) / self.number_of_grids)
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.image_height) / self.number_of_grids)

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.number_of_grids and grid_y < self.number_of_grids:

                        center_w = (obj['xmax'] - obj['xmin']) / (
                            float(self.image_width) / self.number_of_grids)  # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (
                            float(self.image_height) / self.number_of_grids)  # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.true_box_buffer

            # assign input image to x_batch
            if self.norm is None:
                x_batch[instance_count] = self.normalize(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:, :, ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[:, :, ::-1], "person",
                                    (obj['xmin'] + 2, obj['ymin'] + 12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0, 255, 0), 2)

                x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1

            # print(' new batch created', idx)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def scale_image_anot(self, anotations):
        anotations[0] = anotations[0] * self.image_width / self.input_image_width
        anotations[2] = anotations[2] * self.image_width / self.input_image_width
        anotations[1] = anotations[1] * self.image_height / self.input_image_height
        anotations[3] = anotations[3] * self.image_height / self.input_image_height

        return anotations

    def normalize(self, tensor):
        return tensor / 255

    def read_carla_annotations(self, image):
        found = []
        for image_file in self.annotations:
            image_name = image.split("png")[0] + "png"
            json_key = self.annotations[image_file]["filename"]
            if image_name == json_key:
                regions = self.annotations[image_file]["regions"]
                for region in regions:
                    shape_attributes = region["shape_attributes"]
                    xmin = shape_attributes["x"]
                    ymin = shape_attributes["y"]
                    w = shape_attributes["width"]
                    h = shape_attributes["height"]
                    found.append([xmin, ymin, w, h])

        return found

    def load_image(self, image_path):
        return cv2.imread(self.images_dir + image_path)

    def load_image_with_index(self, i):
        return cv2.imread(self.images[i])

    def aug_image(self, image_path, jitter):

        image_name = image_path.split("/")[-1]
        # image_name = image_name.split("\\")[-1]
        # print(image_name)
        features = image_name.split("_")
        main_set = features[0]
        sub_set = features[1]
        frame = features[2].split(".")[0]
        # print(image_name + "\n", main_set+ "\n", sub_set+ "\n", frame+ "\n")
        # image = np.expand_dims(image, 0)

        objects = []
        positions = []

        # CARLA
        persons = self.read_carla_annotations(image_name)
        for person in persons:
            last_person_pos = copy.deepcopy(person)
            positions.append(self.scale_image_anot(last_person_pos))
            obj = {'xmin': last_person_pos[0], 'ymin': last_person_pos[1],
                   'xmax': (last_person_pos[0] + last_person_pos[2]),
                   'ymax': (last_person_pos[1] + last_person_pos[3])}
            if last_person_pos[2] > 10:
                objects.append(obj)
        # draw_rec(self.images_dir + image_path, positions)

        # get all features of the frame
        # persons = []
        # try:
        #     persons = self.annotations[main_set][sub_set]['frames'][frame]
        # except:
        #     pass
        # for every person take their bounding box coordinates and scale them
        # for person in persons:
        #     last_person_pos = copy.deepcopy(person['pos'])
        #     # last_person_pos = self.scale_image_anot(last_person_pos)
        #     obj = {'xmin': last_person_pos[0], 'ymin': last_person_pos[1], 'xmax': (last_person_pos[0] + last_person_pos[2]),
        #            'ymax': (last_person_pos[1] + last_person_pos[3])}
        #     objects.append(obj)

        image = self.load_image(image_path)
        h, w, c = image.shape
        # image = cv2.resize(image, (consts.IMAGE_WIDTH, consts.IMAGE_HEIGHT))

        if image is None: print('Cannot find ', image_name)

        all_objs = objects

        if jitter:
            # scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            # translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            # flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            # image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.image_height, self.image_width))
        image = image[:, :, ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.image_width) / w)
                obj[attr] = max(min(obj[attr], self.image_width), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.image_height) / h)
                obj[attr] = max(min(obj[attr], self.image_height), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.image_width - obj['xmax']
                obj['xmax'] = self.image_width - xmin

        return image, all_objs
