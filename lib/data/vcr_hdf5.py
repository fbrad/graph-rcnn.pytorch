import os
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import glob
import cv2
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.utils.box import bbox_overlaps
from .transforms import Compose
from torchvision.transforms import ToPILImage

class vcr_hdf5(Dataset):
    def __initold__(self, cfg, transforms: Compose = None):
        self.data_dir = cfg.DATASET.PATH

        # keep same object and predicate indices as in Visual Genome
        self.info = json.load(open(os.path.join(self.data_dir,
                                                "VG-SGG-dicts.json"), 'r'))
        # load object and predicate indices
        self.info['label_to_idx']['__background__'] = 0

        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                                        self.class_to_ind[k])

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                        self.predicate_to_ind[k])
        self.transforms = transforms

        # list of all image filenames (from all subdirs of self.image_dir)
        img_fns = glob.glob(self.data_dir + "/*/*.jpg")
        img_fns = [img_fn for img_fn in img_fns if not "detection" in img_fn]
        self.img_fns = img_fns
        img_meta_fns = [fn.replace(".jpg", ".json") for fn in img_fns]

        # allocate memory for all files
        dataset_size = len(img_fns)
        self.images = torch.zeros(dataset_size, 3, 1024, 1024)
        self.img_info = []

        # [[[pic1_box1], [pic1_box2]], [[pic2_box1]], ... ]
        self.boxes = []
        self.objects = []

        indices = list(range(len(img_fns)))

        for idx, img_fn, img_meta_fn in zip(indices, img_fns, img_meta_fns):
            # make sure they refer to the same image
            assert img_fn[:-3] == img_meta_fn[:-4]

            img_meta = json.load(open(img_meta_fns[0], "r"))
            img_boxes = img_meta["boxes"]
            img_objects = img_meta["names"]

            self.boxes.append(torch.tensor(img_boxes))
            self.objects.append(img_objects)
            #print("[vcr_hdf5] boxes[", idx, "] boxes = ", boxes)

            img_data = cv2.imread(img_fn)
            assert img_data.ndim == 3, "Grayscale image"

            # resize image to 1024x1024
            h0, w0 = img_data.shape[0], img_data.shape[1]

            scale = 1024.0/max(h0, w0)
            hscaled = int(h0 * scale)
            wscaled = int(w0 * scale)
            self.img_info.append([wscaled, hscaled])

            #print("[vcr] hscaled, wscaled = ", hscaled, wscaled)
            img = cv2.resize(img_data, dsize=(wscaled, hscaled))

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            #print("[vcr_hdf5] img_tensor = ", img_tensor.size())

            #copy image tensor to dataset tensor
            self.images[idx, :, 0:hscaled] = img_tensor
            print("[vcr_hdf5.py:77] self.images = ", self.images.size())

            #print("[vcr_hdf5:56] img.shape = ", img.shape, " old shape = ",
            # img_data.shape)

            #cv2.imshow('bla', img)
            #cv2.waitKey()

        #print("[VCR] img_folders = ", img_folders)

    def __init__(self, cfg, split: str = "train", transforms: Compose = None):
        # datasets/vcr
        self.data_dir = cfg.DATASET.PATH

        # keep same object and predicate indices as in Visual Genome
        self.info = json.load(open(os.path.join(self.data_dir,
                                                "VG-SGG-dicts.json"), 'r'))

        # load object and predicate indices
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                                        self.class_to_ind[k])
        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                        self.predicate_to_ind[k])

        # load transforms (series of operations on PIL data)
        self.transforms = transforms

        # load all image filenames
        annotation_file = os.path.join(self.data_dir, '{}.jsonl'.format(split))
        self.items = set()
        # with open(annotation_file, 'r') as f:
        #     for line in f:
        #         self.items.add(json.loads(line)["img_fn"])

        # update self.data_dir
        self.data_dir = os.path.join(self.data_dir, "vcr1images")

        # list of all image filepaths (from all subdirs of self.image_dir)
        img_fns = glob.glob(self.data_dir + "/*/*.jpg")
        self.img_fns = [img_fn for img_fn in img_fns if not "detection" in img_fn]
        #img_meta_fns = [fn.replace(".jpg", ".json") for fn in self.img_fns]

        # width and height info
        self.img_info = []

        # load image metadata
        self.boxes = []
        self.objects = []
        for img_fn in self.img_fns:
            # load image metadata
            meta_fn = img_fn.replace(".jpg", ".json")
            with open(meta_fn, 'r') as f:
                meta_item = json.load(f)
                img_boxes = meta_item["boxes"] # num_objs x 5
                img_objects = meta_item["names"] # num_objs

                self.boxes.append(torch.tensor(img_boxes))
                self.objects.append(img_objects)
                self.img_info.append({"width": meta_item["width"],
                                      "height": meta_item["height"]})

        self.to_pil_obj = ToPILImage()

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', mode='answer', **kwargs)] + [
            cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.img_fns)

    def __old_getitem__(self, index):
        # img = Image.fromarray(self.images[index].numpy())
        # sizes = self.images[index].size()
        # w, h = sizes[1], sizes[2]
        to_pil = ToPILImage()
        img = to_pil(self.images[index])
        target = img.copy()
        img, _ = self.transforms(img, target)

        #return img, self.boxes[index], self.objects[index], index
        return self.images[index], self.boxes[index], self.objects[index], index

    def __getitem__(self, index):
        #item = self.items[index]
        #img_fn = os.path.join(self.data_dir, item['img_fn'])
        img_data = cv2.imread(self.img_fns[index])
        assert img_data.ndim == 3, "Grayscale image"

        piled_img = self.to_pil_obj(img_data)
        target = piled_img.copy()
        img, _ = self.transforms(piled_img, target)

        return img, self.boxes[index], self.objects[index], index

    def get_img_info(self, idx):
        w, h = self.img_info[idx]
        return {"height": h, "width": w}

    def get_img_fname(self, idx) -> str:
        """
        Returns the name of the .jpg file corresponding to the example at
        index idx
        :param idx:
        :return: str
        """
        return self.img_fns[idx]

    def _get_dets_to_use(self, item):
        """
        We might want to use fewer detectiosn so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def plot_image_box_from_minivcr(self):
        full_path = os.path.join(os.getcwd(), self.data_dir)

        img_dirs = []
        for f in listdir(full_path):
            full_name = os.path.join(full_path, f)
            if isdir(full_name):
                img_dirs.append(full_name)

        for img_dir in img_dirs:
            print("[VCR] looking for ", img_dir + "/*.jpg")
            img_fns = glob.glob(img_dir + "/*.jpg")
            img_meta_fns = glob.glob(img_dir + "/*.json")
            print("[VCR images] ", img_fns)
            print("[VCR jsons] ", img_meta_fns)

            img_meta = json.load(open(img_meta_fns[0], "r"))
            boxes = img_meta["boxes"]
            print("[VCR meta] ", boxes)

            img_data = cv2.imread(img_fns[0])
            print("[VCR image] ", img_data.shape)
            box = boxes[0]
            cv2.rectangle(img_data, (round(box[0]), round(box[1])),
                          (round(box[2]), round(box[3])), color=(255, 255, 255))
            cv2.imshow('ImageWindow', img_data)
            cv2.waitKey()
