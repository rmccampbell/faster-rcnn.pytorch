from __future__ import print_function
from __future__ import absolute_import

import os
import os.path as osp
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import PIL
import pickle

from svhn.io import SVHN, TRAINING, TEST

# _base_dir = os.path.expanduser('~/.svhn')
_base_dir = '/scratch/rnm6u/svhn/images'
_splits = {'train': TRAINING, 'test': TEST}
_image_dirs = {TRAINING: 'train', TEST: 'test'}

class svhn(imdb):
    def __init__(self, split):
        classes = ['__background__'] + [str(i) for i in range(10)]
        imdb.__init__(self, 'svhn_' + split, classes)
        self._split = _splits[split]
        self._svhn = SVHN()
        size = self._svhn.size_full(self._split)
        self._image_index = range(1, size + 1)
        self._image_dir = os.path.join(_base_dir,
                                       _image_dirs[self._split])
        self.set_proposal_method('gt')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i + 1)

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i + 1

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        return os.path.join(self._image_dir, '%d.png' % index)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._get_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _get_size(self, index):
      return PIL.Image.open(self.image_path_from_index(index)).size

    def _get_annotation(self, index):
        width, height = self._get_size(index)

        objs = self._svhn.bounding_boxes(self._split, index-1).boxes
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = obj.label
            x1 = min(max(0, obj.left), width-1)
            y1 = min(max(0, obj.top), height-1)
            x2 = min(max(x1, x1 + obj.width - 1), width-1)
            y2 = min(max(y1, y1 + obj.height - 1), height-1)
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def append_flipped_images(self):
        """
        Don't include flipped images
        """
        pass

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        # TODO: IMPLEMENT EVALUATION
