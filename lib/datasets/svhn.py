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
from .svhn_eval import svhn_eval

from svhn.io import SVHN, TRAINING, TEST

# _base_dir = os.path.expanduser('~/.svhn')
_base_dir = '/scratch/rnm6u/svhn/images'
_datasets = {'train': TRAINING, 'test': TEST}
_image_dirs = {TRAINING: 'train', TEST: 'test'}

class svhn(imdb):
    def __init__(self, dataset):
        classes = ['__background__'] + [str(i % 10) for i in range(1, 10+1)]
        imdb.__init__(self, 'svhn_' + dataset, classes)
        self._dataset_name = dataset
        self._dataset = _datasets[dataset]
        self._svhn = SVHN()
        size = self._svhn.size_full(self._dataset)
        self._image_index = range(1, size + 1)
        self._image_dir = os.path.join(_base_dir,
                                       _image_dirs[self._dataset])
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

        objs = self._svhn.bounding_boxes(self._dataset, index-1).boxes
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = obj.label
            x1 = min(max(0, obj.left), width - 1)
            y1 = min(max(0, obj.top), height - 1)
            x2 = min(max(x1, x1 + obj.width - 1), width - 1)
            y2 = min(max(y1, y1 + obj.height - 1), height - 1)
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

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'svhn_det_' + self._dataset_name + '_{:s}.txt'
        filedir = 'results'
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} SVHN results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join('results', 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = svhn_eval(
                self._svhn, self._dataset, filename, cls, cachedir, ovthresh=0.5,
                use_07_metric=False)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
