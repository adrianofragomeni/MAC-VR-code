from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset
import numpy as np


class YouCookDataset(RetrievalDataset):
    """YouCook2 dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(YouCookDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        csv_path = {'train': join(self.anno_path, 'YC2_retrieval_train.csv'),
                    'train_test': join(self.anno_path, 'YC2_retrieval_train.csv'),
                    'val': join(self.anno_path, 'YC2_retrieval_test.csv'),
                    'test': join(self.anno_path, 'YC2_retrieval_test.csv')}[subset]
        if exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError

        video_id_list = list(np.unique(csv['video_id'].values))

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        visual_concept_dict = OrderedDict()
        textual_concept_dict = OrderedDict()
        if subset == 'train':
            for row in csv.values:
                if row[0] in video_id_list:
                    sentences_dict[len(sentences_dict)] = (row[0], (row[1], None, None))
                    visual_concept_dict[len(visual_concept_dict)] = (row[0], [row[3], None, None])
                    textual_concept_dict[len(textual_concept_dict)] = (row[0], [row[2], None, None])                    
                    video_dict[row[0]] = join(self.video_path, "training/{}.mp4".format(row[0]))
        elif subset == 'train_test':
          used = []
          for row in csv.values:
                if row[0] in video_id_list and row[0] not in used:
                    used.append(row[0])
                    sentences_dict[len(sentences_dict)] = (row[0], (row[1], None, None))
                    visual_concept_dict[len(visual_concept_dict)] = (row[0], [row[3], None, None])
                    textual_concept_dict[len(textual_concept_dict)] = (row[0], [row[2], None, None])                    
                    video_dict[row[0]] = join(self.video_path, "training/{}.mp4".format(row[0]))
        else:
            for row in csv.values:
                sentences_dict[len(sentences_dict)] = (row[0], (row[1], None, None))                   
                visual_concept_dict[len(visual_concept_dict)] = (row[0], [row[3], None, None])                    
                textual_concept_dict[len(textual_concept_dict)] = (row[0], [row[2], None, None])                      
                video_dict[row[0]] = join(self.video_path, "validation/{}.mp4".format(row[0]))

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict, textual_concept_dict, visual_concept_dict
