# modify from https://github.com/tensorflow/tensorflow/tree/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os import path
import sys
import scipy.io

_SYNSET_ARRAYS_RELATIVE_PATH = 'data/meta.mat'
_VALIDATION_FILE_RELATIVE_PATH = 'data/ILSVRC2012_validation_ground_truth.txt'


def _synset_to_word(filepath):
    """Returns synset to word dictionary by reading sysnset arrays."""
    mat = scipy.io.loadmat(filepath)
    entries = mat['synsets']
    # These fields are listed in devkit readme.txt
    fields = [
        'synset_id', 'WNID', 'words', 'gloss', 'num_children', 'children',
        'wordnet_height', 'num_train_images'
    ]
    synset_index = fields.index('synset_id')
    words_index = fields.index('words')
    synset_to_word = {}
    for entry in entries:
        entry = entry[0]
        synset_id = int(entry[synset_index][0])
        first_word = entry[words_index][0].split(',')[0]
        synset_to_word[synset_id] = first_word
    return synset_to_word


def _validation_file_path(ilsvrc_dir):
    return path.join(ilsvrc_dir, _VALIDATION_FILE_RELATIVE_PATH)


def _synset_array_path(ilsvrc_dir):
    return path.join(ilsvrc_dir, _SYNSET_ARRAYS_RELATIVE_PATH)


def _generate_validation_labels(ilsvrc_dir, synset_dir, output_file):
    synset_to_word = _synset_to_word(_synset_array_path(ilsvrc_dir))
    synset_words_dict = {}
    with open(synset_dir, 'r') as synset_tans:
        for count, line in enumerate(synset_tans):
            synset_words_dict[line.strip().split(',')[0]] = count

    with open(_validation_file_path(ilsvrc_dir), 'r') as synset_id_file, open(output_file, 'w') as output:
        for synset_id in synset_id_file:
            synset_id = int(synset_id)
            output.write('%d\n' % synset_words_dict[synset_to_word[synset_id]])


def _check_arguments(args):
    if not args.validation_labels_output:
        raise ValueError('Invalid path to output file.')
    ilsvrc_dir = args.ilsvrc_devkit_dir
    if not ilsvrc_dir or not path.isdir(ilsvrc_dir):
        raise ValueError('Invalid path to ilsvrc_dir')
    if not path.exists(_validation_file_path(ilsvrc_dir)):
        raise ValueError('Invalid path to ilsvrc_dir, cannot find validation file.')
    if not path.exists(_synset_array_path(ilsvrc_dir)):
        raise ValueError(
            'Invalid path to ilsvrc_dir, cannot find synset arrays file.')


def main():
    parser = argparse.ArgumentParser(
        description='Converts ILSVRC devkit validation_ground_truth.txt to synset'
        ' labels file that can be used by the accuracy script.')
    parser.add_argument(
        '--validation_labels_output',
        type=str,
        help='Full path for outputting validation label id.')
    parser.add_argument(
        '--ilsvrc_devkit_dir',
        type=str,
        help='Full path to ILSVRC 2012 devkit directory.')
    parser.add_argument(
        '--synset_words_file',
        type=str,
        help='Full path to synset words file.')
    args = parser.parse_args()
    try:
        _check_arguments(args)
    except ValueError as e:
        parser.print_usage()
        file_name = path.basename(sys.argv[0])
        sys.stderr.write('{0}: error: {1}\n'.format(file_name, str(e)))
        sys.exit(1)
    _generate_validation_labels(args.ilsvrc_devkit_dir,
                                args.synset_words_file,
                                args.validation_labels_output)


if __name__ == '__main__':
    main()
