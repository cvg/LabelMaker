import pandas as pd
import numpy as np
import os
from pathlib import Path
from segmentation_tools.label_data import ADE150, REPLICA, get_wordnet


def set_ids_according_to_names():
    table_path = str(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    table = pd.read_csv(table_path)
    for row in table.index:
        if not table['ade class'].isnull()[row]:
            name = table.loc[row, 'ade class']
            mapped_id = ''
            if ',' in name:
                for n in name.split(','):
                    n = n.strip(' ')
                    item = next(x for x in ADE150 if x['name'] == n)
                    mapped_id += f"{item['id']},"
                mapped_id = mapped_id[:-1]  # remove last comma
            else:
                item = next(x for x in ADE150 if x['name'] == name)
                mapped_id = str(item['id'])
            table.loc[row, 'ade20k'] = mapped_id
        if not table['replicaclass'].isnull()[row]:
            name = table.loc[row, 'replicaclass']
            mapped_id = ''
            if ',' in name:
                for n in name.split(','):
                    n = n.strip(' ')
                    item = next(x for x in REPLICA if x['name'] == n)
                    mapped_id += f"{item['id']},"
                mapped_id = mapped_id[:-1]  # remove last comma
            else:
                item = next(x for x in REPLICA if x['name'] == name)
                mapped_id = str(item['id'])
            table.loc[row, 'replicaid'] = mapped_id
    table.to_csv(table_path)


class LabelMatcher:

    def __init__(self, mapping_left, mapping_right):
        table = pd.read_csv(
            Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
            'label_mapping.csv')
        # first find out which space has more labels, as we map from more to less labels
        self.left_ids = []
        self.right_ids = []
        for row in table.index:
            if not table[mapping_left].isnull()[row]:
                try:
                    self.left_ids.append(int(table[mapping_left][row]))
                except ValueError:
                    self.left_ids.extend(
                        [int(x) for x in table[mapping_left][row].split(',')])
            if not table[mapping_right].isnull()[row]:
                try:
                    self.right_ids.append(int(table[mapping_right][row]))
                except ValueError:
                    self.right_ids.extend(
                        [int(x) for x in table[mapping_right][row].split(',')])
        self.left_ids = sorted(list(set(self.left_ids)))
        self.right_ids = sorted(list(set(self.right_ids)))
        if len(self.left_ids) >= len(self.right_ids):
            self.mapping_from = mapping_left
            self.mapping_to = mapping_right
            self.left_to_right = True
        else:
            self.mapping_from = mapping_right
            self.mapping_to = mapping_left
            self.left_to_right = False
        # create mapping
        size_left = max(self.left_ids)
        size_right = max(self.right_ids)
        self.mapping = -1 * np.ones(
            (size_left if self.left_to_right else size_right) + 1, dtype=int)
        self.mapping_multiples = {}
        for row in table.index:
            if table[self.mapping_to].isnull()[row] or table[
                    self.mapping_from].isnull()[row]:
                continue
            from_ids = table.loc[row, self.mapping_from]
            try:
                from_ids = [int(from_ids)]
            except ValueError:
                from_ids = [int(x) for x in from_ids.split(',')]
            to_ids = table.loc[row, self.mapping_to]
            try:
                to_ids = [int(to_ids)]
                for from_id in from_ids:
                    self.mapping[from_id] = to_ids[0]
            except ValueError:
                to_ids = [int(x) for x in to_ids.split(',')]
                for from_id in from_ids:
                    self.mapping[from_id] = -2
                    self.mapping_multiples[from_id] = to_ids

    def match(self, left, right):
        left = left.astype(int)
        right = right.astype(int)
        if self.left_to_right:
            right_from_left = self.mapping[left]
            matching = np.equal(right, right_from_left).astype(int)
            matching[right_from_left == -1] = -1
            if np.sum(right_from_left == -2) > 0:
                for from_id, to_ids in self.mapping_multiples.items():
                    matching[left == from_id] = np.isin(
                        right[left == from_id], to_ids)
        else:
            left_from_right = self.mapping[right]
            matching = np.equal(left, left_from_right).astype(int)
            matching[left_from_right == -1] = -1
            if np.sum(left_from_right == -2) > 0:
                for from_id, to_ids in self.mapping_multiples.items():
                    matching[right == from_id] = np.isin(
                        left[right == from_id], to_ids)
        return matching

    def confusion_matrix(self, left, right):
        """
        assumes left is prediction and right is label
        confusion matrix is always built in the space of the label
        """
        confmat = np.zeros((len(self.right_ids), len(self.right_ids)),
                           dtype=int)
        matching = self.match(left, right)
        if self.left_to_right:
            mapped_pred = self.mapping[left]
        else:
            mapped_pred = left
        for l, label in enumerate(self.right_ids):
            mismatches = np.logical_and(right == label, matching == 0)
            for p, pred in enumerate(self.right_ids):
                if label == pred:
                    confmat[p, l] = np.sum(
                        np.logical_and(right == label, matching == 1))
                    continue
                if self.left_to_right:
                    confmat[p, l] = np.sum(
                        np.logical_and(mismatches, mapped_pred == pred))
                else:
                    possible_preds = self.mapping[pred]
                    if possible_preds == -1:
                        continue
                    if possible_preds == -2:
                        confmat[p, l] += np.sum(
                            np.logical_and(
                                mismatches,
                                np.isin(left, self.mapping_multiples[pred])))
                    else:
                        confmat[p, l] = np.sum(
                            np.logical_and(mismatches, left == possible_preds))
        # now handle errors where the mapped prediction is ambiguous
        if self.left_to_right and np.sum(mapped_pred[matching == 0] == -2) > 0:
            for left_id, right_ids in self.mapping_multiples.items():
                wrong_predictions = np.logical_and(matching == 0,
                                                   left == left_id)
                if np.sum(wrong_predictions) == 0:
                    continue
                for l, label in enumerate(self.right_ids):
                    additional_false_matches = np.sum(
                        np.logical_and(wrong_predictions, right == label))
                    for right_id in right_ids:
                        # true label is label, matching is 0, one of the possible matches for left_id is right_id
                        confmat[self.right_ids.index(right_id),
                                l] += additional_false_matches
        return confmat


class MatcherScannetWordnet199:

    def __init__(self):
        self.mapping_from_scannet = get_wn199_from_scannet()

    def match(self, scannet, wordnet199):
        wordnet199 = wordnet199.astype(int)
        scannet = scannet.astype(int)
        wordnet199_from_scannet = self.mapping_from_scannet[scannet]
        matching = np.equal(wordnet199_from_scannet, wordnet199).astype(int)
        matching[wordnet199_from_scannet == -1] = -1
        matching[wordnet199 == -1] = -1
        return matching


def match_scannet_ade150(scannet, ade150):
    matcher = LabelMatcher('id', 'ade20k')
    return matcher.match(scannet, ade150)


def match_scannet_nyu40(scannet, nyu40):
    matcher = LabelMatcher('id', 'nyu40id')
    return matcher.match(scannet, nyu40)


def match_ade150_nyu40(ade150, nyu40):
    matcher = LabelMatcher('ade20k', 'nyu40id')
    return matcher.match(ade150, nyu40)


def match_scannet_wordnet199(scannet, wordnet199):
    matcher = MatcherScannetWordnet199()
    return matcher.match(scannet, wordnet199)


def get_ade150_to_scannet():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    mapping = -1 * np.ones(151, dtype=int)
    for row in table.index:
        if table['ade20k'].isnull()[row]:
            continue
        ade150_ids = [int(x) for x in table.loc[row, 'ade20k'].split(',')]
        scannet_id = table.loc[row, 'id']
        for ade150_id in ade150_ids:
            # we use the first scannet id occuring, since table is sorted by count
            if mapping[ade150_id] != -1:
                continue
            mapping[ade150_id] = scannet_id
    return mapping


def get_wn199_from_scannet():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    wordnet = get_wordnet()
    wordnet_keys = [x['name'] for x in wordnet]
    mapping = -1 * np.ones(table['id'].max() + 1, dtype=int)
    for row in table.index:
        if table['wnsynsetkey'][row] not in wordnet_keys:
            continue
        scannet_id = table.loc[row, 'id']
        wordnet199_id = next(x for x in wordnet
                             if x['name'] == table['wnsynsetkey'][row])['id']
        mapping[scannet_id] = int(wordnet199_id)
    return mapping


def get_scannet_from_wn199():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    wordnet = get_wordnet()
    wordnet_keys = [x['name'] for x in wordnet]
    mapping = -1 * np.ones(199, dtype=int)
    for row in table.index:
        if table['wnsynsetkey'][row] not in wordnet_keys:
            continue
        scannet_id = table.loc[row, 'id']
        wordnet199_id = next(x for x in wordnet
                             if x['name'] == table['wnsynsetkey'][row])['id']
        mapping[int(wordnet199_id)] = scannet_id
    return mapping
