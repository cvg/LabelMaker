import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from segmentation_tools.label_data import ADE150, REPLICA, get_wordnet, get_nyu40, SCANNET_COLOR_MAP_200
from sklearn.metrics import confusion_matrix


def set_ids_according_to_names():
    table_path = str(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    table = pd.read_csv(table_path)
    wn199 = get_wordnet()
    nyu40 = get_nyu40()
    wn199_to_id = {x['name']: int(x['id']) for x in wn199}
    for row in table.index:
        if not table['ade class'].isnull()[row]:
            name = table.loc[row, 'ade class']
            mapped_id = ''
            if ',' in name:
                for n in name.split(','):
                    n = n.strip(' ')
                    item = next(x for x in ADE150 if x['name'] == n)
                    mapped_id += f"{int(item['id'])},"
                mapped_id = mapped_id[:-1]  # remove last comma
            else:
                item = next(x for x in ADE150 if x['name'] == name)
                mapped_id = str(int(item['id']))
            table.loc[row, 'ade20k'] = mapped_id
        if not table['replicaclass'].isnull()[row]:
            name = table.loc[row, 'replicaclass']
            mapped_id = ''
            if ',' in name:
                for n in name.split(','):
                    n = n.strip(' ')
                    item = next(x for x in REPLICA if x['name'] == n)
                    mapped_id += f"{int(item['id'])},"
                mapped_id = mapped_id[:-1]  # remove last comma
            else:
                item = next(x for x in REPLICA if x['name'] == name)
                mapped_id = str(int(item['id']))
            table.loc[row, 'replicaid'] = mapped_id
        if not table['wnsynsetkey'].isnull()[row]:
            name = table.loc[row, 'wnsynsetkey']
            if name in wn199_to_id:
                table.loc[row, 'wn199'] = str(int(wn199_to_id[name]))
        if not table['nyu40class'].isnull()[row]:
            name = table.loc[row, 'nyu40class']
            mapped_id = ''
            if ',' in name:
                for n in name.split(','):
                    n = n.strip(' ')
                    print(n)
                    item = next(x for x in nyu40 if x['name'] == n)
                    mapped_id += f"{int(item['id'])},"
                mapped_id = mapped_id[:-1]
            else:
                item = next(x for x in nyu40 if x['name'] == name)
                mapped_id = str(int(item['id']))
            table.loc[row, 'nyu40id'] = mapped_id
    table.to_csv(table_path, index=False)


def set_colors(label_key='wn199-merged-v2'):
    table_path = str(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    table = pd.read_csv(table_path)
    wn199 = get_wordnet(label_key=label_key)
    
    wd_ids = [x['id'] for x in wn199]
    
    random_colors = np.random.randint(0, 255, (max(wd_ids) + 1, 3))
    for row in table.index:
        scannetid = int(table.loc[row, 'id'])
        if scannetid in SCANNET_COLOR_MAP_200:
            table.loc[row, 'color'] = str("-".join(
                str(int(x)) for x in SCANNET_COLOR_MAP_200[scannetid]))
        elif not table[label_key].isnull()[row]:
            table.loc[row, 'color'] = str("-".join(
                str(int(x)) for x in random_colors[int(table.loc[row, label_key])]))
        else:
            table.loc[row, 'color'] = str("-".join(
                str(int(x)) for x in np.random.randint(0, 255, 3)))
    table.to_csv(table_path, index=False)


class LabelMatcher:

    def __init__(self, mapping_left, mapping_right, verbose=False):
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
        if verbose:
            print(f"Mapping from {self.mapping_from} to {self.mapping_to}")
        # create mapping
        size_left = max(self.left_ids)
        size_right = max(self.right_ids)
        self.left2right = defaultdict(list)
        self.right2left = defaultdict(list)
        self.mapping = -1 * np.ones(
            (size_left if self.left_to_right else size_right) + 1, dtype=int)
        self.mapping_multiples = {}
        from_to = defaultdict(list)
        # to avoid overwriting, we first create a mapping from -> to
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
            except ValueError:
                to_ids = [int(x) for x in to_ids.split(',')]
            for from_id in from_ids:
                from_to[from_id].extend(to_ids)
            if self.left_to_right:
                for from_id in from_ids:
                    self.left2right[from_id].extend(to_ids)
                for to_id in to_ids:
                    self.right2left[to_id].extend(from_ids)
            else:
                for from_id in from_ids:
                    self.right2left[from_id].extend(to_ids)
                for to_id in to_ids:
                    self.left2right[to_id].extend(from_ids)
        for from_id, to_ids in from_to.items():
            to_ids = list(set(to_ids))
            if verbose:
                print(f"{from_id} -> {to_ids}")
            if len(to_ids) == 1:
                self.mapping[from_id] = to_ids[0]
            else:
                self.mapping[from_id] = -2
                self.mapping_multiples[from_id] = to_ids
        self.left2right = {k: list(set(v)) for k, v in self.left2right.items()}
        self.right2left = {k: list(set(v)) for k, v in self.right2left.items()}

    def match(self, left, right, verbose=False):
        """
        matches between left and right label spaces
        returns a matrix of same shape as left and right, where
        1 indicates a match, 0 indicates a mismatch and -1 indicates
        that there is no known mapping
        """
        left = left.astype(int)
        right = right.astype(int)
        if self.left_to_right:
            right_from_left = self.mapping[left]
            matching = np.equal(right, right_from_left).astype(int)
            if np.sum(right_from_left == -2) > 0:
                for from_id, to_ids in self.mapping_multiples.items():
                    matching[left == from_id] = np.isin(
                        right[left == from_id], to_ids)
            matching[right_from_left == -1] = -1
            matching[np.logical_not(np.isin(right, self.right_ids))] = -1
        else:
            left_from_right = self.mapping[right]
            matching = np.equal(left, left_from_right).astype(int)
            if np.sum(left_from_right == -2) > 0:
                for from_id, to_ids in self.mapping_multiples.items():
                    matching[right == from_id] = np.isin(
                        left[right == from_id], to_ids)
            matching[left_from_right == -1] = -1
            matching[np.logical_not(np.isin(left, self.left_ids))] = -1
        return matching

    def confusion_matrix(self, left, right):
        """
        assumes left is prediction and right is label
        confusion matrix is always built in the space of the label
        """
        confmat = np.zeros((len(self.right_ids), len(self.right_ids)),
                           dtype=int)

        if self.mapping_from == self.mapping_to:
            # if we map from and to the same space, we can simply use the
            # confusion matrix function of sklearn
            if np.count_nonzero(np.isin(right, self.right_ids)) == 0:
                return confmat
            confmat += confusion_matrix(right.flatten(),
                                        left.flatten(),
                                        labels=self.right_ids)
            return confmat.T.astype(int)

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

    def match_confmat(self, confmat):
        """
        gets metrics from a confmat of size len(left_ids)+1 x len(right_ids)+1
        where 0 in each dimension is a "not in list" category

        returns a matched confmat of size len(right_ids)+1 x len(right_ids)+1
        """
        matched_confmat = np.zeros(
            (len(self.right_ids) + 1, len(self.right_ids) + 1), dtype=int)
        right_id_to_idx = {x: i + 1 for i, x in enumerate(self.right_ids)}
        left_id_to_idx = {x: i + 1 for i, x in enumerate(self.left_ids)}

        if self.left_to_right:
            for left_idx in range(len(self.left_ids) + 1):
                for right_idx in range(len(self.right_ids) + 1):
                    if confmat[left_idx, right_idx] == 86632:
                        print(f"left_idx: {left_idx}, right_idx: {right_idx}")
                    if right_idx == 0:
                        # we ignore parts where there is no label
                        continue
                    if left_idx == 0:
                        # there is a label but no prediction
                        matched_confmat[0, right_idx] += confmat[left_idx,
                                                                 right_idx]
                        continue
                    assert left_idx > 0 and right_idx > 0
                    left_id = self.left_ids[left_idx - 1]
                    right_id = self.right_ids[right_idx - 1]
                    mapped_left = self.mapping[left_id]
                    #print(f"left_id: {left_id}, right_id: {right_id}, mapped_left: {mapped_left}")
                    if mapped_left == -1:
                        # the prediction cannot be matched to a label
                        matched_confmat[0, right_idx] += confmat[left_idx,
                                                                 right_idx]
                    elif mapped_left == -2:
                        # multiple possible labels
                        options = self.left2right[left_id]
                        if right_id in options:
                            matched_confmat[right_idx,
                                            right_idx] += confmat[left_idx,
                                                                  right_idx]
                        else:
                            for option in options:
                                matched_confmat[right_id_to_idx[option],
                                                right_idx] += confmat[
                                                    left_idx, right_idx]
                    else:
                        matched_confmat[right_id_to_idx[mapped_left],
                                        right_idx] += confmat[left_idx,
                                                              right_idx]
        else:
            for left_idx in range(len(self.left_ids) + 1):
                for right_idx in range(len(self.right_ids) + 1):
                    if right_idx == 0:
                        # we ignore parts where there is no label
                        continue
                    right_id = self.right_ids[right_idx - 1]
                    mapped_right = self.mapping[right_id]
                    if mapped_right == -1:
                        # this label has no mapping in the (larger) prediction space
                        # we have no choice but to ignore it
                        continue
                    options = self.right2left[right_id]
                    if left_idx == 0:
                        for option in options:
                            matched_confmat[0, left_id_to_idx[option]] += \
                                confmat[left_idx, right_idx]
                        continue
                    left_id = self.left_ids[left_idx - 1]
                    if left_id in options:
                        matched_confmat[left_idx, left_idx] += \
                            confmat[left_idx, right_idx]
                    else:
                        for option in options:
                            matched_confmat[left_idx, left_id_to_idx[option]] += \
                                confmat[left_idx, right_idx]
        return matched_confmat


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
