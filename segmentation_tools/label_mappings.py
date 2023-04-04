import pandas as pd
import numpy as np
import os
from pathlib import Path
from segmentation_tools.label_data import ADE150


def set_ids_according_to_names():
    table_path = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'label_mapping.csv')
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
    table.to_csv(table_path)

class MatcherScannetADE150:
    def __init__(self):
        table = pd.read_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'label_mapping.csv')
        mapping_from_scannet = -1 * np.ones(table['id'].max() + 1, dtype=int)
        mapping_multiples = {}
        for row in table.index:
            if table['ade20k'].isnull()[row]:
                continue
            scannet_id = table.loc[row, 'id']
            ade150_id = table.loc[row, 'ade20k']
            try:
                ade150_id = int(ade150_id)
                mapping_from_scannet[scannet_id] = ade150_id
            except ValueError as e:
                mapping_from_scannet[scannet_id] = -2
                mapping_multiples[scannet_id] = [int(x) for x in ade150_id.split(',')]
        self.mapping_from_scannet = mapping_from_scannet
        self.mapping_multiples = mapping_multiples

    def match(self, scannet, ade150):
        ade150 = ade150.astype(int)
        scannet = scannet.astype(int)
        ade150_from_scannet = self.mapping_from_scannet[scannet]
        matching = np.equal(ade150_from_scannet, ade150).astype(int)
        matching[ade150_from_scannet == -1] = -1
        if np.sum(ade150_from_scannet == -2) > 0:
            # check more complicated cases
            for scannet_id, ade150_ids in self.mapping_multiples.items():
                for ade150_id in ade150_ids:
                    matching[np.logical_and(scannet == scannet_id, ade150 == ade150_id)] = 1
        return matching


class MatcherScannetNYU40:
    def __init__(self):
        table = pd.read_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'label_mapping.csv')
        mapping_from_scannet = -1 * np.ones(table['id'].max() + 1, dtype=int)
        for row in table.index:
            if table['nyu40id'].isnull()[row]:
                continue
            scannet_id = table.loc[row, 'id']
            nyu40_id = table.loc[row, 'nyu40id']
            mapping_from_scannet[scannet_id] = int(nyu40_id)
        self.mapping_from_scannet = mapping_from_scannet

    def match(self, scannet, nyu40):
        nyu40 = nyu40.astype(int)
        scannet = scannet.astype(int)
        nyu40_from_scannet = self.mapping_from_scannet[scannet]
        matching = np.equal(nyu40_from_scannet, nyu40).astype(int)
        matching[nyu40_from_scannet == -1] = -1
        return matching


class MatcherADE150NYU40:
    def __init__(self):
        table = pd.read_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'label_mapping.csv')
        mapping_from_ade150 = -1 * np.ones(151, dtype=int)
        mapping_multiples = {}
        for row in table.index:
            if table['nyu40id'].isnull()[row]:
                continue
            if table['ade20k'].isnull()[row]:
                continue
            ade150_ids = [int(x) for x in table.loc[row, 'ade20k'].split(',')]
            nyu40_id = table.loc[row, 'nyu40id']
            for ade150_id in ade150_ids:
                mapping_multiples.setdefault(ade150_id, set())
                mapping_multiples[ade150_id].add(int(nyu40_id))
                mapping_from_ade150[ade150_id] = -2
        # now check for unique mappings
        to_delete = []
        for ade150_id, nyu40_ids in mapping_multiples.items():
            if len(nyu40_ids) == 1:
                mapping_from_ade150[ade150_id] = list(nyu40_ids)[0]
                to_delete.append(ade150_id)
        for ade150_id in to_delete:
            del mapping_multiples[ade150_id]
        self.mapping_from_ade150 = mapping_from_ade150
        self.mapping_multiples = mapping_multiples

    def match(self, ade150, nyu40):
        nyu40 = nyu40.astype(int)
        ade150 = ade150.astype(int)
        nyu40_from_ade150 = self.mapping_from_ade150[ade150]
        matching = np.equal(nyu40_from_ade150, nyu40).astype(int)
        matching[nyu40_from_ade150 == -1] = -1
        if np.sum(nyu40_from_ade150 == -2) > 0:
            # check more complicated cases
            for ade150_id, nyu40_ids in self.mapping_multiples.items():
                for nyu40_id in nyu40_ids:
                    matching[np.logical_and(ade150 == ade150_id, nyu40 == nyu40_id)] = 1
        return matching


def match_scannet_ade150(scannet, ade150):
    matcher = MatcherScannetADE150()
    return matcher.match(scannet, ade150)

def match_scannet_nyu40(scannet, nyu40):
    matcher = MatcherScannetNYU40()
    return matcher.match(scannet, nyu40)

def match_ade150_nyu40(ade150, nyu40):
    matcher = MatcherADE150NYU40()
    return matcher.match(ade150, nyu40)

def get_ade150_to_scannet():
    table = pd.read_csv(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'label_mapping.csv')
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
