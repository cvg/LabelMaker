import pandas as pd
from pathlib import Path
import os
import numpy as np

ADE150 = [{
    'id': 0,
    'name': 'wall',
    'color': [120, 120, 120]
}, {
    'id': 1,
    'name': 'building',
    'color': [180, 120, 120]
}, {
    'id': 2,
    'name': 'sky',
    'color': [6, 230, 230]
}, {
    'id': 3,
    'name': 'floor',
    'color': [80, 50, 50]
}, {
    'id': 4,
    'name': 'tree',
    'color': [4, 200, 3]
}, {
    'id': 5,
    'name': 'ceiling',
    'color': [120, 120, 80]
}, {
    'id': 6,
    'name': 'road',
    'color': [140, 140, 140]
}, {
    'id': 7,
    'name': 'bed',
    'color': [204, 5, 255]
}, {
    'id': 8,
    'name': 'windowpane',
    'color': [230, 230, 230]
}, {
    'id': 9,
    'name': 'grass',
    'color': [4, 250, 7]
}, {
    'id': 10,
    'name': 'cabinet',
    'color': [224, 5, 255]
}, {
    'id': 11,
    'name': 'sidewalk',
    'color': [235, 255, 7]
}, {
    'id': 12,
    'name': 'person',
    'color': [150, 5, 61]
}, {
    'id': 13,
    'name': 'earth',
    'color': [120, 120, 70]
}, {
    'id': 14,
    'name': 'door',
    'color': [8, 255, 51]
}, {
    'id': 15,
    'name': 'table',
    'color': [255, 6, 82]
}, {
    'id': 16,
    'name': 'mountain',
    'color': [143, 255, 140]
}, {
    'id': 17,
    'name': 'plant',
    'color': [204, 255, 4]
}, {
    'id': 18,
    'name': 'curtain',
    'color': [255, 51, 7]
}, {
    'id': 19,
    'name': 'chair',
    'color': [204, 70, 3]
}, {
    'id': 20,
    'name': 'car',
    'color': [0, 102, 200]
}, {
    'id': 21,
    'name': 'water',
    'color': [61, 230, 250]
}, {
    'id': 22,
    'name': 'painting',
    'color': [255, 6, 51]
}, {
    'id': 23,
    'name': 'sofa',
    'color': [11, 102, 255]
}, {
    'id': 24,
    'name': 'shelf',
    'color': [255, 7, 71]
}, {
    'id': 25,
    'name': 'house',
    'color': [255, 9, 224]
}, {
    'id': 26,
    'name': 'sea',
    'color': [9, 7, 230]
}, {
    'id': 27,
    'name': 'mirror',
    'color': [220, 220, 220]
}, {
    'id': 28,
    'name': 'rug',
    'color': [255, 9, 92]
}, {
    'id': 29,
    'name': 'field',
    'color': [112, 9, 255]
}, {
    'id': 30,
    'name': 'armchair',
    'color': [8, 255, 214]
}, {
    'id': 31,
    'name': 'seat',
    'color': [7, 255, 224]
}, {
    'id': 32,
    'name': 'fence',
    'color': [255, 184, 6]
}, {
    'id': 33,
    'name': 'desk',
    'color': [10, 255, 71]
}, {
    'id': 34,
    'name': 'rock',
    'color': [255, 41, 10]
}, {
    'id': 35,
    'name': 'wardrobe',
    'color': [7, 255, 255]
}, {
    'id': 36,
    'name': 'lamp',
    'color': [224, 255, 8]
}, {
    'id': 37,
    'name': 'bathtub',
    'color': [102, 8, 255]
}, {
    'id': 38,
    'name': 'railing',
    'color': [255, 61, 6]
}, {
    'id': 39,
    'name': 'cushion',
    'color': [255, 194, 7]
}, {
    'id': 40,
    'name': 'base',
    'color': [255, 122, 8]
}, {
    'id': 41,
    'name': 'box',
    'color': [0, 255, 20]
}, {
    'id': 42,
    'name': 'column',
    'color': [255, 8, 41]
}, {
    'id': 43,
    'name': 'signboard',
    'color': [255, 5, 153]
}, {
    'id': 44,
    'name': 'chest of drawers',
    'color': [6, 51, 255]
}, {
    'id': 45,
    'name': 'counter',
    'color': [235, 12, 255]
}, {
    'id': 46,
    'name': 'sand',
    'color': [160, 150, 20]
}, {
    'id': 47,
    'name': 'sink',
    'color': [0, 163, 255]
}, {
    'id': 48,
    'name': 'skyscraper',
    'color': [140, 140, 140]
}, {
    'id': 49,
    'name': 'fireplace',
    'color': [250, 10, 15]
}, {
    'id': 50,
    'name': 'refrigerator',
    'color': [20, 255, 0]
}, {
    'id': 51,
    'name': 'grandstand',
    'color': [31, 255, 0]
}, {
    'id': 52,
    'name': 'path',
    'color': [255, 31, 0]
}, {
    'id': 53,
    'name': 'stairs',
    'color': [255, 224, 0]
}, {
    'id': 54,
    'name': 'runway',
    'color': [153, 255, 0]
}, {
    'id': 55,
    'name': 'case',
    'color': [0, 0, 255]
}, {
    'id': 56,
    'name': 'pool table',
    'color': [255, 71, 0]
}, {
    'id': 57,
    'name': 'pillow',
    'color': [0, 235, 255]
}, {
    'id': 58,
    'name': 'screen door',
    'color': [0, 173, 255]
}, {
    'id': 59,
    'name': 'stairway',
    'color': [31, 0, 255]
}, {
    'id': 60,
    'name': 'river',
    'color': [11, 200, 200]
}, {
    'id': 61,
    'name': 'bridge',
    'color': [255, 82, 0]
}, {
    'id': 62,
    'name': 'bookcase',
    'color': [0, 255, 245]
}, {
    'id': 63,
    'name': 'blind',
    'color': [0, 61, 255]
}, {
    'id': 64,
    'name': 'coffee table',
    'color': [0, 255, 112]
}, {
    'id': 65,
    'name': 'toilet',
    'color': [0, 255, 133]
}, {
    'id': 66,
    'name': 'flower',
    'color': [255, 0, 0]
}, {
    'id': 67,
    'name': 'book',
    'color': [255, 163, 0]
}, {
    'id': 68,
    'name': 'hill',
    'color': [255, 102, 0]
}, {
    'id': 69,
    'name': 'bench',
    'color': [194, 255, 0]
}, {
    'id': 70,
    'name': 'countertop',
    'color': [0, 143, 255]
}, {
    'id': 71,
    'name': 'stove',
    'color': [51, 255, 0]
}, {
    'id': 72,
    'name': 'palm',
    'color': [0, 82, 255]
}, {
    'id': 73,
    'name': 'kitchen island',
    'color': [0, 255, 41]
}, {
    'id': 74,
    'name': 'computer',
    'color': [0, 255, 173]
}, {
    'id': 75,
    'name': 'swivel chair',
    'color': [10, 0, 255]
}, {
    'id': 76,
    'name': 'boat',
    'color': [173, 255, 0]
}, {
    'id': 77,
    'name': 'bar',
    'color': [0, 255, 153]
}, {
    'id': 78,
    'name': 'arcade machine',
    'color': [255, 92, 0]
}, {
    'id': 79,
    'name': 'hovel',
    'color': [255, 0, 255]
}, {
    'id': 80,
    'name': 'bus',
    'color': [255, 0, 245]
}, {
    'id': 81,
    'name': 'towel',
    'color': [255, 0, 102]
}, {
    'id': 82,
    'name': 'light',
    'color': [255, 173, 0]
}, {
    'id': 83,
    'name': 'truck',
    'color': [255, 0, 20]
}, {
    'id': 84,
    'name': 'tower',
    'color': [255, 184, 184]
}, {
    'id': 85,
    'name': 'chandelier',
    'color': [0, 31, 255]
}, {
    'id': 86,
    'name': 'awning',
    'color': [0, 255, 61]
}, {
    'id': 87,
    'name': 'streetlight',
    'color': [0, 71, 255]
}, {
    'id': 88,
    'name': 'booth',
    'color': [255, 0, 204]
}, {
    'id': 89,
    'name': 'television receiver',
    'color': [0, 255, 194]
}, {
    'id': 90,
    'name': 'airplane',
    'color': [0, 255, 82]
}, {
    'id': 91,
    'name': 'dirt track',
    'color': [0, 10, 255]
}, {
    'id': 92,
    'name': 'apparel',
    'color': [0, 112, 255]
}, {
    'id': 93,
    'name': 'pole',
    'color': [51, 0, 255]
}, {
    'id': 94,
    'name': 'land',
    'color': [0, 194, 255]
}, {
    'id': 95,
    'name': 'bannister',
    'color': [0, 122, 255]
}, {
    'id': 96,
    'name': 'escalator',
    'color': [0, 255, 163]
}, {
    'id': 97,
    'name': 'ottoman',
    'color': [255, 153, 0]
}, {
    'id': 98,
    'name': 'bottle',
    'color': [0, 255, 10]
}, {
    'id': 99,
    'name': 'buffet',
    'color': [255, 112, 0]
}, {
    'id': 100,
    'name': 'poster',
    'color': [143, 255, 0]
}, {
    'id': 101,
    'name': 'stage',
    'color': [82, 0, 255]
}, {
    'id': 102,
    'name': 'van',
    'color': [163, 255, 0]
}, {
    'id': 103,
    'name': 'ship',
    'color': [255, 235, 0]
}, {
    'id': 104,
    'name': 'fountain',
    'color': [8, 184, 170]
}, {
    'id': 105,
    'name': 'conveyer belt',
    'color': [133, 0, 255]
}, {
    'id': 106,
    'name': 'canopy',
    'color': [0, 255, 92]
}, {
    'id': 107,
    'name': 'washer',
    'color': [184, 0, 255]
}, {
    'id': 108,
    'name': 'plaything',
    'color': [255, 0, 31]
}, {
    'id': 109,
    'name': 'swimming pool',
    'color': [0, 184, 255]
}, {
    'id': 110,
    'name': 'stool',
    'color': [0, 214, 255]
}, {
    'id': 111,
    'name': 'barrel',
    'color': [255, 0, 112]
}, {
    'id': 112,
    'name': 'basket',
    'color': [92, 255, 0]
}, {
    'id': 113,
    'name': 'waterfall',
    'color': [0, 224, 255]
}, {
    'id': 114,
    'name': 'tent',
    'color': [112, 224, 255]
}, {
    'id': 115,
    'name': 'bag',
    'color': [70, 184, 160]
}, {
    'id': 116,
    'name': 'minibike',
    'color': [163, 0, 255]
}, {
    'id': 117,
    'name': 'cradle',
    'color': [153, 0, 255]
}, {
    'id': 118,
    'name': 'oven',
    'color': [71, 255, 0]
}, {
    'id': 119,
    'name': 'ball',
    'color': [255, 0, 163]
}, {
    'id': 120,
    'name': 'food',
    'color': [255, 204, 0]
}, {
    'id': 121,
    'name': 'step',
    'color': [255, 0, 143]
}, {
    'id': 122,
    'name': 'tank',
    'color': [0, 255, 235]
}, {
    'id': 123,
    'name': 'trade name',
    'color': [133, 255, 0]
}, {
    'id': 124,
    'name': 'microwave',
    'color': [255, 0, 235]
}, {
    'id': 125,
    'name': 'pot',
    'color': [245, 0, 255]
}, {
    'id': 126,
    'name': 'animal',
    'color': [255, 0, 122]
}, {
    'id': 127,
    'name': 'bicycle',
    'color': [255, 245, 0]
}, {
    'id': 128,
    'name': 'lake',
    'color': [10, 190, 212]
}, {
    'id': 129,
    'name': 'dishwasher',
    'color': [214, 255, 0]
}, {
    'id': 130,
    'name': 'screen',
    'color': [0, 204, 255]
}, {
    'id': 131,
    'name': 'blanket',
    'color': [20, 0, 255]
}, {
    'id': 132,
    'name': 'sculpture',
    'color': [255, 255, 0]
}, {
    'id': 133,
    'name': 'hood',
    'color': [0, 153, 255]
}, {
    'id': 134,
    'name': 'sconce',
    'color': [0, 41, 255]
}, {
    'id': 135,
    'name': 'vase',
    'color': [0, 255, 204]
}, {
    'id': 136,
    'name': 'traffic light',
    'color': [41, 0, 255]
}, {
    'id': 137,
    'name': 'tray',
    'color': [41, 255, 0]
}, {
    'id': 138,
    'name': 'ashcan',
    'color': [173, 0, 255]
}, {
    'id': 139,
    'name': 'fan',
    'color': [0, 245, 255]
}, {
    'id': 140,
    'name': 'pier',
    'color': [71, 0, 255]
}, {
    'id': 141,
    'name': 'crt screen',
    'color': [122, 0, 255]
}, {
    'id': 142,
    'name': 'plate',
    'color': [0, 255, 184]
}, {
    'id': 143,
    'name': 'monitor',
    'color': [0, 92, 255]
}, {
    'id': 144,
    'name': 'bulletin board',
    'color': [184, 255, 0]
}, {
    'id': 145,
    'name': 'shower',
    'color': [0, 133, 255]
}, {
    'id': 146,
    'name': 'radiator',
    'color': [255, 214, 0]
}, {
    'id': 147,
    'name': 'glass',
    'color': [25, 194, 194]
}, {
    'id': 148,
    'name': 'clock',
    'color': [102, 255, 0]
}, {
    'id': 149,
    'name': 'flag',
    'color': [92, 0, 255]
}]

REPLICA = [{
    'id': 1,
    'name': 'backpack'
}, {
    'id': 2,
    'name': 'base-cabinet'
}, {
    'id': 3,
    'name': 'basket'
}, {
    'id': 4,
    'name': 'bathtub'
}, {
    'id': 5,
    'name': 'beam'
}, {
    'id': 6,
    'name': 'beanbag'
}, {
    'id': 7,
    'name': 'bed'
}, {
    'id': 8,
    'name': 'bench'
}, {
    'id': 9,
    'name': 'bike'
}, {
    'id': 10,
    'name': 'bin'
}, {
    'id': 11,
    'name': 'blanket'
}, {
    'id': 12,
    'name': 'blinds'
}, {
    'id': 13,
    'name': 'book'
}, {
    'id': 14,
    'name': 'bottle'
}, {
    'id': 15,
    'name': 'box'
}, {
    'id': 16,
    'name': 'bowl'
}, {
    'id': 17,
    'name': 'camera'
}, {
    'id': 18,
    'name': 'cabinet'
}, {
    'id': 19,
    'name': 'candle'
}, {
    'id': 20,
    'name': 'chair'
}, {
    'id': 21,
    'name': 'chopping-board'
}, {
    'id': 22,
    'name': 'clock'
}, {
    'id': 23,
    'name': 'cloth'
}, {
    'id': 24,
    'name': 'clothing'
}, {
    'id': 25,
    'name': 'coaster'
}, {
    'id': 26,
    'name': 'comforter'
}, {
    'id': 27,
    'name': 'computer-keyboard'
}, {
    'id': 28,
    'name': 'cup'
}, {
    'id': 29,
    'name': 'cushion'
}, {
    'id': 30,
    'name': 'curtain'
}, {
    'id': 31,
    'name': 'ceiling'
}, {
    'id': 32,
    'name': 'cooktop'
}, {
    'id': 33,
    'name': 'countertop'
}, {
    'id': 34,
    'name': 'desk'
}, {
    'id': 35,
    'name': 'desk-organizer'
}, {
    'id': 36,
    'name': 'desktop-computer'
}, {
    'id': 37,
    'name': 'door'
}, {
    'id': 38,
    'name': 'exercise-ball'
}, {
    'id': 39,
    'name': 'faucet'
}, {
    'id': 40,
    'name': 'floor'
}, {
    'id': 41,
    'name': 'handbag'
}, {
    'id': 42,
    'name': 'hair-dryer'
}, {
    'id': 43,
    'name': 'handrail'
}, {
    'id': 44,
    'name': 'indoor-plant'
}, {
    'id': 45,
    'name': 'knife-block'
}, {
    'id': 46,
    'name': 'kitchen-utensil'
}, {
    'id': 47,
    'name': 'lamp'
}, {
    'id': 48,
    'name': 'laptop'
}, {
    'id': 49,
    'name': 'major-appliance'
}, {
    'id': 50,
    'name': 'mat'
}, {
    'id': 51,
    'name': 'microwave'
}, {
    'id': 52,
    'name': 'monitor'
}, {
    'id': 53,
    'name': 'mouse'
}, {
    'id': 54,
    'name': 'nightstand'
}, {
    'id': 55,
    'name': 'pan'
}, {
    'id': 56,
    'name': 'panel'
}, {
    'id': 57,
    'name': 'paper-towel'
}, {
    'id': 58,
    'name': 'phone'
}, {
    'id': 59,
    'name': 'picture'
}, {
    'id': 60,
    'name': 'pillar'
}, {
    'id': 61,
    'name': 'pillow'
}, {
    'id': 62,
    'name': 'pipe'
}, {
    'id': 63,
    'name': 'plant-stand'
}, {
    'id': 64,
    'name': 'plate'
}, {
    'id': 65,
    'name': 'pot'
}, {
    'id': 66,
    'name': 'rack'
}, {
    'id': 67,
    'name': 'refrigerator'
}, {
    'id': 68,
    'name': 'remote-control'
}, {
    'id': 69,
    'name': 'scarf'
}, {
    'id': 70,
    'name': 'sculpture'
}, {
    'id': 71,
    'name': 'shelf'
}, {
    'id': 72,
    'name': 'shoe'
}, {
    'id': 73,
    'name': 'shower-stall'
}, {
    'id': 74,
    'name': 'sink'
}, {
    'id': 75,
    'name': 'small-appliance'
}, {
    'id': 76,
    'name': 'sofa'
}, {
    'id': 77,
    'name': 'stair'
}, {
    'id': 78,
    'name': 'stool'
}, {
    'id': 79,
    'name': 'switch'
}, {
    'id': 80,
    'name': 'table'
}, {
    'id': 81,
    'name': 'table-runner'
}, {
    'id': 82,
    'name': 'tablet'
}, {
    'id': 83,
    'name': 'tissue-paper'
}, {
    'id': 84,
    'name': 'toilet'
}, {
    'id': 85,
    'name': 'toothbrush'
}, {
    'id': 86,
    'name': 'towel'
}, {
    'id': 87,
    'name': 'tv-screen'
}, {
    'id': 88,
    'name': 'tv-stand'
}, {
    'id': 89,
    'name': 'umbrella'
}, {
    'id': 90,
    'name': 'utensil-holder'
}, {
    'id': 91,
    'name': 'vase'
}, {
    'id': 92,
    'name': 'vent'
}, {
    'id': 93,
    'name': 'wall'
}, {
    'id': 94,
    'name': 'wall-cabinet'
}, {
    'id': 95,
    'name': 'wall-plug'
}, {
    'id': 96,
    'name': 'wardrobe'
}, {
    'id': 97,
    'name': 'window'
}, {
    'id': 98,
    'name': 'rug'
}, {
    'id': 99,
    'name': 'logo'
}, {
    'id': 100,
    'name': 'bag'
}, {
    'id': 101,
    'name': 'set-of-clothing'
}]


def get_ade150():
    return ADE150


def get_replica():
    return REPLICA


def get_nyu40():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    data = []
    for row in table.index:
        if table['nyu40id'].isnull()[row]:
            continue
        nyu40_id = table.loc[row, 'nyu40id']
        try:
            next(x for x in data if x['id'] == nyu40_id)
            # if this passes, id already exists
        except StopIteration:
            data.append({'id': nyu40_id, 'name': table.loc[row, 'nyu40class']})
    return data


def get_scannet_all():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    data = []
    for row in table.index:
        data.append({
            'id': table.loc[row, 'id'],
            'name': table.loc[row, 'category']
        })
    return data


def get_wordnet():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'label_mapping.csv')
    categories = sorted(
        list(
            set(table['wnsynsetkey'][i] for i in table.index
                if not table['wnsynsetkey'].isnull()[i])))
    # now count the occurance of the categories
    counts = {x: 0 for x in categories}
    for row in table.index:
        if table['wnsynsetkey'].isnull()[row]:
            continue
        counts[table['wnsynsetkey'][row]] += table['count'][row]
    data = [{'id': 0, 'name': 'unknown'}]
    for name in sorted(categories, key=lambda x: counts[x], reverse=True):
        if counts[name] > 3:
            # this selects 199 categories
            data.append({'id': len(data), 'name': name})
    return data
