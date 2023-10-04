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

NYU40_CLASSES = [
    {
        "id": 0,
        'name': "unlabeled",
        'color': (0, 0, 0)
    },
    {
        'id': 1,
        'name': "wall",
        'color': (174, 199, 232)
    },
    {
        'id': 2,
        'name': "floor",
        'color': (152, 223, 138)
    },
    {
        'id': 3,
        'name': "cabinet",
        'color': (31, 119, 180)
    },
    {
        'id': 4,
        'name': "bed",
        'color': (255, 187, 120)
    },
    {
        'id': 5,
        'name': "chair",
        'color': (188, 189, 34)
    },
    {
        'id': 6,
        'name': "sofa",
        'color': (140, 86, 75)
    },
    {
        'id': 7,
        'name': "table",
        'color': (255, 152, 150)
    },
    {
        'id': 8,
        'name': "door",
        'color': (214, 39, 40)
    },
    {
        'id': 9,
        'name': "window",
        'color': (197, 176, 213)
    },
    {
        'id': 10,
        'name': "bookshelf",
        'color': (148, 103, 189)
    },
    {
        'id': 11,
        'name': "picture",
        'color': (196, 156, 148)
    },
    {
        'id': 12,
        'name': "counter",
        'color': (23, 190, 207)
    },
    {
        'id': 13,
        'name': "blinds",
        'color': (178, 76, 76)
    },
    {
        'id': 14,
        'name': "desk",
        'color': (247, 182, 210)
    },
    {
        'id': 15,
        'name': "shelves",
        'color': (66, 188, 102)
    },
    {
        'id': 16,
        'name': "curtain",
        'color': (219, 219, 141)
    },
    {
        'id': 17,
        'name': "dresser",
        'color': (140, 57, 197)
    },
    {
        'id': 18,
        'name': "pillow",
        'color': (202, 185, 52)
    },
    {
        'id': 19,
        'name': "mirror",
        'color': (51, 176, 203)
    },
    {
        'id': 20,
        'name': "floormat",
        'color': (200, 54, 131)
    },
    {
        'id': 21,
        'name': "clothes",
        'color': (92, 193, 61)
    },
    {
        'id': 22,
        'name': "ceiling",
        'color': (78, 71, 183)
    },
    {
        'id': 23,
        'name': "books",
        'color': (172, 114, 82)
    },
    {
        'id': 24,
        'name': "refrigerator",
        'color': (255, 127, 14)
    },
    {
        'id': 25,
        'name': "television",
        'color': (91, 163, 138)
    },
    {
        'id': 26,
        'name': "paper",
        'color': (153, 98, 156)
    },
    {
        'id': 27,
        'name': "towel",
        'color': (140, 153, 101)
    },
    {
        'id': 28,
        'name': "showercurtain",
        'color': (158, 218, 229)
    },
    {
        'id': 29,
        'name': "box",
        'color': (100, 125, 154)
    },
    {
        'id': 30,
        'name': "whiteboard",
        'color': (178, 127, 135)
    },
    {
        'id': 31,
        'name': "person",
        'color': (120, 185, 128)
    },
    {
        'id': 32,
        'name': "nightstand",
        'color': (146, 111, 194)
    },
    {
        'id': 33,
        'name': "toilet",
        'color': (44, 160, 44)
    },
    {
        'id': 34,
        'name': "sink",
        'color': (112, 128, 144)
    },
    {
        'id': 35,
        'name': "lamp",
        'color': (96, 207, 209)
    },
    {
        'id': 36,
        'name': "bathtub",
        'color': (227, 119, 194)
    },
    {
        'id': 37,
        'name': "bag",
        'color': (213, 92, 176)
    },
    {
        'id': 38,
        'name': "otherstructure",
        'color': (94, 106, 211)
    },
    {
        'id': 39,
        'name': "otherfurniture",
        'color': (82, 84, 163)
    },
    {
        'id': 40,
        'name': "otherprop",
        'color': (100, 85, 144)
    },
]

SCANNET_COLOR_MAP_200 = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (188., 189., 34.),
    3: (152., 223., 138.),
    4: (255., 152., 150.),
    5: (214., 39., 40.),
    6: (91., 135., 229.),
    7: (31., 119., 180.),
    8: (229., 91., 104.),
    9: (247., 182., 210.),
    10: (91., 229., 110.),
    11: (255., 187., 120.),
    13: (141., 91., 229.),
    14: (112., 128., 144.),
    15: (196., 156., 148.),
    16: (197., 176., 213.),
    17: (44., 160., 44.),
    18: (148., 103., 189.),
    19: (229., 91., 223.),
    21: (219., 219., 141.),
    22: (192., 229., 91.),
    23: (88., 218., 137.),
    24: (58., 98., 137.),
    26: (177., 82., 239.),
    27: (255., 127., 14.),
    28: (237., 204., 37.),
    29: (41., 206., 32.),
    31: (62., 143., 148.),
    32: (34., 14., 130.),
    33: (143., 45., 115.),
    34: (137., 63., 14.),
    35: (23., 190., 207.),
    36: (16., 212., 139.),
    38: (90., 119., 201.),
    39: (125., 30., 141.),
    40: (150., 53., 56.),
    41: (186., 197., 62.),
    42: (227., 119., 194.),
    44: (38., 100., 128.),
    45: (120., 31., 243.),
    46: (154., 59., 103.),
    47: (169., 137., 78.),
    48: (143., 245., 111.),
    49: (37., 230., 205.),
    50: (14., 16., 155.),
    51: (196., 51., 182.),
    52: (237., 80., 38.),
    54: (138., 175., 62.),
    55: (158., 218., 229.),
    56: (38., 96., 167.),
    57: (190., 77., 246.),
    58: (208., 49., 84.),
    59: (208., 193., 72.),
    62: (55., 220., 57.),
    63: (10., 125., 140.),
    64: (76., 38., 202.),
    65: (191., 28., 135.),
    66: (211., 120., 42.),
    67: (118., 174., 76.),
    68: (17., 242., 171.),
    69: (20., 65., 247.),
    70: (208., 61., 222.),
    71: (162., 62., 60.),
    72: (210., 235., 62.),
    73: (45., 152., 72.),
    74: (35., 107., 149.),
    75: (160., 89., 237.),
    76: (227., 56., 125.),
    77: (169., 143., 81.),
    78: (42., 143., 20.),
    79: (25., 160., 151.),
    80: (82., 75., 227.),
    82: (253., 59., 222.),
    84: (240., 130., 89.),
    86: (123., 172., 47.),
    87: (71., 194., 133.),
    88: (24., 94., 205.),
    89: (134., 16., 179.),
    90: (159., 32., 52.),
    93: (213., 208., 88.),
    95: (64., 158., 70.),
    96: (18., 163., 194.),
    97: (65., 29., 153.),
    98: (177., 10., 109.),
    99: (152., 83., 7.),
    100: (83., 175., 30.),
    101: (18., 199., 153.),
    102: (61., 81., 208.),
    103: (213., 85., 216.),
    104: (170., 53., 42.),
    105: (161., 192., 38.),
    106: (23., 241., 91.),
    107: (12., 103., 170.),
    110: (151., 41., 245.),
    112: (133., 51., 80.),
    115: (184., 162., 91.),
    116: (50., 138., 38.),
    118: (31., 237., 236.),
    120: (39., 19., 208.),
    121: (223., 27., 180.),
    122: (254., 141., 85.),
    125: (97., 144., 39.),
    128: (106., 231., 176.),
    130: (12., 61., 162.),
    131: (124., 66., 140.),
    132: (137., 66., 73.),
    134: (250., 253., 26.),
    136: (55., 191., 73.),
    138: (60., 126., 146.),
    139: (153., 108., 234.),
    140: (184., 58., 125.),
    141: (135., 84., 14.),
    145: (139., 248., 91.),
    148: (53., 200., 172.),
    154: (63., 69., 134.),
    155: (190., 75., 186.),
    156: (127., 63., 52.),
    157: (141., 182., 25.),
    159: (56., 144., 89.),
    161: (64., 160., 250.),
    163: (182., 86., 245.),
    165: (139., 18., 53.),
    166: (134., 120., 54.),
    168: (49., 165., 42.),
    169: (51., 128., 133.),
    170: (44., 21., 163.),
    177: (232., 93., 193.),
    180: (176., 102., 54.),
    185: (116., 217., 17.),
    188: (54., 209., 150.),
    191: (60., 99., 204.),
    193: (129., 43., 144.),
    195: (252., 100., 106.),
    202: (187., 196., 73.),
    208: (13., 158., 40.),
    213: (52., 122., 152.),
    214: (128., 76., 202.),
    221: (187., 50., 115.),
    229: (180., 141., 71.),
    230: (77., 208., 35.),
    232: (72., 183., 168.),
    233: (97., 99., 203.),
    242: (172., 22., 158.),
    250: (155., 64., 40.),
    261: (118., 159., 30.),
    264: (69., 252., 148.),
    276: (45., 103., 173.),
    283: (111., 38., 149.),
    286: (184., 9., 49.),
    300: (188., 174., 67.),
    304: (53., 206., 53.),
    312: (97., 235., 252.),
    323: (66., 32., 182.),
    325: (236., 114., 195.),
    331: (241., 154., 83.),
    342: (133., 240., 52.),
    356: (16., 205., 144.),
    370: (75., 101., 198.),
    392: (237., 95., 251.),
    395: (191., 52., 49.),
    399: (227., 254., 54.),
    408: (49., 206., 87.),
    417: (48., 113., 150.),
    488: (125., 73., 182.),
    540: (229., 32., 114.),
    562: (158., 119., 28.),
    570: (60., 205., 27.),
    572: (18., 215., 201.),
    581: (79., 76., 153.),
    609: (134., 13., 116.),
    748: (192., 97., 63.),
    776: (108., 163., 18.),
    1156: (95., 220., 156.),
    1163: (98., 141., 208.),
    1164: (144., 19., 193.),
    1165: (166., 36., 57.),
    1166: (212., 202., 34.),
    1167: (23., 206., 34.),
    1168: (91., 211., 236.),
    1169: (79., 55., 137.),
    1170: (182., 19., 117.),
    1171: (134., 76., 14.),
    1172: (87., 185., 28.),
    1173: (82., 224., 187.),
    1174: (92., 110., 214.),
    1175: (168., 80., 171.),
    1176: (197., 63., 51.),
    1178: (175., 199., 77.),
    1179: (62., 180., 98.),
    1180: (8., 91., 150.),
    1181: (77., 15., 130.),
    1182: (154., 65., 96.),
    1183: (197., 152., 11.),
    1184: (59., 155., 45.),
    1185: (12., 147., 145.),
    1186: (54., 35., 219.),
    1187: (210., 73., 181.),
    1188: (221., 124., 77.),
    1189: (149., 214., 66.),
    1190: (72., 185., 134.),
    1191: (42., 94., 198.),
}


def get_ade150():
    return ADE150


def get_replica():
    return REPLICA


def get_nyu40():
    return NYU40_CLASSES


def get_scannet_all():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'labelmaker/mappings/label_mapping.csv')
    data = []
    for row in table.index:
        data.append({
            'id':
            int(table.loc[row, 'id']),
            'name':
            table.loc[row, 'category'],
            'raw':
            table.loc[row, 'raw_category'],
            'color': [int(x) for x in table.loc[row, 'color'].split('-')]
        })
    return data


def get_wordnet_by_occurance():
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'labelmaker/mappings/label_mapping.csv')
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
    data = [{'id': 0, 'name': 'unknown', 'color': [0, 0, 0]}]
    for name in sorted(categories, key=lambda x: counts[x], reverse=True):
        if counts[name] > 3:
            # this selects 199 categories
            data.append({'id': len(data), 'name': name})
    for category in data:
        if category['name'] == 'unknown':
            continue
        row = table[table['wnsynsetkey'] == category['name']].index[0]
        category['color'] = [
            int(x) for x in table.loc[row, 'color'].split('-')
        ]
    return data


def get_wordnet(label_key='wn199-merged-v2'):
    table = pd.read_csv(
        Path(os.path.dirname(os.path.realpath(__file__))) / '..' /
        'labelmaker/mappings/label_mapping.csv')
    ids_found = []
    data = [{'id': 0, 'name': 'unknown', 'color': [0, 0, 0]}]
    for row in table.index:
        if table[label_key].isnull()[row]:
            continue
        if table.loc[row, label_key] in ids_found:
            continue
        ids_found.append(table.loc[row, label_key])
        data.append({
            'id':
            int(table.loc[row, label_key]),
            'name':
            table.loc[row, 'wnsynsetkey'],
            'color': [int(x) for x in table.loc[row, 'color'].split('-')]
        })
    return data
