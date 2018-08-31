# import os


BATCH_SIZE = 1
EPOCH = 1

LR = 0.001
DECAY_RATE = 0.8
DECAY_STEP = 300000
T = 2
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1000000
REG_WEIGHT = 1e-5


TRAINING = True

# CONTENT_IMG_DIR = '../../../Curricula/创新实践/大三下/train2014/'
CONTENT_IMG_DIR = 'trainset_test/'
HEIGHT = 224  # image height
WIDTH = 224  # image width
IMG_SIZE = [HEIGHT, WIDTH]  #image size

STYLE_IMG_HEIGHT = 224
STYLE_IMG_WIDTH = 224

STYLE_IMG_DIR = 'style_img'
MODEL_WEIGHT_DIR = 'weights_test'

STYLE_DICT = [
    {'name': 'style_0', 'load_path': "style/candy.jpg", "style_weight": 100, "content_weight": 1, "tv_weight": 0, "save_dir": "save/candy/" },
    {'name': 'style_1', 'load_path': "style/colortree.jpg", "style_weight": 200, "content_weight": 1, "tv_weight": 0, "save_dir": "save/colortree/" },
    {'name': 'style_2', 'load_path': "style/countryside.jpg", "style_weight": 1000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/countryside/" },
    {'name': 'style_3', 'load_path': "style/cubist.jpg", "style_weight": 10000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/cubist/" },
    {'name': 'style_4', 'load_path': "style/landscape.jpg", "style_weight": 100000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/landscape/" },
    {'name': 'style_5', 'load_path': "style/scream.jpg", "style_weight": 1000000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/scream/" },
    {'name': 'style_6', 'load_path': "style/starry.jpg", "style_weight": 10000000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/starry/" },
    {'name': 'style_7', 'load_path': "style/udnie.jpg", "style_weight": 500, "content_weight": 1, "tv_weight": 0, "save_dir": "save/udnie/" },
    {'name': 'style_8', 'load_path': "style/wave.jpg", "style_weight": 2000, "content_weight": 1, "tv_weight": 0, "save_dir": "save/wave/" }
            ]
STYLE_NUM = len(STYLE_DICT)

# BANK_WEIGHT_DIR = os.path.join(MODEL_WEIGHT_DIR, 'bank')
# BANK_WEIGHT_PATH = os.path.join(BANK_WEIGHT_DIR, '{}.pth')
# MODEL_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'model.pth')
# ENCODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'encoder.pth')
# DECODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'decoder.pth')
# GLOBAL_STEP_PATH = os.path.join(MODEL_WEIGHT_DIR, 'global_step.log')

JUDGE_NET = 'pre-trained VGG-16'
VGG16_PATH = './model/vgg_16.ckpt'
VGG19_PATH = './model/imagenet-vgg-verydeep-19.mat'
LOG_DIR = './log/'
IMG_SAVE_DIR = './save'
CONTENT_LAYER = ('relu4_2',)
STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_2', 'relu4_2')

MEAN_PIXEL = [123.68, 116.779, 103.939]

RESIZE_SIDE_MIN = 256
RESIZE_SIDE_MAX = 512
K = 1000
MAX_ITERATION = 300 * K
ADJUST_LR_ITER = 10 * K
LOG_ITER = 1 * K