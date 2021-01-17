from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from matplotlib import pyplot
from numpy import expand_dims

# define a configuration for the model
class LicensePlateConfig(Config):
	# define the name of the configuration
	NAME = "LicensePlate_cfg"
	# number of classes (background + License Plate)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
# prepare config
config = LicensePlateConfig()

# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=config)

model.load_weights("/content/drive/MyDrive/LicensePlateProject/licenseplate_cfg20200509T2207/mask_rcnn_licenseplate_cfg_0031.h5", by_name=True)

def detectPlate(imagePath):
    #Detecting License Plate
    #image = pyplot.imread(SUBMISSION_IMAGES+str(imageNum)+".jpg")
    image = pyplot.imread(imagePath)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, config)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

    return yhat, image