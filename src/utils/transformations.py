import albumentations as albu

def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # Add more augmentations as needed
    ],additional_targets={'target':'image','thermal_low_res_image':'image'},is_check_shapes=False)
    
def get_training_augmentation_flir():
    return albu.Compose([
        albu.Resize(512,640, p = 1),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # Add more augmentations as needed
    ],additional_targets={'target':'image'},is_check_shapes=False)
    
def get_training_augmentation_cats():
    return albu.Compose([
        albu.Resize(240,320, p = 1),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # Add more augmentations as needed
    ],additional_targets={'target':'image','thermal_low_res_image':'image'},is_check_shapes=False)
    
def get_validation_augmentation_flir():
    return albu.Compose([
        albu.Resize(512,640, p = 1),
        # Add more augmentations as needed
    ],additional_targets={'target':'image'},is_check_shapes=False)
    
def get_validation_augmentation_cats():
    return albu.Compose([
        albu.Resize(240,320, p = 1),
        # Add more augmentations as needed
    ],additional_targets={'target':'image','thermal_low_res_image':'image'},is_check_shapes=False)
    
def resize():
    return albu.Compose([
        albu.Resize(128,160, p = 1),
        # Add more augmentations as needed
    ])
    
def resize_cats():
    return albu.Compose([
        albu.Resize(60,80, p = 1),
        # Add more augmentations as needed
    ])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    