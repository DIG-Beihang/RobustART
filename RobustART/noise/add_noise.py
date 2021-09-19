
from RobustART.noise.utils.add_noise_utils import noise_list, default_config, function_dict


class AddNoise(object):
    """
    Add Noise for one image
    Support List: noise_list = ['imagenet-s', 'imagenet-c', 'pgd_linf', 'pgd_l2', 'fgsm', 'autoattack_linf', 'mim_linf', 'pgd_l1'],
    you should choose a noise type when init
    """
    def __init__(self, noise_type):
        self.noise_type = noise_type
        self.config = default_config[self.noise_type]
        assert self.noise_type in noise_list, f'Add noise only support for {noise_list}'

    def set_config(self, **kwargs):
        """
        Every Noise has a default config dict, you can use this method to set config
        :param kwargs: dict of config to set
        """
        assert set(kwargs.keys()) & set(self.config.keys()) == set(kwargs.keys()), \
           f'Key Error! Unexpect Keys {set(kwargs.keys()) - set(self.config.keys())}'

        self.config.update(kwargs)
        print(f'Config for {self.noise_type} Noise')
        print(self.config)

    def add_noise(self, image, label=None):
        """
        :param label: Provide the label when add adv noise
        :param image: The file path of one image. Or a (n,w,h,3) numpy array of a batch of image
        :return: If the input is a file path, return a (w,h,3) numpy array of this image after
        adding noise of specific noise_type
        Else return (n,w,h,3) numpy array batch of image
        """
        if isinstance(image, str):
            assert self.noise_type not in ['imagenet-s', 'imagenet-c'], 'Only imagenet-s and imagenet-c support image' \
                                                                        'path input'
        if self.noise_type in ['imagenet-s', 'imagenet-c']:
            return function_dict[self.noise_type](image, **self.config)
        else:
            return function_dict[self.noise_type](image, label, **self.config)
