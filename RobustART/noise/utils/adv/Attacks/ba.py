import numpy as np
import torch
from torch.autograd import Variable

from .attack_base import Attack


class BA:
    def __init__(self, model=None, **kwargs):
        """
        @description: The Boundary Attack
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        self.model = model
        self.device = None
        self.isTargeted = None

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description:
        @param {
            epsilon:
            eps_iter:
            num_steps:
        }
        @return: adversarial_samples
        """

        # 扰动的步长系数
        self.epsilon = float(kwargs.get("eps", 0.01))
        # 重新缩放的扰动的尺度
        self.delta = float(kwargs.get("delta", 0.01))
        # 归一化数据的上下边界
        self.lower_bound = float(kwargs.get("lower_bound", 0.0))
        self.upper_bound = float(kwargs.get("upper_bound", 1.0))
        # 扰动样本更新的最大内层迭代次数
        self.max_iter = int(kwargs.get("max_iter", 10))
        # 用来搜索合适的ｅｐｓｉｌｏｎ的迭代次数
        self.binary_search_steps = int(kwargs.get("binary_search_steps", 20))
        # 单次批处理
        self.batch_size = int(kwargs.get("batch_size", 8))
        # 用来调整ｄｅｌｔａ系数的更新系数
        self.step_adapt = float(kwargs.get("step_adapt", 0.9))
        # 过程中生成的潜在扰动样本的采样的数目
        self.sample_size = int(kwargs.get("sample_size", 80))
        # 初始化的随机样本的数目
        self.init_size = int(kwargs.get("init_size", 200))

    # 获得样本之间距离
    def get_diff(self, sample1, sample2):
        return np.linalg.norm((sample1 - sample2).astype(np.float32))

    # 获得高斯噪声的样本
    def gaussian_sample_noise(self, epsilon, imageshape, bounds):
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = np.random.normal(scale=std, size=imageshape)
        noise = noise.astype(np.float32)
        return noise

    # 获得均匀分布的样本
    def unifom_sample_noise(self, epsilon, imageshape, bounds):
        min_, max_ = bounds
        w = epsilon * (max_ - min_)
        noise = np.random.uniform(-w, w, size=imageshape)
        noise = noise.astype(np.float32)
        return noise

    # 计算样本的Ｌ２距离
    def get_dist(self, xs, x2s):

        l2dist = torch.sum((xs - x2s) ** 2, [1, 2, 3])

        return l2dist

    def _perturb(self, x, y, y_p):
        clip_min, clip_max = self.classifier.clip_values

        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, clip_min, clip_max)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with boundary attack
        if self.targeted:
            x_adv = self._attack(
                initial_sample, x, y, self.delta, self.epsilon, clip_min, clip_max
            )
        else:
            x_adv = self._attack(
                initial_sample, x, y_p, self.delta, self.epsilon, clip_min, clip_max
            )

        return 0

    # 初始化随机样本
    def _init_sample(self, x, y, targeted, clip_min, clip_max):
        nprd = np.random.RandomState()
        initial_sample = None

        if targeted:
            # Attack satisfied
            # Attack unsatisfied yet
            for _ in range(self.init_size):
                random_img_numpy = nprd.uniform(
                    clip_min, clip_max, size=x.shape
                ).astype(x.dtype)
                random_img = np.expand_dims(random_img_numpy, axis=0)
                tensor_random_img = Variable(
                    torch.from_numpy(random_img).to(self.device)
                )
                output = self.model(tensor_random_img)
                random_class = torch.argmax(output, 1)
                random_class = random_class.data.cpu().numpy()
                if random_class[0] == y:
                    initial_sample = random_img_numpy
                    break

        else:
            for _ in range(self.init_size):
                # random_img_numpy = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                mean_, std_ = np.mean(x), np.std(x)
                random_img_numpy = nprd.normal(
                    loc=mean_, scale=2 * std_, size=x.shape
                ).astype(x.dtype)
                random_img = np.expand_dims(random_img_numpy, axis=0)
                tensor_random_img = Variable(
                    torch.from_numpy(random_img).to(self.device)
                )
                output = self.model(tensor_random_img)
                random_class = torch.argmax(output, 1)
                random_class = random_class.data.cpu().numpy()
                initial_sample = random_img_numpy
                if random_class[0] != y:
                    initial_sample = random_img_numpy
                    break

        return initial_sample

    # 正交扰动生成
    def _orthogonal_perturb(self, delta, current_sample, original_sample):

        perturb = np.random.randn(
            original_sample.shape[0], original_sample.shape[1], original_sample.shape[2]
        )

        # Rescale the perturbation
        perturb /= np.linalg.norm(perturb)

        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        # Project the perturbation onto sphere

        direction = original_sample - current_sample
        perturb = np.swapaxes(perturb, 0, 0 - 1)
        direction = np.swapaxes(direction, 0, 0 - 1)

        vdot = np.vdot(perturb, direction)
        perturb -= vdot * direction

        perturb = np.swapaxes(perturb, 0, 0 - 1)

        return perturb

    def compare(object1, object2, target_flag):
        return object1 == object2 if target_flag else object1 != object2

    def generate(self, xs=None, ys=None, target_flag=False):
        """
        @description:
        @param {
            xs:
            ys:
            device:
        }
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted

        var_xs, var_ys = Variable(xs.to(device)), Variable(ys.to(device))
        with torch.no_grad():
            outputs = self.model(var_xs)
        preds = torch.argmax(outputs, 1)

        preds = preds.data.cpu().numpy()
        labels = ys.cpu().numpy()
        n_xs = var_xs.cpu().numpy()

        epsilon_list = [self.epsilon] * self.batch_size
        delta_list = [self.delta] * self.batch_size
        # 注意是复制，不是直接赋值
        adversarial_samples = n_xs.copy()
        # get the first step of boudary as init parameter and input
        adversarial_sample = n_xs[0]
        numbers = n_xs.shape[0]
        rangenumbers = 0
        if numbers <= self.batch_size:
            rangenumbers = numbers
        else:
            rangenumbers = self.batch_size
        for i in range(rangenumbers):
            origin_sample = n_xs[i]
            # Move to the first boundary
            adversarial_sample = self._init_sample(
                origin_sample, preds[i], target_flag, 0, 1
            )
            for search_for_epsilon in range(self.binary_search_steps):

                for iteration_times in range(self.max_iter):
                    potential_perturbed_images = []
                    for _ in range(self.sample_size):

                        perturbed_image = adversarial_sample + self._orthogonal_perturb(
                            delta_list[i], adversarial_sample, origin_sample
                        )
                        perturbed_image = np.array(perturbed_image)
                        perturbed_image = np.clip(
                            perturbed_image, self.lower_bound, self.upper_bound
                        )
                        potential_perturbed_images.append(perturbed_image)
                    # potential_perturbed_images
                    var_images = Variable(
                        torch.from_numpy(np.array(potential_perturbed_images)).to(
                            self.device
                        )
                    )

                    predictions_outputs = self.model(var_images.float())
                    predictions = torch.argmax(predictions_outputs, 1)
                    predictions = predictions.data.cpu().numpy()
                    if target_flag:
                        satisfied = predictions == labels[i]
                    else:
                        satisfied = predictions != labels[i]
                    delta_ratio = np.mean(satisfied)

                    if delta_ratio < 0.5:
                        delta_list[i] *= self.step_adapt
                    else:
                        delta_list[i] /= self.step_adapt

                    if delta_ratio > 0:

                        adversarial_sample = np.array(potential_perturbed_images)[
                            np.where(satisfied)[0][0]
                        ]

                        break
                for _ in range(self.max_iter):

                    perturb = origin_sample - adversarial_sample
                    perturb *= epsilon_list[i]
                    potential_adv = adversarial_sample + perturb
                    potential_adv = np.clip(potential_adv, 0, 1)
                    potential_adv_expand = np.expand_dims(potential_adv, axis=0)
                    potential_image = Variable(
                        torch.from_numpy(potential_adv_expand).to(self.device)
                    )
                    output = self.model(potential_image.float())
                    pred_out = torch.argmax(output, 1)
                    pred_out = pred_out.data.cpu().numpy()

                    if target_flag:
                        satisfied = pred_out == labels[i]
                    else:
                        satisfied = pred_out != labels[i]
                    if satisfied:
                        adversarial_sample = potential_adv

                        epsilon_list[i] /= self.step_adapt
                        break
                    else:
                        epsilon_list[i] *= self.step_adapt

            adversarial_samples[i] = adversarial_sample
        return torch.from_numpy(adversarial_samples)