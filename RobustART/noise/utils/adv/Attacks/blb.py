import numpy as np
import torch
from torch.autograd import Variable


class BLB:
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

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            init_const:
            binary_search_steps:
            max_iter:
        } 
        @return: None
        """
        #
        self.init_const = float(kwargs.get("init_const", 0.01))
        #
        self.max_iter = int(kwargs.get("max_iter", 1000))
        #
        self.binary_search_steps = int(kwargs.get("binary_search_steps", 5))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        device = self.device
        targeted = self.IsTargeted
        batch_size = xs.shape[0]
        copy_xs = np.copy(xs.numpy())
        copy_ys = np.copy(ys.numpy())

        var_xs = Variable(
            torch.from_numpy(copy_xs).float().to(device), requires_grad=True
        )
        var_ys = Variable(torch.LongTensor(copy_ys).to(device))

        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)

        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_xs.shape)
        current_prediction_class = [-1] * batch_size

        def attack_achieved(pre_softmax, target_class):
            targeted = self.IsTargeted
            if targeted:
                return np.argmax(pre_softmax) == target_class
            else:
                return np.argmax(pre_softmax) != target_class

        for search_for_c in range(self.binary_search_steps):
            # the perturbation
            r = torch.zeros_like(var_xs).float()
            r = Variable(r.to(device), requires_grad=True)

            # use LBFGS to optimize the perturbation r, with default learning rate parameter and other parameters
            optimizer = torch.optim.LBFGS([r], max_iter=self.max_iter)
            var_const = Variable(torch.FloatTensor(const_origin).to(device))
            print("\tbinary search step {}:".format(search_for_c))

            # The steps to be done when doing optimization iteratively.
            def closure():
                perturbed_images = torch.clamp(var_xs + r, min=0.0, max=1.0)
                prediction = self.model(perturbed_images)
                l2dist = torch.sum((perturbed_images - var_xs) ** 2, [1, 2, 3])
                constraint_loss = -self.criterion(prediction, var_ys)
                if targeted:
                    constraint_loss = self.criterion(prediction, var_ys)
                loss_f = var_const * constraint_loss
                loss = (
                    l2dist.sum() + loss_f.sum()
                )  # minimize c|r| + loss_f(x+r,l), l is the target label, r is the perturbation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

            perturbed_images = torch.clamp(var_xs + r, min=0.0, max=1.0)
            prediction = self.model(perturbed_images)
            l2dist = torch.sum((perturbed_images - var_xs) ** 2, [1, 2, 3])

            # the following is analogy to CW2 attack
            for i, (dist, score, perturbation) in enumerate(
                zip(
                    l2dist.data.cpu().numpy(),
                    prediction.data.cpu().numpy(),
                    perturbed_images.data.cpu().numpy(),
                )
            ):
                if dist < best_l2[i] and attack_achieved(score, copy_ys[i]):
                    best_l2[i] = dist
                    current_prediction_class[i] = np.argmax(score)
                    best_perturbation[i] = perturbation

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if (
                    current_prediction_class[i] == copy_ys[i]
                    and current_prediction_class[i] != -1
                ):
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2
                    else:
                        const_origin[i] *= 10

        adv_xs = torch.from_numpy(best_perturbation)
        adv_xs = torch.tensor(adv_xs, dtype=torch.float32)
        return adv_xs