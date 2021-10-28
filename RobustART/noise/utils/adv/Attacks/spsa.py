#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description:
@Date: 2019-03-29 09:19:32
@LastEditTime: 2019-04-15 09:25:32
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch import optim

# from .utils import MarginalLoss

from .attack import Attack


class SPSA(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: SPSA
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        super(SPSA, self).__init__(model, device, IsTargeted)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description:
        @param {
            theta:
            gamma:
        }
        @return: None
        """

        self.eps = float(kwargs.get("eps", 0.05))
        self.learning_rate = float(kwargs.get("learning_rate", 0.01))
        self.delta = float(kwargs.get("delta", 0.01))
        self.spsa_samples = int(kwargs.get("spsa_samples", 32))
        self.spsa_iters = int(kwargs.get("spsa_iters", 2))
        self.is_debug = bool(kwargs.get("is_debug", False))
        self.sanity_checks = bool(kwargs.get("sanity_checks", True))
        self.nb_iter = int(kwargs.get("nb_iter", 20))

    def clip_eta(self, eta, norm, eps):
        """
        PyTorch implementation of the clip_eta in utils_tf.
        :param eta: Tensor
        :param norm: np.inf, 1, or 2
        :param eps: float
        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError("norm must be np.inf, 1, or 2.")

        avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
        reduc_ind = list(range(1, len(eta.size())))
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            if norm == 1:
                raise NotImplementedError("L1 clip is not implemented.")
                norm = torch.max(
                    avoid_zero_div,
                    torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True),
                )
            elif norm == 2:
                norm = torch.sqrt(
                    torch.max(
                        avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                    )
                )
            factor = torch.min(
                torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
            )
            eta *= factor
        return eta

    def _project_perturbation(
        self,
        perturbation,
        norm,
        epsilon,
        input_image,
        clip_min=-np.inf,
        clip_max=np.inf,
    ):
        """
        Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
        hypercube such that the resulting adversarial example is between clip_min and clip_max,
        if applicable. This is an in-place operation.
        """

        clipped_perturbation = self.clip_eta(perturbation, norm, epsilon)
        new_image = torch.clamp(input_image + clipped_perturbation, clip_min, clip_max)

        perturbation.add_((new_image - input_image) - perturbation)

    def _compute_spsa_gradient(self, loss_fn, x, delta, samples, iters):
        """
        Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
        given parameters. The gradient is approximated by evaluating `iters` batches
        of `samples` size each.
        """

        assert len(x) == 1
        num_dims = len(x.size())
        x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

        grad_list = []
        for i in range(iters):
            delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
            delta_x = torch.cat([delta_x, -delta_x])
            # print(delta_x.shape,x.shape)
            loss_vals = loss_fn(x + delta_x)
            while len(loss_vals.size()) < num_dims:
                loss_vals = loss_vals.unsqueeze(-1)
            avg_grad = (
                torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta
            )
            grad_list.append(avg_grad)

        return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)

    def _margin_logit_loss(self, logits, labels):
        """
        Computes difference between logits for `labels` and next highest logits.
        The loss is high when `label` is unlikely (targeted by default).
        """
        correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

        logit_indices = torch.arange(
            logits.size()[1], dtype=labels.dtype, device=labels.device,
        )[None, :].expand(labels.size()[0], -1)
        incorrect_logits = torch.where(
            logit_indices == labels[:, None],
            torch.full_like(logits, float("-inf")),
            logits,
        )
        max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

        return max_incorrect_logits - correct_logits

    def spsa(
        self,
        x,
        y=None,
        norm=np.inf,
        clip_min=-np.inf,
        clip_max=np.inf,
        early_stop_loss_threshold=None,
    ):
        """
        @description:
        @param {
            x: [1xCxHxW]
            y: [1xCxHxW]
        }
        @return: adv_x
        """
        device = self.device
        copy_x = np.copy(x.numpy())
        copy_y = np.copy(y.numpy())

        eps = self.eps
        is_debug = self.is_debug
        learning_rate = self.learning_rate
        delta = self.delta
        spsa_samples = self.spsa_samples
        spsa_iters = self.spsa_iters
        targeted = self.IsTargeted
        nb_iter = self.nb_iter
        sanity_checks = self.sanity_checks

        v_x = Variable(torch.from_numpy(copy_x).float().to(device))
        v_y = Variable(torch.LongTensor(copy_y).to(device))

        if v_y is not None and len(v_x) != len(v_y):
            raise ValueError(
                "number of inputs {} is different from number of labels {}".format(
                    len(v_x), len(v_y)
                )
            )
        if v_y is None:
            v_y = torch.argmax(self.model(v_x), dim=1)

            # The rest of the function doesn't support batches of size greater than 1,
            # so if the batch is bigger we split it up.
        if len(x) != 1:
            adv_x = []
            for x_single, y_single in zip(x, y):
                adv_x_single = self.spsa(
                    x=x_single.unsqueeze(0),
                    y=y_single.unsqueeze(0),
                    norm=np.inf,
                    clip_min=-np.inf,
                    clip_max=np.inf,
                    early_stop_loss_threshold=None,
                )
                adv_x.append(adv_x_single)
            return torch.cat(adv_x)

        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return v_x

        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        asserts.append(torch.all(v_x >= clip_min))
        asserts.append(torch.all(v_x <= clip_max))

        if is_debug:
            print("Starting SPSA attack with eps = {}".format(eps))

        perturbation = (torch.rand_like(v_x) * 2 - 1) * eps
        self._project_perturbation(perturbation, norm, eps, v_x, clip_min, clip_max)
        optimizer = optim.Adam([perturbation], lr=learning_rate)

        for i in range(nb_iter):

            def loss_fn(pert):
                """
                Margin logit loss, with correct sign for targeted vs untargeted loss.
                """
                logits = self.model(v_x + pert)
                loss_multiplier = 1 if targeted else -1
                return loss_multiplier * self._margin_logit_loss(
                    logits, v_y.expand(len(pert))
                )

            spsa_grad = self._compute_spsa_gradient(
                loss_fn, v_x, delta=delta, samples=spsa_samples, iters=spsa_iters
            )
            perturbation.grad = spsa_grad
            optimizer.step()

            self._project_perturbation(perturbation, norm, eps, v_x, clip_min, clip_max)

            loss = loss_fn(perturbation).item()
            if is_debug:
                print("Iteration {}: loss = {}".format(i, loss))
            if (
                early_stop_loss_threshold is not None
                and loss < early_stop_loss_threshold
            ):
                break

        adv_x = torch.clamp((v_x + perturbation).detach(), clip_min, clip_max)

        if norm == np.inf:
            asserts.append(torch.all(torch.abs(adv_x - v_x) <= eps + 1e-6))
        else:
            asserts.append(
                torch.all(
                    torch.abs(
                        torch.renorm(adv_x - copy_x, p=norm, dim=0, maxnorm=eps)
                        - (adv_x - v_x)
                    )
                    < 1e-6
                )
            )
        asserts.append(torch.all(adv_x >= clip_min))
        asserts.append(torch.all(adv_x <= clip_max))

        if sanity_checks:
            assert np.all(asserts)

        return adv_x

    def generate(self, xs=None, ys=None):
        """
        @description:
        @param {
            xs:
            ys:
        }
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted
        adv_xs = []
        for i in range(len(xs)):
            # print('\tprocessing {}'.format(i + 1))
            adv_x = self.spsa(x=xs[i : i + 1], y=ys[i : i + 1])
            adv_xs.append(adv_x)

        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs
