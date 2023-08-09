import torch
import torch.nn as nn
import torch.nn.functional as F




class PGD():
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        # super().__init__("PGD")
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.device = next(model.parameters()).device
        self.targeted = False

    def __call__(self, images, labels, epoch):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            outputs = self.model(adv_images, epoch)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            
            grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        return adv_images
