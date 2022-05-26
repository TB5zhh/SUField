import torch


class PerClassCriterion:

    def __init__(self, num_classes=20) -> None:
        self.num_classes = num_classes
        self.stat = torch.zeros((num_classes, num_classes))

    def update(self, prediction, target):
        assert prediction.shape == target.shape
        mask = torch.logical_and(target >= 0, target < self.num_classes)
        self.stat += torch.bincount(
            self.num_classes * target[mask] + prediction.cpu()[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def get_iou(self):
        return self.stat.diag() / (self.stat.sum(0) + self.stat.sum(1) - self.stat.diag())

    def get_precision(self):
        return self.stat.diag() / self.stat.sum(0)

    def get_recall(self):
        return self.stat.diag() / self.stat.sum(1)