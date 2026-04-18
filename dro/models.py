import torch
import torch.nn as nn
import torchvision


def build_resnet50_backbone(pretrained=True):
    try:
        if pretrained:
            try:
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                backbone = torchvision.models.resnet50(weights=weights)
            except AttributeError:
                backbone = torchvision.models.resnet50(pretrained=True)
        else:
            try:
                backbone = torchvision.models.resnet50(weights=None)
            except TypeError:
                backbone = torchvision.models.resnet50(pretrained=False)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to load pretrained ResNet50 weights. Re-run with --train_from_scratch."
            ) from exc
        raise

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    # The backbone now implements phi(a; x_1) and returns a d-dimensional feature vector.
    backbone.train(False)
    return backbone, feature_dim


def init_classifier(feature_dim, num_classes, device):
    # y_1 in the user's notation: a linear classifier W in R^{d x C}.
    classifier = torch.empty(feature_dim, num_classes, device=device)
    nn.init.normal_(classifier, mean=0.0, std=0.01)
    return nn.Parameter(classifier)


def classifier_logits(features, classifier_weight):
    # logits = phi(a; x_1)^T y_1
    return features.matmul(classifier_weight)
