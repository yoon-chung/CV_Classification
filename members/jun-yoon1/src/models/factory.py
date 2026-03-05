import timm


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        if pretrained:
            print(
                f"[WARN] pretrained weights unavailable for {model_name}. "
                f"Fallback to pretrained=False. ({e})"
            )
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )
        else:
            raise
    return model
