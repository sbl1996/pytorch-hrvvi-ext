

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def clip(model, tol=1e-6):
    for p in model.parameters():
        p[p.abs() < tol] = 0
    return model
