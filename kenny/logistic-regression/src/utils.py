import torch

def get_model_parameters(model):
    """Returns the parameters of a PyTorch model."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_params(model, parameters):
    """Sets the parameters of a PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def set_initial_params(model):
    """Sets initial parameters as zeros. Required since model params are uninitialized until model.fit is called."""
    for param in model.parameters():
        param.data.zero_()
