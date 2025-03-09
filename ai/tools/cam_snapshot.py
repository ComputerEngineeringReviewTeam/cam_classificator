import torch.nn
import torchinfo
import os

from ai.paths import LOGS_DIR
import ai.config as config

class CamSnapshot:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 device: str,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.input_data = next(iter(dataloader))[0]
        self.device = device

    def __str__(self):
        summary_str = f"""
    Model:
{self._model_summary()}
    Optimizer: 
{str(self.optimizer)}
    Loss Function: 
{str(self.loss_fn)}
"""
        summary_str += self._config_summary()
        return summary_str

    def _model_summary(self):
        return str(torchinfo.summary(self.model,
                                     input_data=[self.input_data],
                                     device=self.device,
                                     verbose=0))

    def _config_summary(self):
        config_str = "\tConfig:\n"
        for key, value in config.__dict__.items():
            if key.isupper():
                config_str += f"{key}: {value}\n"
        return config_str
    def save(self, path: str, encoding: str = "utf-8", absolute: bool = False):
        if not absolute:
            path = os.path.join(LOGS_DIR, path)
        with open(path, "w", encoding=encoding) as f:
            f.writelines(str(self))
