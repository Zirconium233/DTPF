from .text_xrestormer import Text_XRestormer, XRestormer
from .loss_text_xrestormer import TextFusionLoss

MODELS = {
          "Text_XRestormer": Text_XRestormer,
          "XRestormer": XRestormer,
}

LOSSES = {
          "Loss_TextFusion": TextFusionLoss
}