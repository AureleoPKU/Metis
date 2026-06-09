from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningOpenX
from genie.model_tokenizer import Action_Tokenizer

cli = LightningCLI(
    Action_Tokenizer,
    LightningOpenX,
    seed_everything_default=42,
)
