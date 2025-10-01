from configs.args import get_cumulants_sbi_args
from utils import get_datasets

args = get_cumulants_sbi_args()

config, cumulants_dataset, datasets = get_datasets(args) 