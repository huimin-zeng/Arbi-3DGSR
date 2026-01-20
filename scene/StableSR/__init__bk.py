import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
# from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from scene.StableSR.ldm.data.base import Txt2ImgIterableBaseDataset
from scene.StableSR.ldm.util import instantiate_from_config, instantiate_from_config_sr
from pytorch_lightning.loggers import WandbLogger


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)


    parser.add_argument(
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
 
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--scale_lr",
        type=bool,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config_sr(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config_sr(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


def get_stablesr():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
 
    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load("scene/StableSR/v2-finetune_text_T_512.yaml")]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # model
    model = instantiate_from_config(config.model)
    
    pl_sd = torch.load("../stablesr_000117.ckpt", map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=True)
    
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)
    return model.cuda()
    # model.cuda()
    # model.requires_grad_(False)
    # model(x=HR+noise, c=LR, gt=HR)
    
    # forward(self, x, c, gt, *args, **kwargs):
    # self.p_losses(gt, c, struc_c, t, t_ori, x, *args, **kwargs)
    # p_losses(self, x_start, cond, struct_cond, t, t_ori, z_gt, noise=None)

 

    # model.configs = config 
    # trainer.fit(model, data)
import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from scene.StableSR.ldm.util import instantiate_from_config
import math
import copy
from scene.StableSR.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
          
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    size=image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.,size


def read_image(im_path):
    im=Image.open(im_path).convert("RGB")
    size=im.size
    im = np.array(im).astype(np.float32)/255.0
    im = im[None].transpose(0,3,1,2)
    im = (torch.from_numpy(im) - 0.5) / 0.5
    return im.cuda(),size


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_stablesr_vq():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=200,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=4.0,
        help="upsample scale",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="adain",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )

    opt = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    vqgan_config = OmegaConf.load("scene/StableSR/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, "../vqgan_cfw_00011.ckpt")
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.input_size,opt.input_size)),
        # torchvision.transforms.CenterCrop(opt.input_size),
    ])

    config = OmegaConf.load("scene/StableSR/v2-finetune_text_T_512.yaml")
    model = load_model_from_config(config, "../stablesr_000117.ckpt")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    img_list_ori = os.listdir(opt.init_img)
    img_list = copy.deepcopy(img_list_ori)
    init_image_list = []
    for item in img_list_ori:
        if os.path.exists(os.path.join(outpath, item)):
            img_list.remove(item)
            continue
        # cur_image,(image_h,image_w) = load_img(os.path.join(opt.init_img, item))
        cur_image,(image_h,image_w) = read_image(os.path.join(opt.init_img, item))
        cur_image = transform(cur_image.to(device))
        cur_image = cur_image.clamp(-1, 1)
        init_image_list.append(cur_image)
    init_image_list = torch.cat(init_image_list, dim=0)
    niters = math.ceil(init_image_list.size(0) / batch_size)
    init_image_list = init_image_list.chunk(niters)

    # size_min = min(image_h,image_w)
    # upsample_scale = max(opt.input_size/size_min, opt.upscale)
    

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)

    param_list = []
    untrain_paramlist = []
    name_list = []
    for k, v in model.named_parameters():
        if 'spade' in k or 'structcond_stage_model' in k:
            param_list.append(v)
        else:
            name_list.append(k)
            untrain_paramlist.append(v)
    trainable_params = sum(p.numel() for p in param_list)
    untrainable_params = sum(p.numel() for p in untrain_paramlist)
    print(name_list)
    print(trainable_params)
    print(untrainable_params)

    param_list = []
    untrain_paramlist = []
    for k, v in vq_model.named_parameters():
        if 'fusion_layer' in k:
            param_list.append(v)
        elif 'loss' not in k:
            untrain_paramlist.append(v)
    trainable_params += sum(p.numel() for p in param_list)
    # untrainable_params += sum(p.numel() for p in untrain_paramlist)
    print(trainable_params)
    print(untrainable_params)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    niqe_list = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                count = 0
                for n in trange(niters, desc="Sampling"):
                    init_image = init_image_list[n]
                    init_latent_generator, enc_fea_lq = vq_model.encode(init_image)#[torch.Size([2, 256, 256, 256]),torch.Size([2, 256, 256, 256])]                    
                    init_latent = model.get_first_stage_encoding(init_latent_generator)#torch.Size([2, 4, 64, 64])
                    '''
                    text_init = ['']*init_image.size(0)
                    semantic_c = model.cond_stage_model(text_init)
                    noise = torch.randn_like(init_latent)
                    # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                    t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                    t = t.to(device).long()
                    x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                    x_T = None

                    samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
                    x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq) #samples:torch.Size([2, 4, 64, 64]) [torch.Size([2, 256, 256, 256]),torch.Size([2, 256, 256, 256])]
                    ''' 
                    x_samples = vq_model.decode(init_latent * 1. / model.scale_factor, enc_fea_lq) #samples:torch.Size([2, 4, 64, 64]) [torch.Size([2, 256, 256, 256]),torch.Size([2, 256, 256, 256])]
                    
                    if opt.colorfix_type == 'adain':
                        x_samples = adaptive_instance_normalization(x_samples, init_image)
                    elif opt.colorfix_type == 'wavelet':
                        x_samples = wavelet_reconstruction(x_samples, init_image)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
         
                    x_samples = F.interpolate(x_samples, size=(int(image_w*opt.upscale), int(image_h*opt.upscale)),mode='bicubic' )
                    x_samples = torch.clamp(x_samples, min=0.0, max=1.0)

                    for i in range(init_image.size(0)):
                        img_name = img_list.pop(0)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        
                        
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, basename+'.png'))

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

