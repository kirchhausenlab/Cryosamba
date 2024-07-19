import os, glob
from distutils.util import strtobool
from easydict import EasyDict
import sys
import json
import shutil
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


def listify(x):
    return x if isinstance(x, list) else [x]


### File utils


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def load_json(path):
    with open(path) as f:
        cfg = EasyDict(json.load(f))
    return cfg


def save_json(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


### Loguru utils


def logger_info(rank, message):
    if rank == 0:
        logger.info(message)
    else:
        pass


def console_filter(record):
    return record["extra"].get("to_console", True)


def set_logger(save_dir):
    logger_path = os.path.join(save_dir, "runtime.log")
    logger_format = "<green>{time:YYYY/MM/DD HH:mm:ss!UTC}</green> | <level>{level}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stdout, format=logger_format, filter=console_filter)
    logger.add(logger_path, format=logger_format)


### Tensorboard utils


def set_writer(save_dir, layout):
    writer = SummaryWriter(save_dir)
    writer.add_custom_scalars(layout)
    return writer


def set_writer_layout_train(max_frame_gap):
    train_loss_labels = [f"train_loss/gap {t}" for t in range(1, max_frame_gap + 1)]
    layout = {
        "Plots": {
            "train_loss": ["Multiline", train_loss_labels],
            "val_loss": ["Multiline", ["val_loss/val"]],
            "learning_rate": ["Multiline", ["learning_rate/lr"]],
        },
    }
    return layout


def set_writer_train(cfg):
    layout = set_writer_layout_train(cfg.train_data.max_frame_gap)
    writer = set_writer(cfg.train_dir, layout)
    return writer


### Run utils


def prompt(query):
    sys.stdout.write("%s [y/n]:" % query)
    val = input()

    try:
        ret = strtobool(val)
    except ValueError:
        sys.stdout("please answer with y/n")
        return prompt(query)
    return ret


def setup_run(cfg, mode):
    save_dir = cfg.train_dir if mode == "training" else cfg.inference_dir
    can_resume_run = (
        os.path.exists(os.path.join(save_dir, "last.pt"))
        if mode == "training"
        else True
    )
    new_run = True
    if os.path.exists(save_dir):
        while True:
            if (
                prompt(
                    f"Dir {save_dir} already exists. Would you like to overwrite it?"
                )
                == True
            ):
                shutil.rmtree(save_dir)
            elif can_resume_run:
                if prompt(f"Would you like to resume the existing {mode}?") == True:
                    message = f"Resuming {mode} from {save_dir}"
                    new_run = False
                else:
                    print("Understandable, have a nice day.")
                    sys.exit()
            else:
                print("No checkpoints found.")
                sys.exit()
            break
    if new_run:
        message = f"Starting new {mode} in {save_dir}"
        make_dir(save_dir)
        save_json(os.path.join(save_dir, "config.json"), cfg)
    set_logger(save_dir)
    logger.info(message)
