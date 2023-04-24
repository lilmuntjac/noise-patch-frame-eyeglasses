import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from lib.datasets import FairFace
from lib.models import FairFaceModel
from lib.fabricator import *
from lib.fairnessCategori import *
from lib.utils import *

def main(args):
    start_time = time.time()
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataset and dataloader: FairFace, categorical model
    fairface = FairFace(batch_size=args.batch_size)
    train_dataloader, val_dataloader = fairface.train_dataloader, fairface.val_dataloader
    # The base model, placeholder for its optimizer and scheduler
    model = FairFaceModel(weights=None).to(device)
    _optimizer = torch.optim.RMSprop(model.parameters(),)
    _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=1,)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    load_model(model, _optimizer, _scheduler, name=args.model_ckpt_name, root_folder=model_ckpt_path)
    # adversarial component, adversarial optimizer (SGD), and adversarial scheduler
    advatk_ckpt_path = Path(args.advatk_ckpt_root)/args.advatk_name
    match args.adv_type:
        case "noise" | "clean":
            adv_component = torch.zeros((1, 3, 224, 224)).to(device)
            noise_overlay = NoiseOverlay()
        case "patch" | "frame" | "eyeglasses":
            adv_component = torch.full((1, 3, 224, 224), 0.5).to(device)
            heuristic_masking = HeuristicMasking(args.adv_type, thickness=args.frame_thickness)
        case _:
            assert False, f'the adversarial type {args.adv_type} not supported'
    # try load the adversarial attack element
    if args.resume:
        adv_component = load_stats(name=args.resume, root_folder=advatk_ckpt_path)
        adv_component = torch.from_numpy(adv_component).to(device)

    adv_component = nn.Parameter(adv_component)
    # adversary_optimizer = torch.optim.SGD([adv_component], lr=args.lr, )
    # adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1,)
    # coef = torch.tensor(args.coef).to(device)
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')

    def save(data, name, root_folder='./imgs', denormal=True):
        folder = Path(root_folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        data = denormalize(data).detach().cpu() if denormal else data.detach().cpu()
        save_image(data, path)
        
    def save_adv_image():
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)
            # produce the adversarial image
            match args.adv_type:
                case "noise" | "clean":
                    adv_image, label = noise_overlay.apply(data, label, adv_component)
                case "patch":
                    heuristic_masking.set_random_transform(args.patch_train_rotation, tuple(args.patch_train_ratio2img), args.patch_avoid)
                    adv_image, label = heuristic_masking.apply(data, label, adv_component)
                case "frame":
                    adv_image, label = heuristic_masking.apply(data, label, adv_component)
                case "eyeglasses":
                    data, label, landmark = heuristic_masking.get_landmark(data, label)
                    heuristic_masking.set_eyeglasses_transform(landmark)
                    adv_image, label = heuristic_masking.apply(data, label, adv_component)
                case _:
                    assert False, f'the adversarial type {args.adv_type} not supported'
            # save a batch of image
            for i in range(adv_image.shape[0]):
                im = adv_image[i:i+1,]
                save(im, str(i), root_folder=f'./imgs/{args.adv_type}', denormal=False)
            break

    save_adv_image()

    def save_adv_component(adv_component):
        adv_component_showcase = adv_component.clone()
        black_img = torch.zeros((args.batch_size, 3, 224, 224)).to(device)
        gray_img = torch.full((args.batch_size, 3, 224, 224), 0.5).to(device)
        dummy_label = torch.zeros((args.batch_size, 1)).to(device)
        save(gray_img[0], f'gray', root_folder=f'./exp', denormal=False)
        match args.adv_type:
            case "noise" | "clean":
                # type 1: shift the component value by budget, clip, then save the image
                # adv_component_showcase += 0.1
                # adv_component_showcase.data.clamp_(0.0, 1.0)
                # type 2: apply noise onto a gray image
                adv_component_showcase*=10 # make it more obvious
                adv_component_showcase, _ = noise_overlay.apply(gray_img, dummy_label, adv_component_showcase)
                adv_component_showcase.data.clamp_(0.0, 1.0)
            case "patch" | "eyeglasses":
                # identity transform
                heuristic_masking.set_identity_transform()
                adv_component_showcase, _ = heuristic_masking.apply(black_img, dummy_label, adv_component_showcase)
            case "frame":
                # just apply the component onto ther black image
                adv_component_showcase, _ = heuristic_masking.apply(black_img, dummy_label, adv_component_showcase)
            case _:
                assert False, f'the adversarial type {args.adv_type} not supported'
        # save the result
        save(adv_component_showcase[0], f'{args.adv_type}', root_folder=f'./exp', denormal=False)

    save_adv_component(adv_component)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial component training")
    # For base model loaded
    parser.add_argument("--model-ckpt-root", default='/tmp2/npfe/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-stat-root", default='/tmp2/npfe/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--model-ckpt-name", default='default_model', type=str, help='name for the model checkpoint, without .pth')
    
    # For adversarial element (share by all type)
    parser.add_argument("--advatk-ckpt-root", default='/tmp2/npfe/advatk', type=str, help='root path for adversarial atttack statistic')
    parser.add_argument("--advatk-stat-root", default='/tmp2/npfe/advatk_stats', type=str, help='root path for adversarial attack itself')
    parser.add_argument("--advatk-name", default='default_advatk', type=str, help='name for the advatk trained')
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check on the element loaded")
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=120, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="step size for the adversarial element")
    parser.add_argument("--adv-type", default=None, type=str, help="type of adversarial element, only 'noise', 'patch', 'frame', and 'eyeglasses' are allowed")
    # noise only
    # budget ?
    # patch only
    parser.add_argument("--patch-train-rotation", default=1/12, type=float, help="patch rotation durning training")
    parser.add_argument("--patch-val-rotation", default=1/12, type=float, help="patch rotation durning validation")
    parser.add_argument("--patch-train-ratio2img", default=[0.05, 0.08], type=float, nargs='+', help="patch ratio to image durning training")
    parser.add_argument("--patch-val-ratio2img", default=[0.07, 0.07], type=float, nargs='+', help="patch ratio to image durning validation")
    parser.add_argument("--patch-avoid", default=None, type=float, help="distance from center of image to avoid covering")
    # frame only
    parser.add_argument("--frame-thickness", default=0.25, type=float, help="the thickness of the frame")
    # eyeglasses only
    # transform method ?
    # fairness parameter
    # parser.add_argument("--fairness-matrix", default="prediction quaility", help="how to measure fairness")
    parser.add_argument("--sens-type", default="race", type=str, help="sensitive attribute to divide the dataset into 2 group")
    parser.add_argument("--attr-type", default="all", type=str, help="attribute that fairness is evaluated on")
    parser.add_argument("--fairness-target", default=0.03, type=float, help="Fairness target value")
    parser.add_argument("--quality-target", default=0.05, type=float, help="Max gap loss for prediction quaility")
    parser.add_argument("--coef-mode", default="fix", type=str, help="method to adjust coef durinig training")
    # binary model
    # parser.add_argument("--p-coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on positive recovery loss, need to be match with the number of attributes")
    # parser.add_argument("--n-coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on negative recovery loss, need to be match with the number of attributes")
    # category model
    parser.add_argument("--coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on recovery loss, must match the number of losses used")
    
    # method taken
    parser.add_argument("--loss-type", default='direct', type=str, help="Type of loss used")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)