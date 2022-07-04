import os
import time
import torch
import argparse
import clip
import json

from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix, Accuracy
from torchvision.transforms import transforms
from dataset import *
from config import update_config, cfg

def _convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()
def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--test', type=bool, default=False,
                        help='Test or train.')

    args = parser.parse_args()
    return args


dataset_names = {"pathmnist": PathMNIST,
                 "octmnist": OCTMNIST,
                 "pneumoniamnist":PneumoniaMNIST,
                 "chestmnist": ChestMNIST,
                 "dermamnist":DermaMNIST,
                 "breastmnist":BreastMNIST,
                 "bloodmnist":BloodMNIST,
                 "tissuemnist": TissueMNIST,
                 "organamnist":OrganAMNIST,
                 "organcmnist":OrganCMNIST,
                 "organsmnist":OrganSMNIST,
                 }

def eval(model, data_loader, device):
    texts = clip.tokenize(data_loader.dataset.texts).to(device)
    predictions = []
    true_labels = []
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            logits_per_image, logits_per_text = model(image, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            pred = np.argmax(probs, axis=-1)

            predictions.extend(pred)
            true_labels.extend(label.cpu().tolist())
    print(len(predictions))
    print(len(true_labels))
    confmat = ConfusionMatrix(num_classes=len(data_loader.dataset.info["label"]))
    matrix = confmat(torch.tensor(predictions), torch.tensor(true_labels))
    accuracy = Accuracy()
    acc = accuracy(torch.tensor(predictions), torch.tensor(true_labels))

    print("[INFO] Accuracy: ")
    print(acc)
    print("[INFO] ConfusionMatrix")
    print(matrix)

    return {"accuracy":acc.item(), "confusion_matrix":matrix.tolist()}



if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    update_config(cfg, args)

    # torch.manual_seed(cfg.SEED)
    # torch.cuda.manual_seed(cfg.SEED)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)

    # Load dataset
    Dataset = dataset_names.get(cfg.DATASET.DATASET, PathMNIST)
    test_transform = transforms.Compose([transforms.ToTensor()])
    #train_dataset = Dataset("train", cfg)
    test_dataset = Dataset("test", cfg, transform=test_transform)
    print("[INFO]--------------DATASET INFO -------------------")
    print(test_dataset)
    print("[INFO]------------CLASSIFICATION TEXT--------------")
    print(test_dataset.texts)
    drop_last = False
    drop_last_val = False
    # train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=drop_last,
    #                           pin_memory=True)

    test_loader = DataLoader(test_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=drop_last_val,
                            pin_memory=True)

    # Load model
    model, _ = clip.load(cfg.PRETRAINED.CLIP_VISION_ENCODER, jit=False)
    checkpoint = torch.load(cfg.PRETRAINED.CLIP_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.float()

    # Zero-shot evaluation
    print("[INFO]-----------EVALUATING------------")
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    result_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATASET)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result = eval(model, test_loader, device)

    with open( os.path.join(result_dir, "result.json"), "w") as outfile:
        json.dump(result, outfile)



