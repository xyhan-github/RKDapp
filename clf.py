import torch
from PIL import Image
from gc import collect

TOPN = 5

def predict(image_path, model, transform, ind_to_label):
    # Create transforms
    #https://pytorch.org/docs/stable/torchvision/models.html

    # clean memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    collect()

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    with torch.no_grad:
        out = model(batch_t)
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)

    output = [(ind_to_label[int(idx)], prob[idx].item()) for idx in indices[0][:TOPN]]

    # clean memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    collect()

    return output


# FOR TESTING
if __name__ == "__main__":
    import json
    from torchvision import transforms
    from load_model_app import load_model

    # create dictionary
    info = {'checkpoint'    : 'dataset=RKDMuseum-net=ResNet50-lr=0p01-examples_per_class=None-num_classes=499-train_seed=0-forward_class=Classification-epoch=96.pth',
            'net'           : 'ResNet50',
            'num_classes'   : 499,
            'resnet_type'   : 'big',
            'pretrained'    : False,
            'input_ch'      : 3,
            'device'        : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),}

    # convert dictionary to object
    info = type('obj', (object,), info)

    # load model and transform
    model = load_model(info)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # load json file
    with open('ind_to_label.json') as f:
        ind_to_label = json.load(f)
        # convert keys to int
        ind_to_label = {int(k): v for k, v in ind_to_label.items()}

    print('Prediction:')
    print(predict('/Users/xiaoyan/Github_link/RKD/RKDdata2/images/0000000001.jpg', model, transform, ind_to_label))
