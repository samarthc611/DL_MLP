import torch
from mlp_mixer_pytorch import MLPMixer

model = MLPMixer(
    image_size=256, channels=1, patch_size=16, dim=512, depth=12, num_classes=3
)


# def load_data(batch_size=32, img_size=(256, 256)):
#     dataset_path = "Dataset/Public_Medical_Image_Datasets/covid19-pneumonia-dataset"
#
#     train_dir = os.path.join(dataset_path, "train_dir")
#     valid_dir = os.path.join(dataset_path, "valid_dir")
#     test_dir = os.path.join(dataset_path, "test_dir")
#
#     if not all(os.path.exists(d) for d in [train_dir, valid_dir, test_dir]):
#         raise FileNotFoundError("One or more dataset directories not found.")
#
#     transform = transforms.Compose(
#         [
#             transforms.Resize(img_size),
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#         ]
#     )
#
#     train_ds = datasets.ImageFolder(train_dir, transform=transform)
#     valid_ds = datasets.ImageFolder(valid_dir, transform=transform)
#     test_ds = datasets.ImageFolder(test_dir, transform=transform)
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
#
#     return train_loader, valid_loader, test_loader


img = torch.randn(1, 1, 256, 256)
pred = model(img)  # (1, 1000)
print(pred)