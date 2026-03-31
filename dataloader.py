import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

class FASDataset(Dataset):
    """
    Custom PyTorch Dataset for Face Anti-Spoofing, designed for testing on
    the Oulu-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

    It loads data from paths like:
    - root_dir/
      - domain-generalization/
        - Oulu_images_live.npy
        - Oulu_images_spoof.npy
        - casia_images_live.npy
        - etc.
    
    Returns a tuple:
    (image_tensor, labels_tensor)
    where label_tensor contains the binary label.
    """

    def __init__(self, root_dir, protocol, transform=None, live_only=False):
        """
        Args:
            root_dir (string): Root directory containing the 'domain-generalization' folder.
            protocol (list of strings): List of datasets to use, e.g., ['O', 'Ca', 'I', 'M'].
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.live_only = live_only
        self.protocol_map = {
            'O': 'Oulu',
            'C': 'casia',
            'I': 'replay',
            'M': 'MSU',
            'D': '3DMAD',
            'H': 'HKBUv1+',
            'U': 'casia_3d',
            'paper_glasses' : 'paper_glasses_RGB',
            'mask_silicone' : 'mask_silicone_RGB',
            'funny_eyes' : 'funny_eyes_RGB',
        }

        all_live_images = []
        all_spoof_images = []

        # data_path = self.root_dir / 'domain-generalization'

        for p in protocol:
            if p not in self.protocol_map:
                print(f"Warning: Protocol '{p}' is not recognized. Skipping.")
                continue
            
            base_name = self.protocol_map[p]
            if p == 'D' or p == 'H' or p == 'U':
                data_path = self.root_dir / "3Dmask"
                live_path = data_path / f"{base_name}_images_live.npy"
                spoof_path = data_path / f"{base_name}_images_spoof.npy"
            elif p in ['paper_glasses', 'mask_silicone', 'funny_eyes']:
                data_path = self.root_dir / "padisi" / "partial" / "test"
                live_path = data_path / "live_RGB.npy"
                spoof_path = data_path / f"{base_name}.npy"
            else:
                data_path = self.root_dir / "domain-generalization"
                live_path = data_path / f"{base_name}_images_live.npy"
                spoof_path = data_path / f"{base_name}_images_spoof.npy"

            if live_path.exists():
                live_data = np.load(live_path)
                all_live_images.append(live_data)
            else:
                print(f"Warning: File not found at {live_path}")

            if spoof_path.exists():
                spoof_data = np.load(spoof_path)
                all_spoof_images.append(spoof_data)
            else:
                print(f"Warning: File not found at {spoof_path}")

        if not all_live_images and not all_spoof_images:
            raise RuntimeError("No data loaded. Check root_dir and protocol.")

        # Concatenate all loaded images from the specified protocols
        live_images = np.concatenate(all_live_images, axis=0) if all_live_images else np.array([])
        spoof_images = np.concatenate(all_spoof_images, axis=0) if all_spoof_images else np.array([])
        
        # Create labels: 0 for real/live, 1 for fake/spoof (to match HierarchicalDataLoader)
        # This is the reverse of the original FAS_Dataset for consistency.
        live_labels = np.zeros(len(live_images), dtype=np.int64)
        spoof_labels = np.ones(len(spoof_images), dtype=np.int64)
        
        if self.live_only:
            self.total_images = live_images
            self.total_labels = live_labels
        elif len(live_images) > 0 and len(spoof_images) > 0:
            # Combine live and spoof data
            self.total_images = np.concatenate((live_images, spoof_images), axis=0)
            self.total_labels = np.concatenate((live_labels, spoof_labels), axis=0)
        elif len(live_images) > 0:
            self.total_images = live_images
            self.total_labels = live_labels
        else:
            self.total_images = spoof_images
            self.total_labels = spoof_labels


    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset.
        
        Returns:
            tuple: (primary_image_tensor, dummy_scm_tensor, labels_tensor)
                   to maintain compatibility with the hierarchical dataloader.
        """
        img_data = self.total_images[idx]
        binary_label = self.total_labels[idx]

        # Convert numpy array to PIL Image with proper preprocessing
        try:
            # Handle problematic array shapes and data types
            processed_img = img_data.copy()
            
            # Remove singleton dimensions (e.g., (1,1,3) -> (1,3) -> (3,))
            processed_img = np.squeeze(processed_img)
            # print(processed_img)
                        
            # Convert data type to uint8 if it's float
            if processed_img.dtype == np.float32 or processed_img.dtype == np.float64:
                if processed_img.max() < 0.1:
                    processed_img *= 255.0  # Scale to [0, 1] if in [0, 0.0039]. For visualization only!!! Impact performance
                # Assume values are in range [0,1] and scale to [0,255]
                if processed_img.max() <= 1.0:
                    processed_img = (processed_img * 255).astype(np.uint8)
                else:
                    # Values might already be in [0,255] range but stored as float
                    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
                    # processed_img = (processed_img / 255).astype(np.uint8) # Performance improvement for DHU
            elif processed_img.dtype != np.uint8:
                # Convert other integer types to uint8
                processed_img = processed_img.astype(np.uint8)
            
            primary_image = Image.fromarray(processed_img)
            
        except Exception as e:
            print(f"Error converting numpy array to image at index {idx}. Error: {e}")
            print(f"Original shape: {img_data.shape}, dtype: {img_data.dtype}")
            print(f"Value range: [{img_data.min():.4f}, {img_data.max():.4f}]")
            # Fallback to the next item in the dataset
            return self.__getitem__((idx + 1) % len(self))

        # Apply transformations to the image
        if self.transform:
            primary_image_tensor = self.transform(primary_image)
        else:
            # Apply a default transformation if none is provided - resize to 224x224
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            primary_image_tensor = default_transform(primary_image)
                    
        label = binary_label
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return primary_image_tensor, label_tensor
