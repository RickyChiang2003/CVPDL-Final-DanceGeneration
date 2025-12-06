import torch
from torch.utils.data import Dataset
import os
import glob
import re
from collections import defaultdict

class ProcessedHouseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.pth')))
        
        # Group files by subdir
        # Filename format: {subdir}_beat{start}_beat{end}.pth
        # e.g. gHO_sFM_cAll_d19_mHO0_ch01_windows_beat1_beat8.pth
        self.file_map = defaultdict(dict)
        pattern = re.compile(r"(.*)_beat(\d+)_beat(\d+)\.pth")
        
        for f in self.files:
            basename = os.path.basename(f)
            match = pattern.match(basename)
            if match:
                prefix = match.group(1)
                start_beat = int(match.group(2))
                self.file_map[prefix][start_beat] = f
        
        # Construct 24-beat samples (8+8+8)
        self.samples = []
        for prefix, beats in self.file_map.items():
            sorted_beats = sorted(beats.keys())
            # We look for b, b+8, b+16
            for b in sorted_beats:
                if (b + 8) in beats and (b + 16) in beats:
                    self.samples.append({
                        'file1': beats[b],
                        'file2': beats[b+8],
                        'file3': beats[b+16]
                    })
        
        print(f"Found {len(self.files)} files.")
        print(f"Constructed {len(self.samples)} samples of 24 beats (8+8+8).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        d1 = torch.load(sample['file1'])
        d2 = torch.load(sample['file2'])
        d3 = torch.load(sample['file3'])
        
        # Concatenate codes
        # music_code: [1, K, T]
        # motion_code: [1, K, T]
        
        m1 = d1['music_code'].squeeze(0)
        mo1 = d1['motion_code'].squeeze(0)
        
        m2 = d2['music_code'].squeeze(0)
        mo2 = d2['motion_code'].squeeze(0)
        
        m3 = d3['music_code'].squeeze(0)
        mo3 = d3['motion_code'].squeeze(0)
        
        music_code = torch.cat([m1, m2, m3], dim=-1)
        motion_code = torch.cat([mo1, mo2, mo3], dim=-1)
        
        # Ensure equal length
        L = min(music_code.shape[-1], motion_code.shape[-1])
        music_code = music_code[..., :L]
        motion_code = motion_code[..., :L]
        
        text_cond = "<separation>"
        
        return {
            "music_code": music_code,
            "motion_code": motion_code,
            "text": text_cond
        }


















