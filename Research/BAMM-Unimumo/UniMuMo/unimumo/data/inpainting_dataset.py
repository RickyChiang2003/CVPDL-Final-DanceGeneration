import torch
from torch.utils.data import Dataset
import os
import glob
import re
from collections import defaultdict

class InpaintingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.pth')))
        
        # Group files by prefix
        # Filename format: {prefix}_beat{start}_beat{end}.pth
        self.file_map = defaultdict(dict)
        pattern = re.compile(r"(.*)_beat(\d+)_beat(\d+)\.pth")
        
        for f in self.files:
            basename = os.path.basename(f)
            match = pattern.match(basename)
            if match:
                prefix = match.group(1)
                start_beat = int(match.group(2))
                self.file_map[prefix][start_beat] = f
        
        # Find pairs (start, start+8) to form 24-beat sequences
        self.samples = []
        for prefix, beats in self.file_map.items():
            sorted_beats = sorted(beats.keys())
            for b in sorted_beats:
                if (b + 8) in beats:
                    self.samples.append({
                        'file1': beats[b],
                        'file2': beats[b+8]
                    })
        
        print(f"Found {len(self.files)} files.")
        print(f"Constructed {len(self.samples)} samples of 24 beats (8+8+8).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data1 = torch.load(sample['file1'])
        data2 = torch.load(sample['file2'])
        
        # File 1: Beats B to B+16
        # File 2: Beats B+8 to B+24
        # We want Beats B to B+24
        # Take First 8 beats from File 1, and all 16 beats from File 2 (which are B+8 to B+24)
        # Or take First 8 from File 1, Next 8 from File 1 (overlap), Last 8 from File 2?
        # Better: Take First 8 from File 1, and All from File 2.
        
        m1 = data1['music_code'].squeeze(0) # [K, T1]
        mo1 = data1['motion_code'].squeeze(0) # [K, T1]
        
        m2 = data2['music_code'].squeeze(0) # [K, T2]
        mo2 = data2['motion_code'].squeeze(0) # [K, T2]
        
        # Ensure lengths are consistent for splitting
        # 16 beats approx T1. 8 beats approx T1/2.
        half1 = m1.shape[-1] // 2
        
        # Stitch
        # Part 1: 0 to half1 (First 8 beats)
        m_part1 = m1[..., :half1]
        mo_part1 = mo1[..., :half1]
        
        # Part 2: All of File 2 (Beats 9-24)
        m_part2 = m2
        mo_part2 = mo2
        
        music_code = torch.cat([m_part1, m_part2], dim=-1)
        motion_code = torch.cat([mo_part1, mo_part2], dim=-1)
        
        # Ensure equal length (should be by construction, but safety check)
        L = min(music_code.shape[-1], motion_code.shape[-1])
        music_code = music_code[..., :L]
        motion_code = motion_code[..., :L]
        
        text_cond = "<separation>"
        
        return {
            "music_code": music_code,
            "motion_code": motion_code,
            "text": text_cond
        }
