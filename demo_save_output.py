import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Create output directory
    output_dir = 'output_flows'
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # Convert flow to numpy
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            
            # Visualize flow
            flo_viz = flow_viz.flow_to_image(flo)
            
            # Save visualization
            output_file = os.path.join(output_dir, f'flow_{idx:04d}.png')
            cv2.imwrite(output_file, flo_viz[:, :, [2,1,0]])
            
            # Save raw flow (optional)
            flow_file = os.path.join(output_dir, f'flow_{idx:04d}.npy')
            np.save(flow_file, flo)
            
            print(f'✓ Saved: {output_file}')
            print(f'  Input: {os.path.basename(imfile1)} -> {os.path.basename(imfile2)}')

    print(f'\n🎉 All results saved to: {output_dir}/')
    print(f'   Total files: {len(images)-1} flow visualizations')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
