import os, glob
import numpy as np
import tifffile as tif
import mrcfile
import torch

from core.utils.utils import make_dir

def get_overlap_pad(patch_overlap, device):
    overlap_pad = patch_overlap
    overlap_pad = [[0, 0] , [overlap_pad[0]//2, overlap_pad[0]//2],[overlap_pad[1]//2, overlap_pad[1]//2]]
    overlap_pad = torch.tensor(overlap_pad, device=device).unsqueeze(0).int()
    return overlap_pad

def augment_dataset(imgs):
    if torch.rand(1, device=imgs.device) < 0.5:
        imgs = torch.flip(imgs, [-1])
    if torch.rand(1, device=imgs.device) < 0.5:
        imgs = torch.flip(imgs, [-2])
    if torch.rand(1, device=imgs.device) < 0.5:
        imgs = torch.flip(imgs, [-3])
    return imgs

def unpad3D(array, pad):
    return array[...,pad[0][0]:-pad[0][1] or None, pad[1][0]:-pad[1][1] or None, pad[2][0]:-pad[2][1] or None]

def normalize_imgs(imgs, params):
    imgs = (imgs - params["min"]) / (params["max"] - params["min"])
    imgs = 2*imgs - 1
    return imgs

def denormalize_imgs(imgs, params):
    imgs = (imgs+1)/2
    imgs = imgs*(params["max"] - params["min"]) + params["min"]
    return imgs

def data_extension_to_format(extension):
    if extension == ".tif":
        return "tif_file"
    elif extension == ".mrc":
        return "mrc_file"
    elif extension == ".rec":
        return "rec_file"
    else:
        raise NotImplementedError(f"File extension {extension} is not currently supported")

def data_format_to_extension(format):
    if format == "tif_file" or format == "tif_sequence":
        return ".tif"
    elif format == "mrc_file":
        return ".mrc"
    elif format == "rec_file": 
        return ".rec"
    else:
        raise NotImplementedError(f"Data format {format} is not currently supported")

def get_data_format(path):
    if os.path.isfile(path):
        extension = os.path.splitext(path)[1]
        return data_extension_to_format(extension)
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.tif"))
        if len(files)>0:
            return "tif_sequence"
        else:
            raise NotImplementedError(f"Only sequences of tif files are currently supported")
    else:
        raise ValueError(f"Path {path} is invalid")

class Virtual3DStack:
    def __init__(self, path):
        filelist = sorted(glob.glob(os.path.join(path, "*.tif")))
        self.slices = [tif.TiffFile(f) for f in filelist]
        
    def __getitem__(self, index):
        if isinstance(index, tuple):
            z_slice, y_slice, x_slice = index
            return np.stack([self.slices[z].asarray(out='memmap')[y_slice, x_slice] for z in range(*z_slice.indices(len(self.slices)))])
        else:
            return self.slices[index].asarray(out='memmap')

    @property
    def shape(self):
        z = len(self.slices)
        y, x = self.slices[0].asarray(out='memmap').shape
        return (z, y, x)

    @property
    def dtype(self):
        return self.slices[0].asarray(out='memmap').dtype

    def min(self):
        return min([slice.asarray(out='memmap').min() for slice in self.slices])

    def max(self):
        return max([slice.asarray(out='memmap').max() for slice in self.slices])

    def mean(self):
        return sum([slice.asarray(out='memmap').mean() for slice in self.slices]) / len(self.slices)
    
def memmap_data(path, data_format):
    extra_params = None
    
    if data_format=="tif_file":
        data = tif.memmap(path)
    elif data_format=="mrc_file" or data_format=="rec_file":
        memmap = mrcfile.mmap(path, mode='r')
        data = memmap.data
        extra_params = {
            "voxel_size": memmap.voxel_size.copy()
        }
    elif data_format=="tif_sequence":
        data = Virtual3DStack(path)
        
    return data, extra_params

def get_metadata(data, data_format, extra_params=None):
    metadata = {
        "format": data_format,
        "shape": data.shape,
        "dtype": data.dtype,
        "mean": data.mean().astype('float'),
        "min": data.min().astype('float'),
        "max": data.max().astype('float'),
    }

    if extra_params is not None:
        for key, value in extra_params.items():
            metadata[key] = value

    return metadata

def get_data(path):

    data_format = get_data_format(path)

    data, extra_params = memmap_data(path, data_format)

    metadata = get_metadata(data, data_format, extra_params)
    
    return data, metadata
    
def save_data(path, name, data, metadata, output_format='same'):

    if output_format=='same':
        output_format = metadata["format"]
        
    extension = data_format_to_extension(output_format)

    if output_format=="tif_sequence":
        zfill = len(str(data.shape[0]))
        save_dir = os.path.join(path, name)
        make_dir(save_dir)
        for i in range(data.shape[0]):
            slice_2d = data[i, :, :]
            save_path = os.path.join(save_dir, f'slice_{str(i).zfill(zfill)}'+extension)
            tif.imwrite(save_path, slice_2d)
    else:
        save_path = os.path.join(path, name+extension)
        if output_format=="tif_file":
            tif.imwrite(save_path, data)
        elif output_format=="mrc_file" or output_format=="rec_file":
            with mrcfile.new(save_path, overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = metadata["voxel_size"].copy()
                mrc.update_header_from_data()
                mrc.update_header_stats()





