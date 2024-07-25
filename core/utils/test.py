import numpy as np
import tifffile
import mrcfile
import torch


def mrc_to_rec(mrc_path, rec_path):
    with mrcfile.open(mrc_path, mode="r") as mrc:
        data = mrc.data
        header = mrc.header

        # Implement REC format specifications here
        with open(rec_path, "wb") as rec_file:
            # Write header information if needed
            # Write data
            rec_file.write(data.tobytes())


def main():
    input_path = "../../rotacell_grid1_TS09_ctf_6xBin.rec"
    # output_path='../../output_int16.rec'
    # convert_rec_file_to_type(input_path=input_path, output_path=output_path, dtype=np.dtype(np.int16))
    mrc_to_rec(input_path, "output_file.rec")
