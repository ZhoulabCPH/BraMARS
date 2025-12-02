from histolab.slide import Slide
from histolab.tiler import GridTiler
import os


def tiling_WSI(WSI_root):
    grid_tiles_extractor = GridTiler(
        tile_size=(224, 224),
        level=0,
        check_tissue=True,  # default
        tissue_percent=60,
        pixel_overlap=0,  # default
        prefix="",  # save tiles in the "grid" subdirectory of slide's processed_path
        suffix=".png"  # default
    )
    WSI_path = WSI_root.split('.')[0]
    output_path = os.path.join(WSI_path, 'patch')
    if os.path.exists(output_path):
        # shutil.rmtree(output_path + slide.split(".")[0])
        print("The folder already exists and has been deleted")
    else:
        os.mkdir(output_path)
        print("The folder does not exist, it has been created")
        slide1 = Slide(WSI_root, output_path)
        # # Get the current magnification
        grid_tiles_extractor.extract(slide1)
        print('tiler_finish!')
