import photoshop.api as ps
from photoshop import Session
import os


def hide_all_layers(layers):
    """
    Quick function that receives the layers of a Photoshop file and sets their
    visibility to False.
    :param layers: The layers of a Photoshop file.
    """
    for layer in layers:
        layer.visible = False


def export_layers(project_path):
    """
    Opens all Photoshop files in a given directory one by one, and exports all
    layers in the file. The exports are saved as PNGs with the exact same dimensions
    as the image in the Photoshop file.
    :param project_path: The filepath to the project folder you are working in.
    """
    for file_name in os.listdir("Data/photoshop/files/"):
        # Open a PS Session to open the PS File
        with Session(f"{project_path}Data/photoshop/files/{file_name}", action="open", auto_close=True) as ps_file:
            # Set PS File to 'Active' to be able to export layers
            doc = ps_file.active_document
            # Set export format to PNG
            options = ps.PNGSaveOptions()
            # Obtain layers in PS File and export them one by one
            layers = doc.artLayers
            for layer in layers:
                hide_all_layers(layers)
                layer.visible = True
                # Added if clause, mainly to prevent default "Background" layer
                # from being exported
                if layer.name in ["contents", "surface"]:
                    # Save PNG of layer with name and location of choice
                    doc.saveAs(f"{project_path}Data/photoshop/exports/{file_name[:-4]}_{layer.name}", options, True)


def main():
    project_path = "C:/Users/GotYa/OneDrive - Hogeschool Leiden/School/BAFSTU/Python/"
    export_layers(project_path)


if __name__ == "__main__":
    main()
