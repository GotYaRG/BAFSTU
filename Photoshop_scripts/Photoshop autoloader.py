import photoshop.api as ps
from photoshop import Session
from skimage import io
import pickle


def get_labeled_image_names():
    """
    This function imports a dataframe from a pickle with manually labeled images.
    There are a number of labels present in the dataframe, but in this case only
    the "full_digesting" and "partial_digesting" labels are of interest.
    :return: A dataframe with image names of images where digestion is occurring.
    """
    infile = open("manual_classifier/img_viewer/save_files/first_1k_redo", "rb")
    new_df = pickle.load(infile)
    infile.close()
    is_digesting_full = new_df["label"] == "full_digesting"
    is_digesting_partial = new_df["label"] == "partial_digesting"
    digesting_df = new_df[is_digesting_full | is_digesting_partial]
    return digesting_df


def generate_ps_files(project_path, img_lib_path, df):
    """
    Generates a Photoshop file with three layers for all images in a given dataframe.
    The first layer is the original image, the second and third layer (contents, surface)
    are added for the purpose of tracing over the images with a drawing tablet.
    :param project_path: The filepath to the project folder you are working in.
    :param img_lib_path: The filepath to the folder where all the relevant
                         images are stored.
    :param df: A dataframe with at least the file names of the relevant images under
               a column called "image_name".
    """
    for img_name in df["image_name"]:
        # Load the image
        img = io.imread(f"{img_lib_path}{img_name}")
        # Open a PS Session to create the PS File
        with Session(auto_close=True) as ps_file:
            ps_file.app.preferences.rulerUnits = ps.Units.Pixels
            # Make sure the PS File's dimensions are equal to that of the image
            doc = ps_file.app.documents.add(img.shape[1], img.shape[0], )
            # Set relevant options for the PS File, only paths ought to be altered here
            desc = ps_file.ActionDescriptor
            desc.putPath(ps_file.app.charIDToTypeID("null"), f"{project_path}{img_lib_path}{img_name}")
            event_id = ps_file.app.charIDToTypeID("Plc ")
            ps_file.app.executeAction(ps_file.app.charIDToTypeID("Plc "), desc)
            # Save PS File options
            options = ps.PhotoshopSaveOptions()
            # Add and name extra layers
            contents_layer = doc.artLayers.add()
            contents_layer.name = "contents"
            surface_layer = doc.artLayers.add()
            surface_layer.name = "surface"
            # Save PS File with name and location of choice
            doc.saveAs(f"{project_path}Data/photoshop/files/{img_name[:-4]}", options, True)


def main():
    project_path = "C:/Users/GotYa/OneDrive - Hogeschool Leiden/School/BAFSTU/Python/"
    img_lib_path = "manual_classifier/img_viewer/noctiluca_sample_Scharendijke2021/"
    labeled_img_df = get_labeled_image_names()
    generate_ps_files(project_path, img_lib_path, labeled_img_df)


if __name__ == "__main__":
    main()
