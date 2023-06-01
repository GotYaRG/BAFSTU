import os
from Gut_fullness_algorithm import *
from plankton_cnn.run_classification_model import run_classifier

import warnings

# Filter out the TFA warning to keep the command line clear
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow_addons')

import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image

from tensorflow_addons.optimizers import AdamW


"""
GUTS

"""


def handle_drop(event):
    filepath = event.data
    # Set the file path to the variable
    file_path_var.set(filepath)
    # Close dialog box once file has been submitted
    root.destroy()

def open_drag_drop_menu():
    global root
    root = TkinterDnD.Tk()

    global file_path_var
    file_path_var = tk.StringVar()

    # Create a label to act as the drop target
    drop_label = tk.Label(root, text="Drag and drop an image file here")
    drop_label.pack(pady=20)

    # Register the drop target and bind the handler function
    drop_label.drop_target_register(DND_FILES)
    drop_label.dnd_bind('<<Drop>>', handle_drop)

    root.mainloop()

    # Return the file path as a string
    return file_path_var.get()


def asses_single_image():
    print("A small dialog box should have opened on your computer.\n"
          "The icon in your taskbar should look like a feather.\n"
          "Please drag and drop the image you want to analyze into this box.\n"
          "After the image has been analyzed, the drop box will open again to analyze another image.\n")
    single_image_path = open_drag_drop_menu()
    single_image_path = single_image_path[1:-1]
    run_gfa("", False, False, False, single_image_path)
    stop = input("Please enter 'stop' to stop analyzing images or 'start' to analyze another image:\n")
    while stop != "stop":
        single_image_path = open_drag_drop_menu()
        single_image_path = single_image_path[1:-1]
        run_gfa("", False, False, False, single_image_path)
        stop = input("Please enter 'stop' to stop analyzing images or 'start' to analyze another image:\n")


def main_menu():
    print("Welcome to GUTS!\n"
          "GUTS is an application used for analyzing images of Noctiluca scintillans\n")
    input_string = "Would you like to assess a single image or a collection of images?\n" \
                   "1. Single image\n" \
                   "2. Collection of images\n"
    choice = input(input_string)
    while choice not in ["1", "2"]:
        print("That was not a valid option, pleae enter '1' or '2' based on your choice.")
        choice = input(input_string)
    return choice


def full_run_menu():
    """
    Asks the user what data they would want to be stored by GUTS.
    The image classifications and gut fullness outputs need to be generated
    whether the user wants this or not, however, by not selecting to save it,
    it will be deleted at the end of the run.

    :return: A dictionary with the options selected by the user.
    """
    options_dict = {"1": False, "2": False, "3": False, "4": False}
    options = []
    input_string = "Please select what actions you want to perform with your images:\n" \
                   "1. Classify images into relevant categories\n" \
                   "2. Isolate and store cell surface\n" \
                   "3. Isolate and store cell contents\n" \
                   "4. Caclulate Gut Fullness\n" \
                   "Enter 'start' into the prompt to perform the selected actions.\n"
    while len(options) == 0:
        selection = ""
        while selection.lower() != "start":
            if len(options) == 0:
                selection = input(input_string)
            else:
                selection = input(f"{input_string}Current selection:\n{', '.join(options)}\n")
            if selection in options_dict.keys() and selection not in options:
                options.append(selection)
            elif selection != "start":
                print('\033[93m' + "Selection invalid, option not present or already selected." + '\033[0m')
        if len(options) == 0:
            print('\033[93m' + "No options were selected, please try again." + '\033[0m')
    if "1" not in options:
        print("Option 1 was not selected, however it is required for your selected option(s) and will be performed.")
    for option in options:
        options_dict[option] = True
    return options_dict


def is_filename_valid(filename):
    """
    Checks the validity of a given filename, particularly to test for
    restricted characters that cannot be used in Windows filenames.

    :param filename: The run name the user gave as input, which will be used
           in filenames.

    :return: A boolean indicating whether or not the filename is valid.
    """
    try:
        with open(filename, 'w') as temp_file:
            pass
        os.remove(filename)
        return True
    except OSError:
        print('\033[93m' + "Name invalid, one or more restricted characters was used.\nPlease enter another name"
              + '\033[0m')
        return False


def get_run_name():
    """
    Asks the user to enter a name for the current run of GUTS, if the name
    is valid and usable for filenames, it is returned to the main function.

    :return: A valid name for the run that can be used for filenames.
    """
    valid_name = False
    run_name = ""
    while not valid_name:
        run_name = input("All files that are used and created by GUTS refer to the code or name for this run.\n"
                         "Please enter the name for this run of GUTS:\n")
        valid_name = is_filename_valid(run_name)
    return run_name


def main():
    choice = main_menu()
    if choice == "1":
        asses_single_image()
    elif choice == "2":
        options = full_run_menu()
        run_name = get_run_name()
        if options["1"] and not options["2"] and not options["3"] and not options["4"]:
            run_classifier(run_name)
        else:
            run_classifier(run_name)
            run_gfa(run_name, options["2"], options["3"], options["4"])


if __name__ == '__main__':
    main()

