from functools import partialmethod
import logging
import pathlib
from os.path import join
from warnings import warn
import shutil
import warnings
import os, sys

from tqdm import tqdm

def dict_list_add(output_dict, key, value):
    '''
    Given a dict containing lists, add to a value under a given key, creating the key if it does not exist.
    '''
    if key not in output_dict.keys():
        output_dict[key] = 0
    output_dict[key] += value
    return output_dict

class SuppressPrint():
    '''
    use as 
        with SuppressPrint():

    * https://stackoverflow.com/a/45669280 (print)
    * https://stackoverflow.com/a/67238486 (tqdm)
    * https://stackoverflow.com/a/20251235 (logger)
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        logging.disable(logging.CRITICAL)
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        logging.disable(logging.NOTSET)
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False) # type: ignore


def repo_dir(*paths):
    '''
    input: a path relative to the parent folder of the project, output: a path relative to the current cwd
    '''
    project_root_path = str(pathlib.Path(__file__).parent.resolve()).split("/")[:-1] # root of the project: parent dir of this file

    cwd = os.getcwd().split("/")

    if cwd[-1] in project_root_path:
        # the deepest folder of the cwd is in the path to the project root -> cwd is (at or) above the project root
        cwd_index = project_root_path.index(cwd[-1]) # where in the (absolute) project path is the cwd?
        cwd_path = project_root_path[cwd_index + 1:] # what to append to the cwd to get to the project root
        return join(*cwd_path + list(paths))
    else:
        # deepest cwd folder is not in the project root path -> cwd is in a subfolder of the project. Jupyter (sometimes) does that, and we usually don't want it to.
        warn(f"Changing cwd from {join(*cwd)} to {join(*project_root_path)}")
        os.chdir("/" + join(*project_root_path))
        return(join(*paths))    

def data_dir(*paths):
    '''
    same as repo_dir, but ending up one layer above the repo dir - this is where dataset downloads and pretrained models can be located.
    Since cwd might be "below" the data dir (when it is in the repo dir) and we do not want to change that (as in repo_dir), we are using the absolute path
    '''
    path = os.path.abspath(pathlib.Path(__file__).parent.resolve())
    path = str(path).split("/")[:-2]
    return "/" + join(*path + list(paths))

def exp_name(exp_id):
    '''
    given the id of an experiment, return the full name of the folder
    '''
    for folder_name in next(os.walk(repo_dir("experiments")))[1]:
        if folder_name.startswith(f"{exp_id}_") or folder_name == exp_id: # "_" to prevent matching 0_0a to 0_0
            return folder_name
        
    warnings.warn(f"Did not find an experiment folder for id {exp_id}")
    return exp_id

def clean_saved_model_weights(path, model_name=None, keep_last=False, except_epoch=[]):
    '''
    Deletes every but the final saved model in all subfolder. Ignores named saved models.
    '''
    counter = 0
    for exp_folder in next(os.walk(path))[1]:
        if not os.path.isdir(join(path, exp_folder, "model_weights_local")):
            continue

        final_epoch_file = ""
        final_epoch = 0

        for file in next(os.walk(join(path, exp_folder, "model_weights_local")))[2]:
            if file.endswith(".pth") and file.split("_")[0] == model_name or model_name == None:
                epoch = file.split("_")[-1].replace(".pth", "")

                if epoch.isdigit() or epoch == "-1":
                    epoch = int(epoch)
                    if keep_last:
                        # keep track of the highest number, delete only if the highest number is higher than the current epoch.
                        if final_epoch >= epoch and epoch not in except_epoch:
                            os.remove(os.path.join(path, exp_folder, "model_weights_local", file))
                            counter += 1
                        else:
                            if len(final_epoch_file) > 0 and final_epoch not in except_epoch:
                                os.remove(final_epoch_file)
                                counter += 1
                            final_epoch = epoch
                            final_epoch_file = os.path.join(path, exp_folder, "model_weights_local", file)
                    else:
                        if epoch not in except_epoch:
                            os.remove(os.path.join(path, exp_folder, "model_weights_local", file))
                            counter += 1

    print(f"Deleted {counter} weight files.")

def delete_saved_model_weights(path):
    for exp_folder in next(os.walk(path))[1]:
        if os.path.exists(os.path.join(path, exp_folder, "model_weights_local")):
            shutil.rmtree(os.path.join(path, exp_folder, "model_weights_local"))
        # if os.path.exists(os.path.join(path, exp_folder, "tboard")):
        #     shutil.rmtree(os.path.join(path, exp_folder, "tboard"))

def rename_all_folders_in_dir(dir, rename_function):
    for file in os.listdir(dir):
        # print(rename_function(file))
        os.rename(join(dir, file), join(dir, rename_function(file)))

if __name__ == "__main__":
    clean_saved_model_weights(repo_dir("experiments", exp_name("00_0"), "output"), model_name="model", except_epoch=[800, 2000, *range(0, 4000, 200)], keep_last=True)

    # def rename(file):
    #     # return file + "_radius0.5"
    #     parts = file.split("_")
    #     return "0_" + "_".join(parts[1:])

    # rename_all_folders_in_dir(repo_dir("experiments", exp_name("11_6"), "output"), rename)
