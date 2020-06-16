import os

rootdir = os.getcwd()+'\\test_browse\\downloads\\'


def rename_folder_structure(rootdir):
    for index, dirs in enumerate(next(os.walk(rootdir))[1]):
        os.rename(os.path.join(rootdir, dirs), os.path.join(
            rootdir, 'Family_' + str(index+1)))


rename_folder_structure(rootdir)
