#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Jimenez
"""

import os
import shutil
import time

class GenerateFiles(object):
    """
    GenerateFiles is a class used for file/directory creation and removal.
    Its main function 'make_directories' will generate all necessary directories 
    if they do not already exist.
    """

    def __init__(self, dataset, output_path=None):
        """
        Args:
            dataset (str): file name for the cluster catalog that will used.
                        Options are 'planck_z', 'planck_z_no-z', 'MCXC', 'RM30', 'RM50'.
            output_path (str, optional): Path to output directory where main results will be saved.
                        If None, xcluster directory path will be used (recommended). Defaults to None.
        """

        self.path = os.getcwd() +'/'
        self.dataset = dataset 
        self.temp_path = self.path + 'to_clean/'
        self.output_name = time.strftime("/%Y-%m-%d")
        if output_path is None:
            self.output_path = self.path
        else:
            self.output_path = output_path


    def make_directory(self, path_to_file):
        """Tries to create a directory in path_to_file.

        Args:
            path_to_file (str): Path where the directory will be created.
        """

        try:
            os.mkdir(path_to_file)
        except OSError:
            pass
        else:
            print ("Successfully created the directory %s " % path_to_file)


    def make_directories(self, output=False, replace=False):
        """Creates temporary directory and all output related directories needed for xcluster to work properly.
        If directories already exist, function does nothing, except if replace=True.

        Args:
            output (bool, optional): If True, will create necessary directories for the output directory. Defaults to False.
            replace (bool, optional): If True, will rename latest daily output directory to today's date. Defaults to False.
        """

        if output == False:
            self.make_directory(self.temp_path)

        elif output == True:
            self.make_directory(self.output_path + 'output/')
            self.make_directory(self.output_path + 'tf_saves/')
            self.make_directory(self.output_path + 'catalogs/')
            self.make_directory(self.output_path + 'datasets/')
            self.make_directory(self.output_path + 'datasets/'+ self.dataset)
            self.make_directory(self.output_path + 'healpix/')
            self.make_directory(self.output_path + 'healpix/figures/')
            self.make_directory(self.output_path + 'healpix/figures/PSZ2')
            if replace == True:
                last_dir = '/' + os.listdir(self.output_path + 'output/' + self.dataset)[-1]
                if last_dir != self.output_name:
                    os.rename(self.output_path + 'output/' + self.dataset + last_dir, self.output_path + 'output/' + self.dataset + self.output_name)
                else:
                    self.make_directory(self.output_path + 'output/' + self.dataset)
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name)
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name + '/files')
            self.make_directory(self.output_path + 'output/' + self.dataset + self.output_name + '/figures')



    def remove_files_from_directory(self, directory):
        """Removes files for a given directory.

        Args:
            directory (str): directory path.
        """

        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def clean_temp_directories(self):
        """Removes files from temporary directory.
        """

        if os.path.exists(self.temp_path) and os.path.isdir(self.temp_path):
            if not os.listdir(self.temp_path):
                print("Directory %s is empty"%(self.temp_path))
            else:
                self.remove_files_from_directory(self.temp_path)
                print("Successfully removed the directory %s " % (self.temp_path))
        else:
            print("Directory %s does not exist"%(self.temp_path))


    def is_directory_empty(self, path_to_dir):
        """Checks if a given directory is empty or not.

        Args:
            path_to_dir (str): path to the directory

        Returns:
            bool: True if directory is empty, False if not.
        """

        if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
            if not os.listdir(path_to_dir):
                print("Directory %s is empty"%path_to_dir)
                return True
            else:
                print("Directory %s is not empty"%path_to_dir)
                return False
        else:
            print("Directory %s don't exists"%path_to_dir)
