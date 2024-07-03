from pathlib import Path
from distutils.dir_util import copy_tree
import config as c
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import itertools
import numpy as np

ipcode_size_crop = {(450, 1215): (0, 0, 1215, 397),  # 35800 SIDE A
                    (457, 1224): (0, 0, 1224, 384),  # 35800 SIDE B
                    (499, 1383): (0, 0, 1383, 425),  # 29847 A
                    (482, 1365): (0, 0, 1365, 413),  # 29847 B
                    (493, 1357): (0, 0, 1357, 442),  # 25016 A1
                    (484, 1354): (0, 0, 1354, 419),  # 25016 B1
                    (507, 1297): (0, 0, 1297, 429),
                    (497, 1289): (0, 0, 1289, 416),
                    (501, 1297): (0, 0, 1297, 420),
                    (501, 1301): (0, 0, 1301, 420),
                    (487, 1291): (0, 0, 1291, 418),
                    (476, 1312): (0, 0, 1312, 403),  # 24788 A
                    (464, 1331): (0, 0, 1331, 390),  # 24788 B
                    (450, 1234): (0, 0, 1234, 383),  # 24787 A
                    (456, 1230): (0, 0, 1230, 395),  # 24787 B
                    (559, 1607): (0, 0, 1607, 531),  # 22059 A
                    (543, 1597): (0, 0, 1597, 518),  # 22059 B
                    (714, 1900): (0, 0, 1900, 622),  # 21532 A
                    (680, 1918): (0, 0, 1918, 615),  # 21532 B
                    (630, 1877): (0, 0, 1877, 617),  # 21373 A
                    (631, 1892): (0, 0, 1892, 586),  # 21373 B
                    (431, 1169): (0, 0, 1169, 353),  # 20410 in mezzo al bianco
                    (443, 1172): (0, 0, 1172, 362),  # 20410
                    (434, 1168): (0, 0, 1168, 355),  # 20410
                    }

def copy_element_from_directory(fromDirectory,
                                toDirectory,
                                folder,
                                anomaly_df,
                                borderline_df,
                                ok_df,
                                tag = None,
                                excluded_types = ["2D.FE.DR.1.A_001", "2D.FE.DR.1.A_002", "2D.FE.DR.1.B_001", "2D.FE.DR.1.B_002"],
                                image_format=['.png'],
                                useTagAsCompleteName = False,
                                includeBorderlineInAnomaly = False):
    for root, dirs, files in os.walk(fromDirectory):
        for filename in files:
            # I use absolute path, case you want to move several dirs.
            source = os.path.join( os.path.abspath(root), filename )
            toDirectory_temp = toDirectory

            # Reset new_name
            new_name = None

            # Separate base from extension
            base, extension = os.path.splitext(filename)
            if base not in excluded_types and extension in image_format:
                if ((anomaly_df['Folder'] == folder) & (anomaly_df['Type'] == filename)).any():
                    toDirectory_temp += "/test/anomaly/"
                    # Initial new name
                    if useTagAsCompleteName:
                        new_name = os.path.join(toDirectory_temp, tag + extension)
                    else:
                        new_name = os.path.join(toDirectory_temp, base + "_" + tag + extension)
                elif ((borderline_df['Folder'] == folder) & (borderline_df['Type'] == filename)).any():
                    if includeBorderlineInAnomaly:
                        toDirectory_temp += "/test/anomaly/"
                    else:
                        toDirectory_temp += "/train/good/"
                    # Initial new name
                    if useTagAsCompleteName:
                        new_name = os.path.join(toDirectory_temp, tag + extension)
                    else:
                        new_name = os.path.join(toDirectory_temp, base + "_" + tag + extension)
                # We must control even ok images because there are some images that are not present in the csv
                # Images not present in the csv are badly cropped
                elif ((ok_df['Folder'] == folder) & (ok_df['Type'] == filename)).any():
                    toDirectory_temp += "/train/good/"
                    # Initial new name
                    if useTagAsCompleteName:
                        new_name = os.path.join(toDirectory_temp, tag + extension)
                    else:
                        new_name = os.path.join(toDirectory_temp, base + "_" + tag + extension)
                if(new_name is not None):
                    shutil.copy(source, new_name)

def split_train_test_good(directory,fraction_test,seed):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory + '/train/good/'):
        files = filenames
        break
    train_good, test_good = train_test_split(files, test_size=fraction_test, random_state=seed)
    for image in test_good:
        shutil.copy(directory+'/train/good/'+image, directory+'/test/good/'+image)



def create_tire_dataset( main_directory_source = "data/P_Dataset/",
                            ipcodes = ["21532"],
                            main_directory_dest = "data/copertoni/",
                            all_info_directory = "all_info/"):



    Path(main_directory_dest).mkdir(parents=True, exist_ok=True)
    for code in ipcodes:
        Path(main_directory_dest + code + "/train").mkdir(parents=True, exist_ok=True)
        Path(main_directory_dest + code + "/test").mkdir(parents=True, exist_ok=True)
        Path(main_directory_dest + code + "/train/good").mkdir(parents=True, exist_ok=True)
        Path(main_directory_dest + code + "/test/anomaly").mkdir(parents=True, exist_ok=True)
        Path(main_directory_dest + code + "/test/good").mkdir(parents=True, exist_ok=True)

        csv_names = ["Number", "Folder", "Type"]

        try:
            print(all_info_directory + code + "/anomaly_images.csv", sep=';')
            anomaly_df = pd.read_csv(all_info_directory + code + "/anomaly_images.csv", sep=';',header=None, names=csv_names)
        except:
            anomaly_df = []
            print("code " + str(code) + " has no anomaly csv or it is empty")

        try:
            borderline_df = pd.read_csv(all_info_directory + code + "/borderline_images.csv", sep=';',header=None, names=csv_names)
        except:
            borderline_df = []
            print("code " + str(code) + " has no borderline csv or it is empty")

        try:
            ok_df = pd.read_csv(all_info_directory + code + "/ok_images.csv", sep=';',header=None, names=csv_names)
        except:
            ok_df = []
            print("code " + str(code) + " has no ok csv or it is empty")

        for element in os.listdir(main_directory_source + code + "/nas" + "/crops"):

            copy_element_from_directory(fromDirectory=main_directory_source + code + "/nas" + "/crops/" + element,
                                        toDirectory=main_directory_dest + code,
                                        folder=element,
                                        anomaly_df=anomaly_df,
                                        borderline_df=borderline_df,
                                        ok_df=ok_df,
                                        tag= element.split('-')[2],
                                        useTagAsCompleteName=False)
        split_train_test_good(directory=main_directory_dest + code, fraction_test=0.2, seed=42)

def full_image_copy(row, filename, directory_dest, side=None, type_dest=None):
    tag= row['Barcode'].split('-')[2]
    if side:
        new_name = os.path.join(directory_dest, str(side), type_dest, str(row['Ipcode']) + '_' + tag + '_' + str(row['Side'])).replace("\\","/")
        new_name = get_slideimage_name(new_name, side)
    else:
        new_name = os.path.join(directory_dest, str(row['Ipcode']) + '_' + tag + '_' + str(row['Side'])).replace("\\","/")
    try:
        shutil.copy(filename, new_name)
        #crop image
        im = Image.open(new_name)
        width_source, height_source = im.size
        left, top, bottom, right = ipcode_size_crop[width_source,height_source]
        im = im.crop((left, top, right, bottom))
        im = im.save(new_name)
    except FileNotFoundError:
        print(f"File {filename} is missing")

def patch_image_copy(row, filename, directory_dest, num_horizontal_patches, num_vertical_patches, side=None, type_dest=None):
    base_tag = row['Barcode'].split('-')[2]
    for hor in range(num_horizontal_patches):
        for ver in range(num_vertical_patches):
            patch_tag = 'P_' + str(hor) + '_' + str(ver)
            type_dest_temp = type_dest
            directory_dest_temp = directory_dest
            if "anomaly" in directory_dest or (type_dest is not None and "anomaly" in type_dest):
                if side:
                    mask_anomaly_directory_dest = os.path.join(c.dataset_path, c.class_name, type_dest)
                    mask_anomaly_directory_dest = mask_anomaly_directory_dest.replace("validation", "masks")
                    mask_anomaly_directory_dest = mask_anomaly_directory_dest.replace("test", "masks")
                else:
                    mask_anomaly_directory_dest = directory_dest.replace("validation", "masks")
                    mask_anomaly_directory_dest = mask_anomaly_directory_dest.replace("test", "masks")
                filename_mask = os.path.join(mask_anomaly_directory_dest, str(row['Ipcode']) + '_' + base_tag + '_' + str(row['Side'])).replace("\\","/")
                mask_im = get_mask_patch(filename_mask, hor, ver, num_horizontal_patches, num_vertical_patches)
                max_value_mask = np.amax(mask_im)
                if max_value_mask == 0 or max_value_mask == 3:
                    if side:
                        type_dest_temp = type_dest_temp.replace("anomaly", "good")
                        print(patch_tag, filename_mask, " Moved to good ", type_dest_temp)
                    else:
                        directory_dest_temp = directory_dest_temp.replace("anomaly", "good")
                        print(patch_tag, filename_mask, " Moved to good ", directory_dest_temp)
                else:
                    print("ANOMALY ", patch_tag, filename_mask, directory_dest_temp)
            if side:
                new_name = os.path.join(directory_dest_temp, str(side), type_dest_temp, str(row['Ipcode']) + '_' + base_tag + '_' + patch_tag + '_' + str(row['Side'])).replace("\\","/")
                new_name = get_slideimage_name(new_name, side)
            else:
                new_name = os.path.join(directory_dest_temp, str(row['Ipcode']) + '_' + base_tag + '_' + patch_tag + '_' + str(row['Side'])).replace("\\","/")
            shutil.copy(filename, new_name)
            #crop image
            im = Image.open(new_name)
            width_source, height_source = im.size
            left, top, bottom, right = ipcode_size_crop[width_source,height_source]
            im= im.crop((left, top, right, bottom))

            print("full image: ", left, right, top, bottom)


            #crop patch
            width_new, height_new = im.size
            left, top, right, bottom = crop_coordinates_patch(hor, ver, width_new, height_new, num_horizontal_patches, num_vertical_patches)
            
            print("patch cut: ", left, right, top, bottom)
            im = im.crop((left, top, right, bottom))
            im = im.save(new_name)

def crop_coordinates_patch(hor, ver, width, height, num_horizontal_patches, num_vertical_patches, overlap_hor=c.overlap_hor, overlap_ver=c.overlap_ver):
    step_width = width // num_horizontal_patches
    step_height = height // num_vertical_patches
    if step_width < overlap_hor:
        raise ValueError("horizontal padding is greater than patch widht")
    if step_height < overlap_ver:
        raise ValueError("vertical padding is greater than patch height")
    #HORIZONTAL
    #only one
    if num_horizontal_patches == 1:
        left = 0
        right = width
    #not last hor
    elif(hor < num_horizontal_patches - 1):
        left = 0 + (step_width * hor)
        if left > overlap_hor:
            left -= overlap_hor
        right = left + step_width + overlap_hor
    #last hor
    else:
        #last patch must have correct dimensions even if width is not perfectly divisible
        left = width - step_width - overlap_hor
        right = width

    #VERTICAL
    #only one
    if num_vertical_patches == 1:
        top = 0
        bottom = height
    #not last ver
    elif(ver < num_vertical_patches - 1):
        bottom = height - (step_height * ver)
        if bottom < height:
            bottom += overlap_ver
        top = bottom - step_height - overlap_ver
    #last ver
    else:
        #last patch must have correct dimensions even if height is not perfectly divisible
        top = 0
        bottom = step_height + overlap_ver
    return left, top, right, bottom


def get_mask_patch(filename_full_mask, hor, ver, num_horizontal_patches_max, num_vertical_patches_max):
    im = Image.open(filename_full_mask)
    width_source, height_source = im.size
    left, top, bottom, right = ipcode_size_crop[width_source,height_source]
    im = im.crop((left, top, right, bottom))
    width_new, height_new = im.size
    left, top, right, bottom = crop_coordinates_patch(hor, ver, width_new, height_new, num_horizontal_patches_max, num_vertical_patches_max)
    im = im.crop((left, top, right, bottom))
    return im

def copy_from_csv(df,directory_source, directory_dest):
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        filename = directory_source + str(row['Ipcode']) + "/nas/crops/" + str(row['Barcode']) + '/' + str(row['Side'])
        if c.use_patches:
            patch_image_copy(row, filename, directory_dest, c.num_horizontal_patches, c.num_vertical_patches)
        else:
            full_image_copy(row, filename, directory_dest)


def create_dataset_split_accepted( main_directory_source = "data/",
                                   name_class = "tire_split",
                                   main_directory_dest = "data/copertoni/",
                                   all_info_directory = "all_info/datasplit",
                                   coco_labels_directory = "all_info/coco_labels"):
    Path(main_directory_dest).mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/train").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/test").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/validation").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/train/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/test/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/test/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/validation/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + name_class + "/validation/good").mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(all_info_directory + "/train.csv",sep=';')
    test_ok_df = pd.read_csv(all_info_directory + "/test_ok.csv",sep=';')
    test_anomaly_df = pd.read_csv(all_info_directory + "/test_anomaly.csv",sep=';')
    vali_ok_df = pd.read_csv(all_info_directory + "/vali_ok.csv",sep=';')
    vali_anomaly_df = pd.read_csv(all_info_directory + "/vali_anomaly.csv",sep=';')
    
    # I must create the masks before cppying the images if I use patches
    '''
    mask_good_directory_dest = main_directory_dest + name_class + "/masks/good"
    Path(mask_good_directory_dest).mkdir(parents=True, exist_ok=True)
    test_ok_df = test_ok_df.reset_index()  # make sure indexes pair with number of rows
    vali_ok_df = vali_ok_df.reset_index()  # make sure indexes pair with number of rows
    for index, row in itertools.chain(test_ok_df.iterrows(), vali_ok_df.iterrows()):
    #try:
        #coco = COCO(os.path.join(coco_labels_directory, str(row['Ipcode']) + '_new.json'))
        coco = COCO(os.path.join(coco_labels_directory,'all_ipcodes.json'))
        image_id = row['Id']
        img = coco.imgs[image_id]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        #I just save a mask of 0 (okay) for test good
        mask = coco.annToMask(anns[0])
        im = Image.fromarray(mask)
        new_name = os.path.join(mask_good_directory_dest, str(row['Ipcode']) + '_' + str(row['Barcode'].split('-')[2]) + '_' + str(row['Side'])).replace("\\","/")
        im.save(new_name)
    #except Exception as e:
        #   print("Can't create mask of good ", str(row['Ipcode']),str(row['Barcode']),str(row['Side']))
        #  print(e)
    
    mask_anomaly_directory_dest = main_directory_dest + name_class + "/masks/anomaly"
    Path(mask_anomaly_directory_dest).mkdir(parents=True, exist_ok=True)
    test_anomaly_df = test_anomaly_df.reset_index()  # make sure indexes pair with number of rows
    vali_anomaly_df = vali_anomaly_df.reset_index()  # make sure indexes pair with number of rows
    for index, row in itertools.chain(test_anomaly_df.iterrows(), vali_anomaly_df.iterrows()):
    #try:
        #coco = COCO(os.path.join(coco_labels_directory, str(row['Ipcode']) + '_new.json'))
        coco = COCO(os.path.join(coco_labels_directory,'all_ipcodes.json'))
        image_id = row['Id']
        img = coco.imgs[image_id]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        im = Image.fromarray(mask)
        new_name = os.path.join(mask_anomaly_directory_dest, str(row['Ipcode']) + '_' + str(row['Barcode'].split('-')[2]) + '_' + str(row['Side'])).replace("\\","/")
        im.save(new_name)
    #except:
        #   print("Can't create mask of anomaly ", str(row['Ipcode']),str(row['Barcode']),str(row['Side']))
    '''
    copy_from_csv(train_df,main_directory_source,os.path.join(main_directory_dest, name_class, "train/good"))
    copy_from_csv(test_ok_df,main_directory_source,os.path.join(main_directory_dest, name_class, "test/good"))
    copy_from_csv(test_anomaly_df,main_directory_source,os.path.join(main_directory_dest, name_class, "test/anomaly"))
    copy_from_csv(vali_ok_df,main_directory_source,os.path.join(main_directory_dest, name_class, "validation/good"))
    copy_from_csv(vali_anomaly_df,main_directory_source,os.path.join(main_directory_dest, name_class, "validation/anomaly"))
    

def get_slideimage_name(filename, number):
    return filename[:-4] + "_00" + str(number) + filename[-4:]


def copy_sidelight_from_csv(df,directory_source, directory_dest, type_dest, sides):
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        filename = directory_source + str(row['Ipcode']) + "/nas/crops/" + str(row['Barcode']) + '/' + str(row['Side'])
        for side in sides:
            filename_slide = get_slideimage_name(filename,side)
            if c.use_patches:
                patch_image_copy(row, filename_slide, directory_dest, c.num_horizontal_patches, c.num_vertical_patches, side, type_dest)
            else:
                full_image_copy(row, filename_slide, directory_dest, side, type_dest)

def create_sidelight_dataset(main_directory_source = "data/",
                                name_class = "tire_split",
                                main_directory_dest = "sidelight",
                                all_info_directory = "all_info/datasplit",
                                coco_labels_directory = "all_info/coco_labels"):

    Path(main_directory_dest).mkdir(parents=True, exist_ok=True)
    #Path(main_directory_dest + "/" + name_class + "/train/").mkdir(parents=True, exist_ok=True)
    #Path(main_directory_dest + "/" + name_class + "/test/").mkdir(parents=True, exist_ok=True)
    #Path(main_directory_dest + "/" + name_class + "/validation/").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/1/train/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/2/train/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/1/test/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/2/test/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/1/test/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/2/test/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/1/validation/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/2/validation/anomaly").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/1/validation/good").mkdir(parents=True, exist_ok=True)
    Path(main_directory_dest + "/" + name_class + "/2/validation/good").mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(all_info_directory + "/train.csv",sep=';')
    test_ok_df = pd.read_csv(all_info_directory + "/test_ok.csv",sep=';')
    test_anomaly_df = pd.read_csv(all_info_directory + "/test_anomaly.csv",sep=';')
    vali_ok_df = pd.read_csv(all_info_directory + "/vali_ok.csv",sep=';')
    vali_anomaly_df = pd.read_csv(all_info_directory + "/vali_anomaly.csv",sep=';')

    sides = [1,2]

    copy_sidelight_from_csv(train_df,main_directory_source,os.path.join(main_directory_dest, name_class), "train/good", sides)
    copy_sidelight_from_csv(test_ok_df,main_directory_source,os.path.join(main_directory_dest, name_class), "test/good", sides)
    copy_sidelight_from_csv(test_anomaly_df,main_directory_source,os.path.join(main_directory_dest, name_class), "test/anomaly", sides)
    copy_sidelight_from_csv(vali_ok_df,main_directory_source,os.path.join(main_directory_dest, name_class), "validation/good", sides)
    copy_sidelight_from_csv(vali_anomaly_df,main_directory_source,os.path.join(main_directory_dest, name_class), "validation/anomaly", sides)


