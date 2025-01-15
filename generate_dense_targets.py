import os
import cv2
import glob
import random
import numpy as np
import os.path as osp
import skimage.measure
from PIL import Image
from scipy.optimize import curve_fit
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree

src_img_path = 'dense_dataset/target/img'
src_mask_path = 'dense_dataset/target/mask'
dst_img_path = 'data/SIRSTdevkit/PNGImages'
dst_mask_path = 'data/SIRSTdevkit/SIRST/BinaryMask'
dst_sky_path = 'data/SIRSTdevkit/SkySeg/BinaryMask'
dense_img_path = 'DenseSIRSTv2/SIRSTdevkit/PNGImages'
dense_mask_path = 'DenseSIRSTv2/SIRSTdevkit/SIRST/BinaryMask'
dense_palette_path = 'DenseSIRSTv2/SIRSTdevkit/SIRST/PaletteMask'
dense_xml_path = 'DenseSIRSTv2/SIRSTdevkit/SIRST/BBox'

if not os.path.exists(dense_img_path):
    os.makedirs(dense_img_path)
if not os.path.exists(dense_mask_path):
    os.makedirs(dense_mask_path)
if not os.path.exists(dense_palette_path):
    os.makedirs(dense_palette_path)
if not os.path.exists(dense_xml_path):
    os.makedirs(dense_xml_path)


def write_bbox_to_file(img, idx) -> None:
    bboxes = []
    masked_image = skimage.measure.label(img)
    props_mask = skimage.measure.regionprops(masked_image)
    for region in props_mask:
        ymin, xmin, ymax, xmax = np.array(region.bbox)
        bboxes.append((xmin, ymin, xmax, ymax))

    annotation = ET.Element('annotation')
    filename = ET.SubElement(annotation, 'filename')
    filename.text = idx
 
    _height, _width = img.shape
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(_width)
    height = ET.SubElement(size, 'height')
    height.text = str(_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(1)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        label_object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(label_object, 'name')
        name.text = 'Target'

        _bbox = ET.SubElement(label_object, 'bndbox')
        xmin_elem = ET.SubElement(_bbox, 'xmin')
        xmin_elem.text = str(xmin)

        ymin_elem = ET.SubElement(_bbox, 'ymin')
        ymin_elem.text = str(ymin)

        xmax_elem = ET.SubElement(_bbox, 'xmax')
        xmax_elem.text = str(xmax)

        ymax_elem = ET.SubElement(_bbox, 'ymax')
        ymax_elem.text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree_str = ET.tostring(tree.getroot(), encoding='unicode')
    if not os.path.exists(dense_xml_path):
        # 如果目录不存在，则创建它
        os.makedirs(dense_xml_path)
    save_xml_path = os.path.join(dense_xml_path, idx + '.xml')
    if os.path.exists(save_xml_path):
        os.remove(save_xml_path)
    root = etree.fromstring(tree_str).getroottree()
    root.write(save_xml_path, pretty_print=True)

def save_palette(mask_path, dst_img):
    # Save palette
    # Create a new image with a red palette
    mask = Image.open(mask_path)
    red_palette = [0, 0, 0, 255, 0, 0]  # Red color in RGB format
    red_image = Image.new("P", mask.size)
    red_image.putpalette(red_palette)

    # Convert the input mask to "P" mode using the red palette
    red_image.paste(mask)
    red_image = red_image.convert("P")

    # Save the mask with the red palette
    red_image.save(os.path.join(dense_palette_path, os.path.basename(dst_img)), format='PNG', optimize=True)

def extract_target(mask_list):
    while True:
        while True:
            mask_file = random.choice(mask_list)
            src_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if np.mean(src_mask) != 0:
                break
        img_file = osp.join("data/SIRSTdevkit/PNGImages", osp.basename(mask_file).replace('_pixels0.png', '.png'))
        src_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        # 寻找轮廓
        contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取轮廓的坐标
        coordinates = contours[0].reshape(-1, 2)

        # 创建一个空白图像
        mask = np.zeros_like(src_mask)
        # 在空白图像上绘制目标的轮廓
        cv2.drawContours(mask, [coordinates], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(mask)

        '''
        只提取目标，不提取背景。
        '''
        extracted_object = cv2.bitwise_and(src_img, src_img, mask=mask)
        extracted_object = extracted_object[y : y + h, x : x + w]
        extracted_mask = mask[y : y + h, x : x + w]
        break

    extracted_object = cv2.resize(extracted_object, (5, 5), interpolation=cv2.INTER_NEAREST)
    extracted_mask = np.ones_like(extracted_object, dtype=np.uint8) * 255
    extracted_mask[extracted_object == 0] = 0

    return extracted_object, extracted_mask

def generate_gaussian_matrix(h, w, sigma_x=0.8, sigma_y=0.6, mu_x=0.0, mu_y=0.0, theta=0.0):
    """
    Generate a 2D rotated anisotropic Gaussian matrix.
    """

    sigma_x = np.random.uniform(0.3, 0.6)
    sigma_y = np.random.uniform(0.3, 0.6)
    mu_x = -np.random.uniform(0, 0.2)
    mu_y = -np.random.uniform(0, 0.2)
    theta = np.random.randint(-90, 90)

    # Angle of rotation in radians
    theta_radians = np.radians(theta)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians)],
                                [np.sin(theta_radians), np.cos(theta_radians)]])

    # Create a coordinate grid
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    
    # Stack the coordinates for matrix multiplication
    coords = np.stack([X.ravel() - mu_x, Y.ravel() - mu_y])
    
    # Apply the rotation matrix to the coordinates
    rot_coords = rotation_matrix @ coords
    
    # Calculate the squared distance from the center for each axis after rotation
    d_x2 = (rot_coords[0] ** 2)
    d_y2 = (rot_coords[1] ** 2)
    
    # Apply the anisotropic Gaussian formula
    gaussian = np.exp(-(d_x2 / (2.0 * sigma_x**2) + d_y2 / (2.0 * sigma_y**2)))
    
    # Reshape back to the original shape and normalize
    gaussian = gaussian.reshape(h, w)
    gaussian -= gaussian.min()
    gaussian /= gaussian.max()
    
    return gaussian


if __name__ == '__main__':  
    src_img_all = os.listdir(src_img_path)
    dst_img_all = sorted([os.path.join(dst_img_path, img_name) for img_name in os.listdir(dst_img_path) if img_name.endswith('.png')]) 
    print(len(dst_img_all))

    sky_list = []
    for dst_img in dst_img_all:
        dst_mask = os.path.join(dst_mask_path, os.path.splitext(os.path.basename(dst_img))[0]+"_pixels0.png")
        dst_sky = os.path.join(dst_sky_path, os.path.splitext(os.path.basename(dst_img))[0]+"_pixels0.png")
        img_dst = cv2.imread(dst_img, cv2.IMREAD_GRAYSCALE)
        mask_dst = cv2.imread(dst_mask, cv2.IMREAD_GRAYSCALE)
        sky_dst = cv2.imread(dst_sky, cv2.IMREAD_GRAYSCALE)

        print(dst_img)

        # sky_coordinates存储了所有值为 255 的元素的行坐标和列坐标
        sky_array = np.array(sky_dst)
        sky_coordinates = np.where(sky_array == 255)
        if len(sky_coordinates[0]) <= 0:
            continue

        sky_list.append(len(sky_coordinates[0]))
        
        result_img = img_dst.copy()
        result_mask = mask_dst.copy()
        
        random_sky = random.randint(3, 5)
        offsets = [12, 16, 20, 24]
        offsets_count = [0, 0, 0, 0]
        random_numbers = [[10, 12], [12, 14], [14, 16], [16, 18]]

        # 获取所有含目标的源图像
        # src_mask_path = "data/SIRSTdevkit/SkySeg/BinaryMask"
        # pattern = os.path.join(src_mask_path, "Misc_*.png")
        # src_mask_list = glob.glob(pattern)

        for _ in range(random_sky):
            # 随机选择一个天空区域生成稠密目标
            while True:
                index = random.randint(0, len(offsets) - 1)
                if offsets_count[index] <= 1:
                    offsets_count[index] += 1
                    break
            offset = offsets[index]
            k = 0
            flag = 0
            while True:
                k = k + 1
                random_index = np.random.randint(0, len(sky_coordinates[0]))
                # shape():返回(height, width, channels)
                if 1 < sky_coordinates[1][random_index] < result_img.shape[1]-offset-1 and 1 < sky_coordinates[0][random_index] < result_img.shape[0]-offset-1:
                    random_pos_y = sky_coordinates[1][random_index] # 列坐标
                    random_pos_x = sky_coordinates[0][random_index] # 行坐标
                    target_overlap = result_mask[random_pos_x:random_pos_x+offset-10, random_pos_y:min(random_pos_y+offset+10, result_img.shape[1])].any()
                    
                    # 判断选取的区域是否完全是天空区域
                    random_mask = np.zeros_like(sky_dst)
                    random_mask[random_pos_x:random_pos_x+offset, random_pos_y:random_pos_y+offset] = 255
                    # 使用位运算检查随机区域是否完全在天空区域内
                    random_sky = cv2.bitwise_and(random_mask, sky_dst)
                    is_completely_in_sky = np.all(random_sky == random_mask)

                    if is_completely_in_sky and not target_overlap:
                        print("已选择天空区域生成稠密目标！")
                        break
                if k > 20 and index > 0:
                    index -= 1
                    offset = offsets[index]
                    k = 0
                elif k > 60:
                    flag = 1
                    break
            if flag == 1 :
                continue
            
            random_number_choice = random_numbers[index]
            random_number = random.randint(random_number_choice[0], random_number_choice[1])
            print(random_number)
            for _ in range(random_number):
                # 获取要粘贴的目标
                # img_src, mask_src = extract_target(src_mask_list)
                random_image_file = random.choice(src_img_all)
                choice_image_path = os.path.join(src_img_path, random_image_file)
                choice_mask_path = os.path.join(src_mask_path, random_image_file)
                img_src = cv2.imread(choice_image_path, cv2.IMREAD_GRAYSCALE)
                mask_src = cv2.imread(choice_mask_path, cv2.IMREAD_GRAYSCALE)
                extracted_object = img_src.copy()
                extracted_mask = mask_src.copy()

                h = img_src.shape[0]
                w = img_src.shape[1]

                r = 0
                flag_ = 0
                while True:
                    r = r + 1
                    # 随机选择一个位置粘贴目标
                    paste_x = np.random.randint(random_pos_x, random_pos_x+offset-h)
                    paste_y = np.random.randint(random_pos_y, random_pos_y+offset-w)
                    target_overlap = result_mask[paste_x-1:paste_x+h+1, paste_y-1:paste_y+w+1].any()
                    print("已随机选择一个坐标进行粘贴！")
                    if not target_overlap:
                        break
                    if r > 60:
                        flag_ = 1
                        break
                
                if flag_ == 1 :
                    continue

                lamda = np.random.uniform(0.25, 0.35)

                # 生成高斯矩阵
                gaussian_matrix = generate_gaussian_matrix(h, w)

                for i in range(h):
                    for j in range(w):
                        # pixel_value = result_img[paste_x + i, paste_y + j] + result_img[paste_x + i, paste_y + j] * (1 - lamda) * gaussian_matrix[i, j] + extracted_object[i, j] * lamda * gaussian_matrix[i, j]
                        pixel_value = result_img[paste_x + i, paste_y + j] + extracted_object[i, j] * lamda * gaussian_matrix[i, j]
                        result_img[paste_x + i, paste_y + j] = min(255, pixel_value)  # 确保值不超过255
                
                result_mask[paste_x:paste_x+h, paste_y:paste_y+w] = 255
                gauss_mask = result_mask[paste_x:paste_x+h, paste_y:paste_y+w] * lamda * gaussian_matrix
                ret, binary = cv2.threshold(gauss_mask, 25, 255, cv2.THRESH_BINARY)
                result_mask[paste_x:paste_x+h, paste_y:paste_y+w] = binary

        cv2.imwrite(os.path.join(dense_img_path, os.path.basename(dst_img)), result_img)
        cv2.imwrite(os.path.join(dense_mask_path, os.path.basename(dst_mask)), result_mask)
        write_bbox_to_file(result_mask, idx = os.path.splitext(os.path.basename(dst_img))[0])
        save_palette(os.path.join(dense_mask_path, os.path.basename(dst_mask)), dst_img)
