import os
import numpy as np
import csv
def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels

path = "seed_2026"
# 指定你的文件夹路径
pred_path = 'G:/result-2/' + path
gt_path = 'G:/result-2/Label-point'


import numpy as np
from scipy.spatial import cKDTree

from scipy.spatial import distance_matrix


def chamfer_distance_3d(array1, array2):
    """
    Calculate the Chamfer Distance between two 3D point clouds.

    Parameters:
    array1: numpy.ndarray
        The first point cloud array of shape (n, 3).
    array2: numpy.ndarray
        The second point cloud array of shape (n, 3).

    Returns:
    float
        The Chamfer Distance between the two point clouds.
    """

    # Compute distance matrices between each point in one array to every point in the other array
    dist_matrix_1_to_2 = distance_matrix(array1, array2)
    dist_matrix_2_to_1 = distance_matrix(array2, array1)

    # For each point in array1, find the closest point in array2 and vice versa
    nearest_dist_1_to_2 = np.min(dist_matrix_1_to_2, axis=1)
    nearest_dist_2_to_1 = np.min(dist_matrix_2_to_1, axis=1)

    # Compute the Chamfer Distance
    chamfer_dist = np.mean(nearest_dist_1_to_2) + np.mean(nearest_dist_2_to_1)

    return chamfer_dist


def accuracy(pred, label):
    return (pred == label).mean()


def dice_coefficient(pred, label):
    intersection = (pred * label).sum()
    return (2 * intersection) / (pred.sum() + label.sum())


# 计算指标及其标准差
def calculate_metrics(pred, label, num_samples):
    acc_list = []
    dice_list = []

    for i in range(num_samples):
        # 随机选择两个数组的子集进行计算
        rand_indices = np.random.choice(len(pred), size=min(10, len(pred)), replace=False)
        sub_pred = pred[rand_indices]
        sub_label = label[rand_indices]

        # 计算ACC和Dice
        acc = accuracy(sub_pred, sub_label)
        dice = dice_coefficient(sub_pred, sub_label)

        acc_list.append(acc)
        dice_list.append(dice)

    # 计算平均ACC和Dice
    mean_acc = np.mean(acc_list)
    mean_dice = np.mean(dice_list)

    # 计算标准差
    std_acc = np.std(acc_list)
    std_dice = np.std(dice_list)

    return mean_acc, std_acc, mean_dice, std_dice


num, cd_l, cd_r, cd_a, acc_l, acc_r, acc_a, dice_l, dice_r, dice_a = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
acc_l_list, acc_r_list, acc_a_list = [], [], []
dice_l_list, dice_r_list, dice_a_list = [], [], []
cd_l_list, cd_r_list, cd_a_list = [], [], []
# 遍历文件夹
name_list = []
for filename in os.listdir(pred_path):
    if filename.endswith('.eseg') and "-vs." not in filename:
        num = num + 1
        name_list.append(filename)
        # 拼接完整的文件路径
        file_path = os.path.join(pred_path, filename)
        gt_file_path = os.path.join(gt_path, filename)

        y_true = read_seg(gt_file_path)
        y_pred = read_seg(file_path)

        lig_t, lig_p = [], []
        rid_t, rid_p = [], []
        all_t, all_p = [], []
        for indexn in range(y_true.shape[0]):
            if y_true[indexn, 3] == 1:
                lig_t.append(y_true[indexn,:3])
                all_t.append(y_true[indexn, :3])
            elif y_true[indexn, 3] == 2:
                rid_t.append(y_true[indexn, :3])
                all_t.append(y_true[indexn, :3])

            if y_pred[indexn, 3] == 1:
                lig_p.append(y_pred[indexn, :3])
                all_p.append(y_pred[indexn, :3])
            elif y_pred[indexn, 3] == 2:
                rid_p.append(y_pred[indexn, :3])
                all_p.append(y_pred[indexn, :3])



        if len(lig_p) == 0:
            cd_l = cd_l + 50
            cd_l_list.append(50)
        else:
            cd_l = cd_l +chamfer_distance_3d(lig_t, lig_p)
            cd_l_list.append(chamfer_distance_3d(lig_t, lig_p))


        if len(rid_p) == 0:
            cd_r = cd_r + 50
            cd_r_list.append(50)
        else:
            cd_r = cd_r +chamfer_distance_3d(rid_t, rid_p)
            cd_r_list.append(chamfer_distance_3d(rid_t, rid_p))

        if len(all_p) == 0:
            cd_a = cd_a + 50
            cd_a_list.append(50)
        else:
            cd_a = cd_a +chamfer_distance_3d(all_t, all_p)
            cd_a_list.append(chamfer_distance_3d(all_t, all_p))

        """
        ACC and Dice
        """
        y_t = y_true[:, 3]+0
        y_p = y_pred[:, 3]+0
        # lig
        y_t[y_t==2] = 0
        y_p[y_p==2] = 0
        acc_l = acc_l + accuracy(y_p, y_t)
        dice_l = dice_l + dice_coefficient(y_p, y_t)
        # print("lig:", accuracy(y_p, y_t), dice_coefficient(y_p, y_t))
        acc_l_list.append(accuracy(y_p, y_t))
        dice_l_list.append(dice_coefficient(y_p, y_t))


        y_t = y_true[:, 3]+0
        y_p = y_pred[:, 3]+0
        # rid
        y_t[y_t < 2] = 0
        y_p[y_p < 2] = 0
        y_t[y_t == 2] = 1
        y_p[y_p == 2] = 1
        acc_r = acc_r + accuracy(y_p, y_t)
        dice_r = dice_r + dice_coefficient(y_p, y_t)
        # print("rid:", accuracy(y_p, y_t), dice_coefficient(y_p, y_t))
        acc_r_list.append(accuracy(y_p, y_t))
        dice_r_list.append(dice_coefficient(y_p, y_t))




        y_t = y_true[:, 3]+0
        y_p = y_pred[:, 3]+0
        # all
        y_t[y_t >1] = 1
        y_p[y_p >1] = 1
        acc_a = acc_a + accuracy(y_p, y_t)
        dice_a = dice_a + dice_coefficient(y_p, y_t)
        # print("all:", accuracy(y_p, y_t), dice_coefficient(y_p, y_t))
        acc_a_list.append(accuracy(y_p, y_t))
        dice_a_list.append(dice_coefficient(y_p, y_t))


# print("平均acc为：", acc_l/num, np.std(acc_l_list), acc_r/num, np.std(acc_r_list), acc_a/num, np.std(acc_a_list))
# print("平均dice为：", dice_l/num, np.std(dice_l_list), dice_r/num, np.std(dice_r_list), dice_a/num, np.std(dice_a_list))
# print("平均cd为：", cd_l/num, np.std(cd_l_list), cd_r/num, np.std(cd_r_list), cd_a/num, np.std(cd_a_list))
print("平均acc为：",
      "{:.3f}".format(acc_l/num),
      "±{:.3f}".format(np.std(acc_l_list)),
      "{:.3f}".format(acc_r/num),
      "±{:.3f}".format(np.std(acc_r_list)),
      "{:.3f}".format(acc_a/num),
      "±{:.3f}".format(np.std(acc_a_list)))

print("平均dice为：",
      "{:.3f}".format(dice_l/num),
      "±{:.3f}".format(np.std(dice_l_list)),
      "{:.3f}".format(dice_r/num),
      "±{:.3f}".format(np.std(dice_r_list)),
      "{:.3f}".format(dice_a/num),
      "±{:.3f}".format(np.std(dice_a_list)))

print("平均cd为：",
      "{:.3f}".format(cd_l/num),
      "±{:.3f}".format(np.std(cd_l_list)),
      "{:.3f}".format(cd_r/num),
      "±{:.3f}".format(np.std(cd_r_list)),
      "{:.3f}".format(cd_a/num),
      "±{:.3f}".format(np.std(cd_a_list)))





# 确保num和name_list的长度一致
print(len(name_list), len(dice_l_list), len(dice_r_list))
assert len(name_list) == len(dice_l_list) == len(dice_r_list), "所有列表的长度必须一致"

# 创建或覆盖CSV文件
csv_file_path = path + '-results.csv'  # 你可以根据需要修改文件名和路径
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写入标题行
    writer.writerow(['Name', 'Dice_L', 'Dice_R'])

    # 写入数据行
    for i in range(len(name_list)):
        writer.writerow([name_list[i], dice_l_list[i], dice_r_list[i]])

print(f"数据已写入 {csv_file_path}")
