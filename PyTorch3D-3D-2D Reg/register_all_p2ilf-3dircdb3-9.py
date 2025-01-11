import torch
import torchvision.transforms as T
from pytorch3d.structures import Meshes
from PIL import Image
from torch import nn
from torch.nn import Parameter
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, look_at_view_transform, PerspectiveCameras, SoftPhongShader,
    TexturesVertex, PointLights, BlendParams,
)
from pytorch3d.transforms import (
    Rotate, Translate, euler_angles_to_matrix, 
    quaternion_multiply, quaternion_to_matrix, 
    axis_angle_to_quaternion,
    rotation_6d_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d,
)
from pytorch3d.utils import (
    cameras_from_opencv_projection,
)
from lightless_shader import LightlessShader
from modelallrt import Model
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
import matplotlib.pyplot as plt
import utils
import optuna
import nibabel as nib
import os
import math
from loss import overlap_loss, combined_loss
import random
import datetime
import torchvision.transforms as T
# 设定设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import torchvision.transforms
from pytorch3d.io import save_obj
def load_data(
    path_mesh = "/home/zxk/code/PyTorch3D/3Dircadb-10.obj",
    path_image = "/home/zxk/code/PyTorch3D/p2ilf-png-3/P2ILF22_patient3_1.png",
    path_camera_param = "/home/zxk/code/PyTorch3D/registration/camera_parameters/patient3/calibration.xml",
    scale=1e-3,
    dilate=True,
    image_width=3840
):
    image_scale_factor = image_width / 3840
    print("image_scale_factor:", image_scale_factor)
    # 加载 3D 模型和 2D 图像
    mesh = load_objs_as_meshes([path_mesh], device=device)
    # 假设 mesh 是一个预先加载的Meshes对象，且vertices已经有颜色信息
    # ridge_indices 和 ligament_indices 分别是脊和韧带顶点的索引列表
    # ridge_indices = [128, 523, 655, 273, 146, 531, 662, 283, 284, 286, 287, 415, 422, 550, 297, 298, 299, 308, 564, 570, 705, 194, 195, 324, 583, 455, 457, 458, 214, 215, 349, 221, 351, 483, 230, 238, 624, 501, 254]
    # ligament_indices = [484, 773, 422, 552, 458, 811, 906, 781, 879, 722, 856, 543]
    
    
    

    ligament_indices = [1157, 1158, 1156, 1169, 1171, 1166, 1177, 1186, 1176, 1175, 1188, 1167, 1189, 1191, 1190, 1168, 1195, 1194, 1170, 1199, 1201, 1200, 1204, 1202, 1205, 1272, 1203, 1196, 1197, 1207, 1216, 1273, 1218, 1275, 1219, 1241, 1245, 1240, 1247, 1246, 1242, 1249, 1250, 1192, 1263, 1251, 1252, 1258, 1266, 1206, 1293, 1265, 1274, 1253, 1220]
    ridge_indices = [822, 829, 826, 823, 824, 825, 830, 870, 828, 831, 1094, 882, 917, 864, 927, 871, 872, 873, 879, 855, 880, 869, 874, 875, 881, 878, 883, 884, 947, 885, 886, 866, 893, 892, 898, 940, 899, 902, 900, 925, 939, 926, 941, 943, 942, 944, 945, 946, 950, 951, 949, 954, 955, 957, 956, 958, 972, 959, 953, 929, 960, 889, 952, 962, 961, 965, 966, 968, 964, 967, 969, 963, 971, 975, 976, 977, 974, 890, 891, 888, 918, 973, 868, 1069, 1092, 1079, 832, 1093, 1178, 1239, 1174, 1180, 1182, 1183, 1181, 1184, 1185, 1187, 1238, 1427, 1429, 1466, 1469, 1468, 1487, 1470, 1465, 1476, 1477, 1481, 1479, 1492, 1483, 1769, 1694, 1695, 1698, 1699, 1701, 1702, 1696, 1703, 1716, 1704, 1697, 1709, 1710, 1708, 1713, 1765, 1717, 1714, 1711, 1715, 1718, 1772, 1760, 1764, 1768, 1770, 1767, 1766, 1482, 903, 948, 1428, 1705]
    # ligament_indices = [2257, 2413, 2414, 2565, 2567, 2655, 2734, 2917, 2792, 2880, 2916, 2918, 2981, 2984, 3075, 3017, 3018, 3149, 3114, 3151, 3146, 3156, 3313, 3228, 3312, 3229, 3347]
    # ridge_indices = [322, 364, 371, 436, 480, 512, 527, 537, 570, 602, 568, 600, 588, 571, 623, 642, 599, 640, 681, 710, 680, 684, 685, 748, 679, 709, 728, 729, 814, 753, 807, 806, 727, 750, 778, 813, 776, 804, 772, 774, 771, 827, 829, 805, 809, 881, 891, 877, 882, 884, 918, 878, 1007, 883, 1081, 890, 999, 941, 1002, 976, 977, 1001, 1005, 1042, 1067, 1076, 1045, 1186, 1252, 1066, 1172, 1131, 1182, 1078, 1175, 1285, 1189, 1248, 1174, 1176, 1321, 1356, 1287, 1402, 1429, 1401, 1466, 1465, 1597, 1573, 1691, 1823, 1822, 2048, 2082]
    
    # 将顶点颜色设置为灰色
    colors = torch.full_like(mesh.verts_packed(), 0.0, device=device)  # 灰色，RGB值为0.5
    # colors = torch.zeros_like(mesh.verts_packed(), device=device)  # 初始化顶点颜色为0
    colors[ridge_indices] = torch.tensor([1.0, 0.0, 0.0], device=device)  # 脊顶点着红色
    colors[ligament_indices] = torch.tensor([0.0, 0.0, 1.0], device=device)  # 韧带顶点着蓝色

    # 创建一个包含颜色信息的Textures对象
    vertex_colors = TexturesVertex(verts_features=colors.unsqueeze(0))
    # mesh.textures = vertex_colors
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    verts = utils.center_mesh(verts)

    mesh = Meshes(
        verts=[verts * scale],
        # verts=[verts],
        faces=[faces],
        textures=vertex_colors
    )
    

    
    # 保存 mesh 到 obj 文件
    # save_obj("modified_mesh.obj", verts_1, faces_1, textures_1)
    

    # ... 加载 2D 图像代码 ...
    if path_image.endswith(".jpg"):
    # image_path = "./2D_mask/P2ILF22_patient1_1_mask.jpg"
        # image = Image.open(path_image)
        # print("image.size:", image.size)
        # # transform = T.Compose([T.Resize((384, 216)), T.ToTensor()])
        # # target_image = transform(image).to(device)
        # # print("target_image.size:", type(transform(image)), T.Compose([T.Resize((384, 216))])(image).size, target_image.max())
        # image_label = image.ToTensor().to(device)
        
        
        image = Image.open(path_image).convert('RGB')  # 确保转换为RGB模式
        # print("image.size:", image.size)
    
        # 这里需要使用 torchvision 的 ToTensor() 转换
        transform = T.Compose([T.ToTensor()])
        image_label = transform(image).to(device)
        # print("image_label.shape:", image_label.shape)
        image_label = image_label.unsqueeze(0)
        # 现在 permute 操作
        image_label = image_label.permute(0, 2, 3, 1)  # 注意这里的顺序
        
        
        
        
    elif path_image.endswith(".nii.gz"):
        nimg = nib.load(path_image)
        label_image = torch.from_numpy(nimg.get_fdata())
        lbl = torch.zeros_like(label_image, dtype=torch.float32)
        lbl[label_image[:,:,0]==1] = torch.Tensor( (1.0, 0, 0) ) # RED # torch.Tensor( (1, 0, 0) ) # ridge
        lbl[label_image[:,:,0]==2] = torch.Tensor( (0, 1.0, 0) ) # GREEN # silhouette
        lbl[label_image[:,:,0]==3] = torch.Tensor( (0, 0, 1.0) ) #BLUE #torch.Tensor( (0, 0, 1) ) # ligament
        image_label = lbl.unsqueeze(0).to(device)
        print("image_label.shape:", image_label.shape)
        
    elif path_image.endswith(".png"):
        # print("path_image:", path_image)
        image = Image.open(path_image).convert('RGB')  # 确保转换为RGB模式
        # print("image.size:", image.size)
    
        # 这里需要使用 torchvision 的 ToTensor() 转换
        transform = T.Compose([T.ToTensor()])
        image_label = transform(image).to(device)
        # print("---image_label.shape:", image_label.shape, image_label.max(), image_label.min())
        image_label = image_label.unsqueeze(0)
        # 现在 permute 操作
        image_label = image_label.permute(0, 2, 3, 1)  # 注意这里的顺序
        
        
        """处理肝脏label"""
        image_liver = Image.open("/home/zxk/code/PyTorch3D/P2ILF22_patient10_9_liver.png").convert('RGB')  # 确保转换为RGB模式
        # print("image.size:", image.size)
    
        # 这里需要使用 torchvision 的 ToTensor() 转换
        transform = T.Compose([T.ToTensor()])
        image_label_liver = transform(image_liver).to(device)
        # print("---image_label.shape:", image_label.shape, image_label.max(), image_label.min())
        image_label_liver = image_label_liver.unsqueeze(0)
        # 现在 permute 操作
        image_label_liver = image_label_liver.permute(0, 2, 3, 1)  # 注意这里的顺序
        
    if dilate:
        image_label = utils.dilate_image(image_label)
        image_label_liver = utils.dilate_image(image_label_liver)
    if image_scale_factor < 1.0:
        img = image_label.permute(0, 3, 1, 2)
        img = torch.nn.functional.interpolate(
            img,
            scale_factor=image_scale_factor,
            mode="bilinear",
        )
        image_label = img.permute(0, 2, 3, 1)
        
        
        img_liver = image_label_liver.permute(0, 3, 1, 2)
        img_liver = torch.nn.functional.interpolate(
            img_liver,
            scale_factor=image_scale_factor,
            mode="bilinear",
        )
        image_label_liver = img_liver.permute(0, 2, 3, 1)

    # Load camera parameters
    # camera_params = utils.construct_camera_matrix(
    #     utils.load_camera_parameters_xml(path_camera_param),
    #     scale=image_scale_factor,
    # )
    camera_params = utils.load_camera_parameters_xml(path_camera_param, scale_factor=image_scale_factor)


    return mesh, image_label, camera_params, image_label_liver



def setup_render(
    camera_params,
    # image_size=(1080, 1920),
    R=torch.eye(3).unsqueeze(0),
    tvec=torch.zeros(1, 3)

):
    image_size = [int(camera_params["height"]), int(camera_params["width"])]
    # print("image_size:", image_size)
    # 设置相机（固定在原点）
    # R, T = look_at_view_transform(
    #     dist= 100.0, #5.0, 
    #     elev= 30, #0.0, 
    #     azim=0.0,
    # )
    # 假设的相机内参矩阵
    # focal_length = torch.tensor([[float("9.5206967601396957e+02"), float("9.5206967601396957e+02")]], device=device)  # fx 和 fy 是焦距
    # principal_point = torch.tensor([[float("8.8447392341747241e+02"), float("5.5368748726315528e+02")]], device=device)  # px 和 py 是主点

    # print("focal_length, principal_point:", focal_length, principal_point)
    # 创建带有内参的相机对象
    # cameras = PerspectiveCameras( # problem???
    #     # focal_length=focal_length,
    #     # principal_point=principal_point,
    #     R=R,  # 旋转矩阵
    #     T=T,  # 平移向量
    #     device=device
    # )

    # Create a perspective camera
    camera_matrix =  utils.construct_camera_matrix(camera_params )
    cameras = cameras_from_opencv_projection( 
            R=R, 
            tvec = tvec, 
            camera_matrix = camera_matrix,
            image_size=torch.Tensor(image_size).unsqueeze(0)
        ).to(device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0))
    # blend_params = BlendParams(sigma=1e-4, gamma=1e-4, )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0, 
        # blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=1, #faces_per_pixel=10,
    )
    # We can add a point light in front of the object. 
    #lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=LightlessShader(blend_params=blend_params, device=device)#, cameras=cameras)#, lights=lights)
        # shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    return renderer

    # # 创建渲染器
    # raster_settings = RasterizationSettings(
    #         image_size=image_size, #(216, 384),
    #     )
    # # 设置固定光源的位置
    # # light_location = [[0.0, 0.0, -3.0]]  # 例如，位于场景前方
    # # # 创建 PointLights 对象
    # # lights = PointLights(device=device, location=light_location)
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    #     # shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)     # 创建着色器，并使用上面定义的光源
    #     shader=HardPhongShader(device=device, cameras=cameras)
    # )




def random_init_mesh_pos():
    # random initial translation:
    init_obj_T = [
        0.0, 
        0.0, 
        0.20,#,0.17,
    ]
    # init_obj_T = [
    #     random.uniform(-0.05, 0.05),
    #     random.uniform(-0.05, 0.05),
    #     random.uniform(0.13, 0.22),
    # ]
    # init_obj_T = [
    #     0.0,
    #     0.0,
    #     0.0,
    # ]
    init_obj_T = torch.tensor(init_obj_T) #.detach().to(device)

    # random initial rotation:
    # euler angle -> quaternion -> rotation matrix
    # ang_x = -math.pi*0.5
    # ang_y = 0
    # ang_z = math.pi
    """原代码，旋转角度错误"""
    # ang_x = random.uniform(-math.pi*0.5-1.3, -math.pi*0.5+1.3)
    # ang_y = random.uniform(-1, 1)
    # ang_z = random.uniform(math.pi-0.5, math.pi+0.5)

    """完全随机化，但是需要大量的初始化才能找到对应的正确位置"""
    ang_x = random.uniform(0, 2 * math.pi)  # 0 到 360 度
    ang_y = random.uniform(0, 2 * math.pi)  # 0 到 360 度
    ang_z = random.uniform(0, 2 * math.pi)  # 0 到 360 度

    """固定位置不发生旋转，因为在传入mesh之前已经将其设置为相机的正前方"""
    ang_x = 0  # 0 到 360 度
    ang_y = 0  # 0 到 360 度
    ang_z = 0  # 0 到 360 度
    """部分随机化，但是需要大量的初始化才能找到对应的正确位置"""
    ang_x = random.uniform(-math.pi / 2, math.pi / 2)  # -90° 到 90° 之间的随机角度
    ang_y = random.uniform(-math.pi / 2, math.pi / 2)  # -90° 到 90° 之间的随机角度
    ang_z = random.uniform(-math.pi / 2, math.pi / 2)  # -90° 到 90° 之间的随机角度

    euler_x = torch.Tensor((ang_x,0,0)).unsqueeze(0)
    euler_y = torch.Tensor((0,ang_y,0)).unsqueeze(0)
    euler_z = torch.Tensor((0,0,ang_z)).unsqueeze(0)
    rot_x = axis_angle_to_quaternion(euler_x)
    rot_y = axis_angle_to_quaternion(euler_y)
    rot_z = axis_angle_to_quaternion(euler_z)

    init_obj_R = quaternion_multiply(quaternion_multiply( rot_x, rot_z ), rot_y)
    init_obj_R = quaternion_to_matrix( init_obj_R )
    
    
    
    
#     """直接设置绕 x 轴 90 度旋转的 3x3 矩阵"""
#     R_x = torch.tensor([
#         [1, 0, 0], 
#         [0, 0, -1], 
#         [0, 1, 0]
#     ], dtype=torch.float32)  # 绕 x 轴旋转 90 度

#     init_obj_R = R_x.to(device)  # 直接将 R_x 赋值给 init_obj_R
    
#     """直接设置绕 y 轴 90 度旋转的 3x3 矩阵"""
#     R_y = torch.tensor([
#         [0, 0, 1], 
#         [0, 1, 0], 
#         [-1, 0, 0]
#     ], dtype=torch.float32)  # 绕 y 轴旋转 90 度
    
#     init_obj_R = R_y.to(device)  # 直接将 R_x 赋值给 init_obj_R
    
#     """直接设置绕 z 轴 90 度旋转的 3x3 矩阵"""
#     R_z = torch.tensor([
#         [0, -1, 0], 
#         [1, 0, 0], 
#         [0, 0, 1]
#     ], dtype=torch.float32)  # 绕 z 轴旋转 90 度
    
#     init_obj_R = R_z.to(device)  # 直接将 R_x 赋值给 init_obj_R
    
    print("init_obj_R:", init_obj_R)
    # init_obj_R = init_obj_R.to(device)

    return init_obj_R, init_obj_T

import numpy as np 
# 保存旋转矩阵到文件
def save_rotation_matrix_to_file(R_matrix, file_name="rotation_matrices.txt"):
    # 将 R_matrix 从计算图中分离，并转换为 NumPy 数组
    R_matrix_np = R_matrix.detach().squeeze().cpu().numpy()  # detach 去除梯度信息

    # 将矩阵保存到 .txt 文件中，追加写入模式 'a'
    with open(file_name, "a") as f:
        for i in range(R_matrix_np.shape[0]):  # 遍历每个旋转矩阵
            matrix = R_matrix_np[i]
            for row in matrix:  # 遍历矩阵的每一行
                if isinstance(row, np.ndarray):  # 确保 row 是一个可迭代的数组
                    f.write(" ".join(map(str, row.tolist())) + "\n")
                else:  # 如果 row 只是一个单一的 float
                    f.write(str(row) + "\n")
    print("旋转矩阵已保存到文件中。")

def register(
        n_trials,
        n_iter,
        output_folder,
        log_step=30,
):
    output_folder = os.path.join(output_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    mesh, image_label, camera_params, image_label_liver = load_data(
        path_mesh, 
        path_image, 
        path_camera_param,
        image_width=382,
        )
    renderer = setup_render(camera_params,)
#     print("mesh.shape:", mesh.verts_packed().shape, mesh.faces_packed().shape)
#     # 提取顶点、面和纹理数据
#     verts = mesh.verts_packed()  # 顶点数据，形状：[num_vertices, 3]
#     faces = mesh.faces_packed()  # 面数据，形状：[num_faces, 3]
#     textures = mesh.textures  # 纹理数据（如果有的话）

#     # 如果 mesh 没有纹理数据，可以跳过 textures 参数
#     # 使用 save_obj 函数保存 mesh 为 .obj 文件
#     if textures is not None:
#         # 如果有纹理数据，保存时将纹理传递给 save_obj
#         save_obj("mesh_with_textures.obj", verts, faces, textures)
#     else:
#         # 如果没有纹理数据，直接保存顶点和面数据
#         save_obj("mesh_without_textures.obj", verts, faces)
    
    # print("label.shape:", image_label.shape, image_label.max(), image_label.min())


    losses = []
    minminlossindex = 0
    # 打开目标文件
    with open("loss_output.txt", "w") as file:
        for idx_trial in range(n_trials):
            loss_values = []
            print("===========================================================Trial:", idx_trial)
            init_obj_R, init_obj_T = random_init_mesh_pos()
            print("init_obj_R:", init_obj_R)
            print("init_obj_T:", init_obj_T)

            model = Model(
                meshes=mesh, 
                renderer=renderer, 
                dilate_labels=True,
                initial_obj_pos=init_obj_T, #(0,0,0), 
                initial_obj_rot=init_obj_R, #torch.eye(3),
                device=device,
            )
            model = model.to(device)
            # 执行初始对齐，并检查返回值
            if not model.initial_alignment(image_label):
                print("Alignment failed, skipping to the next trial.")
                losses.append(1000000)
                continue  # 如果对齐失败，则跳过当前循环的剩余部分

            model.initial_alignment(image_label)

            lr_T = 1e-9
            lr_R = 1e-5
            optimizer = torch.optim.SGD([
                {"params": [model.T], "lr": lr_T},
                {"params": [model.R], "lr": lr_R},
            ])

            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_iter)

            # init_obj_R, init_obj_T = random_init_mesh_pos()
            # print("random init_obj_R:", init_obj_R)
            # print("random init_obj_T:", init_obj_T)

            trial_dir = os.path.join(output_folder, "trial_{:02d}".format(idx_trial))
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)

            best_loss = 1e9
            loss_list = []
            for idx_iter in range(n_iter):
                if idx_iter%5 == 0:
                    print("--------------------Iteration:", idx_iter)
                mesh_label_rendered, mesh_gray_rendered, mesh_edge_1, R_matrix, T_matrix  = model()
                # print("mesh_label_rendered.shape, mesh_gray_rendered.shape:", mesh_label_rendered.shape, mesh_gray_rendered.shape)

                """
                打印模型此时的R/T矩阵
                """
                print("R_matrix:", R_matrix, idx_iter)
                print("model.T:", model.T, T_matrix)
                """
                重新保存每一个打印的图片
                """
                output_path_lab = os.path.join(trial_dir, "test_rendered_image_{:03d}_lab.png".format(idx_iter))
                transform = torchvision.transforms.ToPILImage()
                img_cpu = transform(mesh_label_rendered.squeeze().permute(2,0,1))
                img_cpu.save(output_path_lab)

                output_path_img = os.path.join(trial_dir, "test_rendered_image_{:03d}_img.png".format(idx_iter))
                transform = torchvision.transforms.ToPILImage()
                img_cpu = transform(mesh_gray_rendered.squeeze().permute(2,0,1))
                img_cpu.save(output_path_img)



                loss, _ = overlap_loss(
                    idx_iter,
                    img = mesh_label_rendered, 
                    ref = image_label, 
                    liver = mesh_gray_rendered, 
                    ref_liver = image_label_liver,
                    ligament_weight=5, silhouette_weight=2, black_mask=False
                )


                # loss = combined_loss(
                #     img = mesh_label_rendered, 
                #     ref = image_label, 
                #     ligament_weight=5, silhouette_weight=2, black_mask=False
                # )
                print("loss:", loss)
                # 将 loss 保留 1 位小数并添加到列表
                loss_values.append(f"{loss:.1f}")

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_list.append(loss.item())
                print("min loss:", min(loss_list))
                if len(losses) > 0:
                    if loss.item() < min(losses):
                        print("idx_iter:", idx_iter)
                        minminlossindex = idx_iter

                # if loss.item() < best_loss:
                #     best_loss = loss.item()
                #     best_T = model.T.clone()
                #     best_R = model.R.clone()


                mesh_label_rendered = mesh_label_rendered.squeeze().detach().cpu().numpy()
                mesh_gray_rendered = mesh_gray_rendered.squeeze().detach().cpu().numpy()

                # # if idx_iter % log_step == 0 or idx_iter < 5:
                # if idx_iter % 2 == 0:
                #     output_path = os.path.join(trial_dir, "test_rendered_image_{:03d}.jpg".format(idx_iter))
                #     plt.figure(figsize=(15, 15))
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(image_label.squeeze().detach().cpu().numpy())
                #     plt.grid(False)
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(mesh_label_rendered[..., :3]) 
                #     plt.grid(False)
                #     plt.subplot(1, 3, 3)
                #     plt.imshow(mesh_gray_rendered[..., :3])
                #     plt.grid(False)
                #     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                #     # print("Saved to:", output_path)

                # 保存当前迭代的 R_matrix
                save_rotation_matrix_to_file(R_matrix)
    
            # file.write(R_matrix.item() + "\n")
            losses.append(min(loss_list))

    print("losses:", losses)
    best_trials = torch.topk(
        torch.tensor(losses), 
        k=int(0.2 * n_trials) if n_trials > 5 else 1, 
        largest=False, 
        sorted=True,
        )
    print("best trials:", best_trials)
    print("minminlossindex:", minminlossindex)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="/home/zxk/code/PyTorch3D/3Dircadb-10.obj")
    parser.add_argument("--label_3d", type=str, default="")
    # parser.add_argument("--label_2d", type=str, default="/home/zxk/code/PyTorch3D/2D_mask/niigz/P2ILF22_patient1_1.nii.g")
    parser.add_argument("--label_2d", type=str, default="/home/zxk/code/PyTorch3D/p2ilf-png-3/P2ILF22_patient3_9.png")
    parser.add_argument("--path_camera_param", type=str, default="/home/zxk/code/PyTorch3D/camera_parameters/patient10/calibration.xml")
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=300)

    args = parser.parse_args()
    path_mesh = args.mesh
    path_mesh_label = args.label_3d
    path_image = args.label_2d
    path_camera_param = args.path_camera_param
    output_folder = args.output_folder
    n_iter = args.n_iter
    n_trials = args.n_trials


    register(
        n_trials=n_trials,
        n_iter=n_iter,
        output_folder=output_folder,
    )




