import numpy as np
import pyvista as pv
import colorsys



# 加载OBJ格式的网格数据
mesh = pv.read("G:/xxx/3Dircadb-10.obj")  # LiTS-22.obj, LiTS-63.obj, LiTS-105.obj, 3Dircadb-10.obj, LiTS-117.obj, LiTS-81.obj

# 定义您的其他函数和数据处理步骤...
def find_vertex_index(mesh, vertex_coord):
    # 计算所有点与目标点的差异
    differences = mesh.points - vertex_coord[:3]
    # 计算每个差异的平方和
    squared_distances = np.sum(differences**2, axis=1)
    # print("squared_distances.shape:", squared_distances.shape)
    # 找到最小的距离
    min_dist_index = np.argmin(squared_distances)
    # 如果最小距离非常接近零，则我们认为找到了对应的顶点
    if np.isclose(squared_distances[min_dist_index], 0):
        return min_dist_index, vertex_coord[3]
    else:
        # print("squared_distances[min_dist_index]:", squared_distances[min_dist_index])
        return None, None
def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


"""
方法的对比
"""


# all_index = read_seg("G:/xxx/PointNet2Plus/3Dircadb-10-vs.eseg")
# all_index = read_seg("G:/xxx/MeshCNN/3Dircadb-10.eseg")
# all_index = read_seg("G:/xxx/MeshCNN-1-355/3Dircadb-10-vs.eseg")
# all_index = read_seg("G:/xxx/TSGCNet/3Dircadb-10-vs.eseg")
all_index = read_seg("G:/xxxx/ours/3Dircadb-10-vs.eseg")
all_index = read_seg("G:/xxxx/Lable-point/3Dircadb-10-vs.eseg")

print("all_index.shape:", all_index.shape)

rendai_index = all_index[all_index[:, 3]==1]
ridge_index = all_index[all_index[:, 3]==2]
# rendai_index = all_index
# ridge_index = all_index
print("rendai_index.shape, ridge_index.shape:", rendai_index.shape, ridge_index.shape)
# print("rendai_index:", rendai_index)
# print("ridge_index:", ridge_index)
# 获取顶点数量
num_points = mesh.n_points
# 创建颜色数组，初始颜色设置为灰色（RGB）
colors = np.ones((num_points, 3)) * [255, 255, 255]  # 所有顶点初始设为灰色

list_vertex_rendai, list_vertex_ganji = [], []


# 将指定索引行号的顶点设置为蓝色
if len(rendai_index) == 0:
    for i in range(rendai_index.shape[0]):
        vertex_index, weight = find_vertex_index(mesh, rendai_index[i])
        # print("weight:", weight)
        if vertex_index is None:
            print(f"Vertex at {rendai_index[i]} not found in the mesh.")
        list_vertex_rendai.append(vertex_index)
        colors[vertex_index] = [0, 0, 255]


# 将另一组指定索引行号的顶点设置为红色
for i in range(ridge_index.shape[0]):
    vertex_index, weight = find_vertex_index(mesh, ridge_index[i])


    if vertex_index is None:
        print(f"Vertex at {ridge_index[i]} not found in the mesh.")
    list_vertex_ganji.append(vertex_index)
    colors[vertex_index] = [255, 0, 0]

print("colors.shape:", colors.shape)
print("list_vertex_rendai:", list_vertex_rendai)
print("list_vertex_ganji:", list_vertex_ganji)


# # 将颜色数组添加到网格数据中
mesh.point_data['colors'] = colors.astype(np.uint8)
print("mesh.point_data:", mesh.point_data)
# 调整网格的大小（缩放）
mesh.scale([2, 2, 2])

# 获取网格的所有边缘
edges = mesh.extract_all_edges()

# 创建一个绘图器
plotter = pv.Plotter()

# 添加带有颜色的网格表面
plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=True)

# 添加边缘，可以指定颜色和线宽
plotter.add_mesh(edges, color='black', line_width=1)

# 开始交互式可视化
plotter.show()



# 交互式会话结束后，获取相机的属性
camera_position = plotter.camera_position
print("Camera Position:", camera_position[0])
print("Focus Point:", camera_position[1])
print("View Up:", camera_position[2])







"""
3Dircadb-10.eseg ；
"""
camera_position = ((365.9306182021362, 109.54889454220735, -0.4977416414122029),
                   (-14.642887896890663, -5.704183105681363, -7.713787722357607),
                   (0.08946999735130778, -0.3537300130247656, 0.9310586434051595))


# 基于打印出的相机属性，用户可以调整以下变量来固定视角
fixed_camera_position = camera_position  # 将这里替换为打印出的值

# 再次设置相机位置以固定视角进行可视化
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=True)
plotter.add_mesh(edges, color='black', line_width=1)
plotter.camera_position = fixed_camera_position
plotter.show()
