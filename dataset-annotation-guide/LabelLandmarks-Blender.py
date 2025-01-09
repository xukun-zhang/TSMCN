'''
设置两个定点组，然后给两个定点组手工设置顶点
'''

import bpy
bpy.ops.object.mode_set(mode='EDIT')
group = bpy.context.object.vertex_groups.new(name="1")
group = bpy.context.object.vertex_groups.new(name="2")


##Assign material to vertex group
#red = bpy.data.materials.new(name='red')
#bpy.context.object.data.materials.append(red)

#blue = bpy.data.materials.new(name='blue')
#bpy.context.object.data.materials.append(blue)



#bpy.data.materials['red'].diffuse_color = (0.938686, 0.0069953, 0.0122865, 1)
#bpy.data.materials['blue'].diffuse_color = (0.0116122, 0.0742136, 0.799103, 1)


#bpy.ops.object.mode_set(mode='EDIT')
#bpy.ops.object.vertex_group_deselect()

#for i in range(1,3):
# bpy.ops.object.vertex_group_set_active(group=str(i))
# bpy.ops.object.vertex_group_select()
# bpy.context.object.active_material_index = i
# bpy.ops.object.material_slot_assign()
# bpy.ops.object.vertex_group_deselect()



'''
将两个定点组保存为.eseg文件，表示分割的标签（注意此时必须处于“目标模式”下运行代码） 
'''
import bpy
import numpy as np

# '''
# 注意：抽取label且生成eseg文件时，必须模式为对象模式，且选中该对象
# '''

verticeGroups = {}
edgeLabels = []
faceLabels = []
edges = []
faces_edges = []
edge_nb = []
edge2key = {}
edges_count = 0
nb_count = []
edge_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/obj-0430/{}.edges'
eseg_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/obj-0430/{}.eseg'


ob = bpy.context.object
obdata = bpy.context.object.data

vgroup_names = {vgroup.index: vgroup.name for vgroup in ob.vertex_groups}
vgroups = {v.index: [vgroup_names[g.group] for g in v.groups] for v in obdata.vertices}

#get group for vertices; save in dict
for idx in range(len(obdata.vertices)):
    group = vgroups[idx]
    verticeGroups[idx] = group     # 这里就是存储所有点的所属组的字典，每一行表示一个点的ID代表的组；

for face in obdata.polygons:
     vert = [face.vertices[0],face.vertices[1],face.vertices[2]]
     for i in range(3):
            cur_edge = (vert[i], vert[(i + 1) % 3])
            faces_edges.append(cur_edge)     # 这里就是存储了所有面的3条边，这样的话存储的边的数量为面的3倍啊；有很多冗余的边，比如一个边的顺序一致的多个边；也有可能顺序不同的多个边；

for idx, edge in enumerate(faces_edges):
     edge = tuple(sorted(list(edge)))
     faces_edges[idx] = edge     # 这里就是将边的列表中的每个边改为元组形式而已，整体的长度不变的；
     if edge not in edge2key:
            edge2key[edge] = edges_count
            edges.append(list(edge))
# 这一行就是将重复的边全部剔除；
np.savetxt(edge_path.format(bpy.context.active_object.name), edges, delimiter=',',fmt='%s')


# 这里要改一下，即只要不是两个顶点同属一组的边，其余的边都按照背景0进行计算；
# 即对于1、2、3组，判断两个顶点是否都是1、2、3，这样就获得了这些边对应的组；
# 然后遍历其余所有边，对于该边的两顶点只要存在不属于1、2、3的全部设置为背景类/组，即0；哪怕其中一个顶点属于1但另外一个顶点属于2呢，仍然设置该边为背景边；
#each edge made of two vertices; get group of first vertice
for idx, edge in enumerate(edges):
      vertix_0 = edge[0]
      vertix_1 = edge[1]
      groups_0 = verticeGroups[vertix_0]
      groups_1 = verticeGroups[vertix_1]
      if groups_0 == groups_1 and len(groups_0)>=1:
        if len(groups_0) == 2:
            edgeLabels.append(groups_0[1])
        elif len(groups_0) == 1:
            edgeLabels.append(groups_0[0])
      else:
         edgeLabels.append(0)

np.savetxt(eseg_path.format(bpy.context.active_object.name), edgeLabels, delimiter=',',fmt='%s')



















import bpy

import numpy as np

labels_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/obj-0430/{}.eseg'

#group = bpy.context.object.vertex_groups.new(name="1")
#group = bpy.context.object.vertex_groups.new(name="2")


obdata = bpy.context.object.data
active_object = bpy.context.active_object
vertex_groups = active_object.vertex_groups[:]

input = labels_path.format(bpy.context.active_object.data.name)
seg_labels = np.loadtxt(open(input, 'r'), dtype='int')
faces_edges = []
edges = []
edge2key = {}
edge_nb = []
edges_count = 0
nb_count = []


dict = {"1": [],
"2": [],
 }

for face in obdata.polygons:
     vert = [face.vertices[0],face.vertices[1],face.vertices[2]]
     for i in range(3):
            cur_edge = (vert[i], vert[(i + 1) % 3])
            faces_edges.append(cur_edge)

for idx, edge in enumerate(faces_edges):
     edge = tuple(sorted(list(edge)))
     faces_edges[idx] = edge
     if edge not in edge2key:
            edge2key[edge] = edges_count
            edges.append(list(edge))
# 上面是对文件中，即label文件中，提取所有边；
#edge labels
for idx, edge in enumerate(edges):
 group = str(seg_labels[idx])
 print(idx,group, edge, idx)
 if seg_labels[idx] == 0:
     continue
 dict[group].append(edge[0])
 dict[group].append(edge[1])


bpy.ops.object.mode_set(mode='OBJECT')

for vertex_group in vertex_groups:
 indices = vertex_group.name
 result = dict[indices]
 vertex_group.add(result, 1.0, 'ADD')


bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.vertex_group_deselect()

for i in range(1,3):
 bpy.ops.object.vertex_group_set_active(group=str(i))
 bpy.ops.object.vertex_group_select()














import bpy

import numpy as np

labels_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/new_data/LiTS-129.eseg'

# group = bpy.context.object.vertex_groups.new(name="1")
# group = bpy.context.object.vertex_groups.new(name="2")


obdata = bpy.context.object.data
active_object = bpy.context.active_object
vertex_groups = active_object.vertex_groups[:]

input = labels_path.format(bpy.context.active_object.data.name)
seg_labels = np.loadtxt(open(input, 'r'), dtype='int')
faces_edges = []
edges = []
edge2key = {}
edge_nb = []
edges_count = 0
nb_count = []

dict = {"1": [],
        "2": [],
        }

for face in obdata.polygons:
    vert = [face.vertices[0], face.vertices[1], face.vertices[2]]
    for i in range(3):
        cur_edge = (vert[i], vert[(i + 1) % 3])
        faces_edges.append(cur_edge)

for idx, edge in enumerate(faces_edges):
    edge = tuple(sorted(list(edge)))
    faces_edges[idx] = edge
    if edge not in edge2key:
        edge2key[edge] = edges_count
        edges.append(list(edge))
# 上面是对文件中，即label文件中，提取所有边；
# edge labels
for idx, edge in enumerate(edges):
    group = str(seg_labels[idx])
    print(idx, group, edge, idx)
    if seg_labels[idx] == 0:
        continue
    dict[group].append(edge[0])
    dict[group].append(edge[1])

bpy.ops.object.mode_set(mode='OBJECT')

for vertex_group in vertex_groups:
    indices = vertex_group.name
    result = dict[indices]
    vertex_group.add(result, 1.0, 'ADD')

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.vertex_group_deselect()

for i in range(1, 3):
    bpy.ops.object.vertex_group_set_active(group=str(i))
    bpy.ops.object.vertex_group_select()

# '''
# 将两个定点组保存为.eseg文件，表示分割的标签（注意此时必须处于“目标模式”下运行代码）
# '''
# import bpy
# import numpy as np

## '''
## 注意：抽取label且生成eseg文件时，必须模式为对象模式，且选中该对象
## '''

# verticeGroups = {}
# edgeLabels = []
# faceLabels = []
# edges = []
# faces_edges = []
# edge_nb = []
# edge2key = {}
# edges_count = 0
# nb_count = []
# edge_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/new_data/{}.edges'
# eseg_path = 'G:/Data/TMI_Couinaud/Train_data/3D landmarks/new_data/{}.eseg'

# ob = bpy.context.object
# obdata = bpy.context.object.data

# vgroup_names = {vgroup.index: vgroup.name for vgroup in ob.vertex_groups}
# vgroups = {v.index: [vgroup_names[g.group] for g in v.groups] for v in obdata.vertices}

##get group for vertices; save in dict
# for idx in range(len(obdata.vertices)):
#    group = vgroups[idx]
#    verticeGroups[idx] = group     # 这里就是存储所有点的所属组的字典，每一行表示一个点的ID代表的组；

# for face in obdata.polygons:
#     vert = [face.vertices[0],face.vertices[1],face.vertices[2]]
#     for i in range(3):
#            cur_edge = (vert[i], vert[(i + 1) % 3])
#            faces_edges.append(cur_edge)     # 这里就是存储了所有面的3条边，这样的话存储的边的数量为面的3倍啊；有很多冗余的边，比如一个边的顺序一致的多个边；也有可能顺序不同的多个边；

# for idx, edge in enumerate(faces_edges):
#     edge = tuple(sorted(list(edge)))
#     faces_edges[idx] = edge     # 这里就是将边的列表中的每个边改为元组形式而已，整体的长度不变的；
#     if edge not in edge2key:
#            edge2key[edge] = edges_count
#            edges.append(list(edge))
## 这一行就是将重复的边全部剔除；
# np.savetxt(edge_path.format(bpy.context.active_object.name), edges, delimiter=',',fmt='%s')


## 这里要改一下，即只要不是两个顶点同属一组的边，其余的边都按照背景0进行计算；
## 即对于1、2、3组，判断两个顶点是否都是1、2、3，这样就获得了这些边对应的组；
## 然后遍历其余所有边，对于该边的两顶点只要存在不属于1、2、3的全部设置为背景类/组，即0；哪怕其中一个顶点属于1但另外一个顶点属于2呢，仍然设置该边为背景边；
##each edge made of two vertices; get group of first vertice
# for idx, edge in enumerate(edges):
#      vertix_0 = edge[0]
#      vertix_1 = edge[1]
#      groups_0 = verticeGroups[vertix_0]
#      groups_1 = verticeGroups[vertix_1]
#      if groups_0 == groups_1 and len(groups_0)>=1:
#        if len(groups_0) == 2:
#            edgeLabels.append(groups_0[1])
#        elif len(groups_0) == 1:
#            edgeLabels.append(groups_0[0])
#      else:
#         edgeLabels.append(0)

# np.savetxt(eseg_path.format(bpy.context.active_object.name), edgeLabels, delimiter=',',fmt='%s')