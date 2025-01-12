import os
import numpy as np
# 指定你的文件夹路径
folder_path = 'G:/Data/xxx/real_test'
folder_path_eseg = 'G:/result/seed_2026'
folder_path_cd = 'G:/result-2/seed_2026'
def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels

ave_pre, ave_rec, ave_f1, num = 0, 0, 0, 0
# 遍历文件夹
for filename in os.listdir(folder_path):

    if filename.endswith('.obj'):
        # 拼接完整的文件路径
        file_path = os.path.join(folder_path, filename)
        if filename[-4:] == ".obj":
            num = num + 1

            label_name = filename[:-4] + "-pred.eseg"
            label_path = os.path.join(folder_path_eseg, label_name)
            obj_path = os.path.join(folder_path, filename)

            vs, faces = [], []
            f = open(obj_path)
            for line in f:
                line = line.strip()
                splitted_line = line.split()
                if not splitted_line:
                    continue
                elif splitted_line[0] == 'v':
                    vs.append([float(v) for v in splitted_line[1:4]])
                elif splitted_line[0] == 'f':
                    face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                    assert len(face_vertex_ids) == 3
                    face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                    faces.append(face_vertex_ids)
            f.close()
            vs = np.asarray(vs)
            faces = np.asarray(faces, dtype=int)

            edge_nb = []
            sides = []
            edge2key = dict()
            edges = []
            edges_count = 0
            nb_count = []
            for face_id, face in enumerate(faces):
                faces_edges = []
                for i in range(3):
                    cur_edge = (face[i], face[(i + 1) % 3])
                    faces_edges.append(cur_edge)
                for idx, edge in enumerate(faces_edges):
                    edge = tuple(sorted(list(edge)))
                    faces_edges[idx] = edge
                    if edge not in edge2key:
                        edge2key[edge] = edges_count
                        edges.append(list(edge))


                        nb_count.append(0)
                        edges_count += 1
            edges = np.array(edges, dtype=np.int32)
            print("edges.shape, edges.max(), edges.min():", filename, edges.shape, edges.max(), edges.min(), vs.shape)

            y_true = read_seg(label_path)
            print("y_true.shape:", y_true.shape)
            print("y_true.shape:", y_true.shape)
            save_edge = []
            for i in range(edges.shape[0]):
                index_v, index_y = edges[i, 0], edges[i, 1]
                vs_xyz = (vs[index_v] + vs[index_y])/2.0
                # print("vs[index_v], vs[index_y], vs_xyz:", vs[index_v], vs[index_y], vs_xyz)

                save_edge.append([])

                save_edge[i].extend(vs_xyz)
                save_edge[i].append(y_true[i])


            file_name_label = filename[:-4] + ".eseg"
            path_file_i_l = os.path.join(folder_path_cd, file_name_label)

            np.savetxt(path_file_i_l, save_edge, fmt='%f')


# f.close()
# vs = np.asarray(vs)