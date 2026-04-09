import copy
import operator

bl_info = {
    "name": "Star Wars Outlaws Mesh Tool",
    "author": "AlexPo",
    "location": "Scene Properties > Star Wars: Outlaws Mesh Tool Panel",
    "version": (0, 0, 5),
    "blender": (5, 0, 0),
    "description": "This addon imports/exports skeletal meshes\n from Star Wars Outlaws's .mmb files",
    "category": "Import-Export"
    }

import bpy
import bmesh
from struct import unpack, pack
import numpy as np
import math
from mathutils import Matrix, Euler, Vector
from pathlib import Path
import os
import io


class ByteReader:
    @staticmethod
    def int8(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        return i
    @staticmethod
    def bool(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        if i == 0:
            return False
        elif i == 1:
            return True
        else:
            raise Exception("Byte at {v} wasn't a boolean".format(v=f.tell()))
    @staticmethod
    def uint8(f):
        b = f.read(1)
        i = unpack('<B', b)[0]
        return i
    @staticmethod
    def int16(f):
        return unpack('<h', f.read(2))[0]
    @staticmethod
    def uint16(f):
        b = f.read(2)
        i = unpack('<H', b)[0]
        return i
    @staticmethod
    def hash(f):
        b = f.read(8)
        return b
    @staticmethod
    def guid(f):
        return f.read(16)
    @staticmethod
    def int32(f):
        b = f.read(4)
        i = unpack('<i',b)[0]
        return i
    @staticmethod
    def uint32(f):
        b = f.read(4)
        i = unpack('<I',b)[0]
        return i
    @staticmethod
    def uint64(f):
        b = f.read(8)
        i = unpack('<Q',b)[0]
        return i
    @staticmethod
    def int64(f):
        b = f.read(8)
        i = unpack('<q', b)[0]
        return i
    @staticmethod
    def string(f,length):
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def name(f):
        return br.string(f,br.uint16(f))
    @staticmethod
    def path(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def hashtext(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        f.seek(4,1)
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def float(f):
        b = f.read(4)
        fl = unpack('<f',b)[0]
        return fl
    @staticmethod
    def vector3(f):
        b = f.read(12)
        return unpack('<fff', b)
    @staticmethod
    def dvector3(f):
        #double vector 3
        b = f.read(24)
        return unpack('<ddd', b)
    @staticmethod
    def vector4(f):
        b = f.read(16)
        return unpack('<ffff', b)
    @staticmethod
    def int16_norm(f):
        i = unpack('<H', f.read(2))[0]
        v = i ^ 2**15
        v -= 2**15
        v /= 2**15 - 1
        return v
    @staticmethod
    def uint16_norm(f):
        int16 = unpack('<H', f.read(2))[0]
        return int16 / 2 ** 16
    @staticmethod
    def uint8_norm(f):
        uint8 = unpack('<B', f.read(1))[0]
        maxint = (2 ** 8)-1
        return uint8 / maxint
    @staticmethod
    def int8_norm(f):
        int8 = unpack('<B', f.read(1))[0]
        v = int8 ^ 2**7
        v -= 2**7
        v /= 2**7 -1
        return v
    @staticmethod
    def X10Y10Z10W2_normalized(f):
        i = unpack('<I', f.read(4))[0]  # get 32bits of data

        x = i >> 0
        x = ((x & 0x3FF) ^ 512) - 512

        y = i >> 10
        y = ((y & 0x3FF) ^ 512) - 512

        z = i >> 20
        z = ((z & 0x3FF) ^ 512) - 512

        w = i >> 30
        w = w & 0x1

        vectorLength = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        # # print(x,y,z)
        if vectorLength != 0:
            x /= vectorLength
            y /= vectorLength
            z /= vectorLength
        return [x, y, z, w]
    @staticmethod
    def matrix_4x4(f):
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        for i in range(4):
            for c in range(4):
                value = br.float(f)
                if c == 0:
                    row1.append(value)
                if c == 1:
                    row2.append(value)
                if c == 2:
                    row3.append(value)
                if c == 3:
                    row4.append(value)
        # print(Matrix((row1,row2,row3,row4)))
        matrix = Matrix((row1, row2, row3, row4))#.inverted()
        return matrix
class BytePacker:
    @staticmethod
    def int8(v):
        return pack('<b', v)
    @staticmethod
    def uint8(v):
        return pack('<B', v)
    @staticmethod
    def uint8_norm(v):
        if 0.0 <= v <= 1.0:
            i = max(0,min(int(v * ((2 ** 8)-1)),255))
        else:
            raise Exception("Couldn't normalize value as uint8Norm, "
                            "it wasn't between 0.0 and 1.0. Unknown max value."
                            +str(v))
        return pack('<B', i)
    @staticmethod
    def int16(v):
        return pack('<h', v)
    @staticmethod
    def uint16(v):
        return pack('<H', v)
    @staticmethod
    def int16_norm(v):
        # print(v)
        if -1.0 <= v <= 1.0:
            # if v >= 0:
            #     v = int(abs(v) * (2 ** 15))
            # else:
            #     v = 2 ** 16 - int(abs(v) * (2 ** 15))
            v = max(min(int(v * (2 ** 15)),32767), -32768)
        else:
            raise Exception("Couldn't normalize value as int16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<h', v)
    @staticmethod
    def uint16_norm(v):
        if 0.0 < v < 1.0:
            i = v * (2 ** 16) - 1
            i = int(i)
        else:
            raise Exception("Couldn't normalize value as uint16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<H', i)
    @staticmethod
    def float16(v):
        f32 = np.float32(v)
        f16 = f32.astype(np.float16)
        b16 = f16.tobytes()
        return b16
    @staticmethod
    def int32(v):
        return pack('<i', v)
    @staticmethod
    def uint32(v):
        return pack('<I', v)
    @staticmethod
    def uint64(v):
        return pack('<Q', v)
    @staticmethod
    def int64(v):
        return pack('<q', v)
    @staticmethod
    def float(v):
        return pack('<f', v)
    @staticmethod
    def X10Y10Z10W2(x,y,z,w):
        if x >= 0:
            x = int(abs(x) * 2 ** 9)
        else:
            x = 2**10 - int(abs(x) * 2 ** 9)
        if y >= 0:
            y = int(abs(y) * 2 ** 9)
        else:
            y = 2**10 - int(abs(y) * 2 ** 9)
        if z >= 0:
            z = int(abs(z) * 2 ** 9)
        else:
            z = 2**10 - int(abs(z) * 2 ** 9)


        w = int(w)


        x = (abs(x) & 0x3FF)
        y = (abs(y) & 0x3FF) << 10
        z = (abs(z) & 0x3FF) << 20
        w = (abs(w) & 0x3) << 30

        v = x | y | z | w
        return pack("<I", v)

br = ByteReader
bp = BytePacker

def CopyFile(read,write,offset,size,buffer_size=500000):
    read.seek(offset)
    chunks = size // buffer_size
    for o in range(chunks):
        write.write(read.read(buffer_size))
    write.write(read.read(size%buffer_size))
def get_merged_mmb(mmb):
    files = []
    if str(mmb).endswith("mmb"):
        files.append(mmb)
    else:
        i = 0
        while True:
            current_file = f"{str(mmb)[:-1]}{i}"
            if os.path.isfile(current_file):
                files.append(current_file)
                i += 1
            else:
                break

    f = io.BytesIO()
    for file_dir in files:
        with open(file_dir, 'rb') as file:
            f.write(file.read())
    return f

class Asset:
    def __init__(self):
        self.magic = ""
        self.version = 0
        self.size = 0
    def parse(self,f):
        self.magic = br.string(f,3)
        self.version = br.uint8(f)
        self.size = br.uint32(f)
        f.seek(4,1)
    def write(self,f):
        f.seek(4)
        f.write(bp.uint32(self.size))

class SkeletalMeshAsset(Asset):
    class Mesh:
        class LOD:
            def __init__(self, parent_mesh,index):
                self.start_offset = 0
                self.parent_mesh:SkeletalMeshAsset.Mesh = parent_mesh
                self.index = index
                self.vertex_count = 0
                self.index_count = 0
                self.size_a = 0
                self.vertex_data_offset_a = 0
                self.vertex_data_offset_b = 0
                self.face_block_offset = 0
                self.data_offset = 0
                self.data_size = 0
                self.data_x_offset = 0
                self.data_x_size = 0
                self.data_y_offset = 0
                self.data_y_size = 0
                self.local_vertex_stride = 0
                self.lod_info_type = 0
                self.is_header_lod = False
                self.vertex_end_bytes = None
                self.normals_end_bytes = None
                self.faces_end_bytes = None

            def parse(self, f, lod_info_type = 0):
                self.lod_info_type = lod_info_type
                self.start_offset = f.tell()
                self.vertex_count = br.uint32(f)
                self.index_count = br.uint32(f)
                self.size_a = br.uint32(f) # seems to be face_block_offset divided by 2
                self.vertex_data_offset_a = br.uint32(f)
                self.vertex_data_offset_b = br.uint32(f)
                self.face_block_offset = br.uint32(f)
                self.data_offset = br.uint32(f)
                self.data_size = br.uint32(f)
                lod_screen_size = br.float(f)  # not confirmed screen size
                if lod_info_type == 0:
                    pass
                elif lod_info_type == 12:
                    unk1 = br.uint32(f)
                    self.data_x_offset = br.uint32(f)
                    self.data_x_size = br.uint32(f)
                    unk2 = br.uint32(f)
                    self.data_y_offset = br.uint32(f)
                    self.data_y_size = br.uint32(f)
                    self.local_vertex_stride = int(self.data_y_size / self.vertex_count)
                    print("New Vertex Stride = ", self.local_vertex_stride)
                else:
                    f.seek(28,1)
                if self.data_offset < self.parent_mesh.parent_sk_mesh.size:
                    self.is_header_lod = True
                    print("Lod ", self.index, "is in header.")

            def write(self, f):
                f.seek(self.start_offset)
                f.write(bp.uint32(self.vertex_count))
                f.write(bp.uint32(self.index_count))
                f.write(bp.uint32(int(self.face_block_offset / 2)))
                f.write(bp.uint32(self.vertex_data_offset_a))
                f.write(bp.uint32(self.vertex_data_offset_b))
                f.write(bp.uint32(self.face_block_offset))
                f.write(bp.uint32(self.data_offset))
                f.write(bp.uint32(self.data_size))

            def gather_extra_bytes(self,f):
                offset = f.tell()
                f.seek(self.data_offset)
                real_vertex_size = self.vertex_count * self.parent_mesh.vertex_stride
                if self.vertex_data_offset_a == self.vertex_data_offset_b:
                    extra_bytes_size = self.face_block_offset - self.vertex_data_offset_a - real_vertex_size
                else:
                    extra_bytes_size = self.vertex_data_offset_b - self.vertex_data_offset_a - real_vertex_size
                f.seek(real_vertex_size, 1)
                print(f.tell())
                self.vertex_end_bytes = f.read(extra_bytes_size)
                print("Vertex Extra Bytes:",self.vertex_end_bytes)

                if not self.vertex_data_offset_a == self.vertex_data_offset_b:
                    real_normals_size = self.vertex_count * self.parent_mesh.normals_stride
                    extra_bytes_size = self.face_block_offset - self.vertex_data_offset_b - real_normals_size
                    f.seek(real_normals_size, 1)
                    print(f.tell())
                    self.normals_end_bytes = f.read(extra_bytes_size)
                    print("Normal Extra Bytes:",self.normals_end_bytes)

                real_face_size = self.index_count * 2
                size_without_face = self.face_block_offset - self.vertex_data_offset_a
                extra_bytes_size = self.data_size - size_without_face - real_face_size
                f.seek(real_face_size, 1)
                print(f.tell())
                self.faces_end_bytes = f.read(extra_bytes_size)
                print("Face Extra Bytes:",self.faces_end_bytes)

                f.seek(offset)

            def get_vertex_positions(self,raw_mesh_file):
                vertices = []
                if self.local_vertex_stride > 0:
                    stride = self.local_vertex_stride
                else:
                    stride = self.parent_mesh.vertex_stride
                print(stride)
                if self.data_y_offset != 0:
                    print("Vertex Data is in mmb file.")
                    SWOMT = bpy.context.scene.SWOMT
                    file = SWOMT.AssetPath
                    f = open(file,"rb")
                    print(self.data_y_offset)
                    f.seek(self.data_y_offset)
                else:
                    f = raw_mesh_file
                    f.seek(self.vertex_data_offset_a)
                pos = (0.0,0.0,0.0)
                for v in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.position_type == 0:
                        x = br.int16_norm(f)
                        y = br.int16_norm(f)
                        z = br.int16_norm(f)
                        w = br.int16(f)
                        pos = (x*w,y*w,z*w)
                    elif self.parent_mesh.position_type == 1:
                        x = br.float(f)
                        y = br.float(f)
                        z = br.float(f)
                        pos = (x,y,z)

                    f.seek(stride_start + stride)
                    if v  == 0:
                        print(f.tell())
                    vertices.append(pos)
                return vertices

            def get_bone_weights(self, raw_mesh_file):
                bone_weights = []
                pos_length = 0  # size of vertex position in stride
                if self.lod_info_type == 12:
                    stride = self.local_vertex_stride
                    SWOMT = bpy.context.scene.SWOMT
                    file = SWOMT.AssetPath
                    f = open(file, "rb")
                    f.seek(self.data_y_offset)
                else:
                    stride = self.parent_mesh.vertex_stride
                    f = raw_mesh_file
                    f.seek(self.vertex_data_offset_a)
                if self.parent_mesh.position_type == 0:
                    pos_length = 8
                elif self.parent_mesh.position_type == 1:
                    pos_length = 12
                else:
                    pos_length = 12
                weight_type = self.parent_mesh.vertex_weight_type
                weight_count = 8
                if weight_type == 0:
                    weight_count = int((stride - pos_length) / 2)
                elif weight_type == 1:
                    weight_count = 12
                for v in range(self.vertex_count):
                    vertex_stride_start = f.tell()
                    f.seek(pos_length,1)
                    weights = []
                    iw = {}
                    for w in range(weight_count):
                        weight = br.uint8_norm(f)
                        if weight > 0.0:
                            weights.append(weight)

                    for i in range(weight_count):
                        if i < len(weights):
                            if weight_type == 1:
                                iw[br.uint16(f)] = weights[i]
                            else:
                                iw[br.uint8(f)] = weights[i]
                        else:
                            if weight_type == 1:
                                f.seek(2,1)
                            else:
                                f.seek(1,1)
                    f.seek(vertex_stride_start + stride)

                    bone_weights.append(iw)
                return bone_weights

            def get_triangles(self,raw_mesh_file):
                """
                Seeks to Lod.face_block_offset and reads all triangle indices.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 3 vertex indices to form a triangle.
                """
                tris = []
                f = raw_mesh_file
                f.seek(self.face_block_offset)
                tris_count = int(self.index_count/3)
                if self.vertex_count == self.index_count:
                    index_count = int(self.size_a / 4)
                    print(index_count, self.size_a)
                    tris_count = int(index_count/3)
                print("Triangles Count:",tris_count)
                for i in range(tris_count):
                        f1 = br.uint16(f)
                        f2 = br.uint16(f)
                        f3 = br.uint16(f)
                        tris.append((f1,f2,f3))
                return tris

            def get_normals_size(self):
                if self.parent_mesh.normal_type == 0:
                    return 8
                else:
                    return 28

            def get_normals(self,raw_mesh_file):
                normals = []
                stride = self.parent_mesh.normals_stride
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                v = Vector((0.0,0.0,1.0))
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.normal_type == 0:
                        x = br.int8_norm(f) *-1
                        y = br.int8_norm(f)
                        z = br.int8_norm(f)
                        w = br.int8(f)
                        v = Vector((x*w, y*w, z*w)).normalized()
                        v.negate()  # TODO not sure about this
                    elif self.parent_mesh.normal_type == 1:
                        x = br.float(f) *-1
                        y = br.float(f)
                        z = br.float(f)
                        v = Vector((x,y,z)).normalized()
                    f.seek(stride_start + stride)
                    normals.append(v)
                return normals

            def get_uvs(self,raw_mesh_file, index=0):
                """
                Seeks to Lod.vertex_data_offset_b and reads all UV data.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 2 floats as UV coordinates.
                """
                uvs = []
                color_count = self.parent_mesh.color_count
                f = raw_mesh_file
                if self.lod_info_type == 12:
                    stride = self.parent_mesh.vertex_stride
                    f.seek(self.vertex_data_offset_a)
                    for i in range(self.vertex_count):
                        stride_start = f.tell()
                        f.seek(4 * color_count, 1)
                        u = br.int16_norm(f)
                        v = br.int16_norm(f)
                        f.seek(stride_start + stride)
                        uvs.append((u,v))
                else:
                    stride = self.parent_mesh.normals_stride
                    f.seek(self.vertex_data_offset_b)
                    for i in range(self.vertex_count):
                        stride_start = f.tell()
                        f.seek(self.get_normals_size(), 1) #skip normals
                        f.seek(4 * color_count, 1) #skip color
                        f.seek(index * 4, 1)  # skip previous uv
                        u = br.int16_norm(f)
                        v = br.int16_norm(f)
                        f.seek(stride_start + stride)
                        uvs.append((u,v))
                return uvs

            def get_color(self,raw_mesh_file, index=0):
                """
               Seeks to Lod.vertex_data_offset_b and reads all Color data.
               :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
               :return: a List of Tuples containing 4 floats as RGBA coordinates.
               """
                colors = []
                f = raw_mesh_file
                if self.lod_info_type == 12:
                    stride = self.parent_mesh.vertex_stride
                    f.seek(self.vertex_data_offset_a)
                    for i in range(self.vertex_count):
                        stride_start = f.tell()
                        f.seek(index * 4, 1) #skip previous color
                        r = br.uint8_norm(f)
                        g = br.uint8_norm(f)
                        b = br.uint8_norm(f)
                        a = br.uint8_norm(f)
                        f.seek(stride_start + stride)
                        colors.append((r,g,b,a))
                else:
                    stride = self.parent_mesh.normals_stride
                    f.seek(self.vertex_data_offset_b)
                    for i in range(self.vertex_count):
                        stride_start = f.tell()
                        f.seek(self.get_normals_size(), 1) #skip normals
                        f.seek(index * 4, 1) #skip previous color
                        r = br.uint8_norm(f)
                        g = br.uint8_norm(f)
                        b = br.uint8_norm(f)
                        a = br.uint8_norm(f)
                        f.seek(stride_start + stride)
                        colors.append((r,g,b,a))
                return colors

            def write_vertex_position(self,file,pos=(0.0,0.0,0.0),scale=2):
                """
                Writes a single vertex position into file at the current position and skips to the end of stride.
                :param file: file to write on
                :param pos: (x,y,z)
                :return:
                """
                f = file
                x = pos[0]
                y = pos[1]
                z = pos[2]
                stride = self.parent_mesh.vertex_stride
                stride_start = f.tell()
                if self.parent_mesh.position_type == 0:
                    f.write(bp.int16_norm(x / scale))
                    f.write(bp.int16_norm(y / scale))
                    f.write(bp.int16_norm(z / scale))
                    f.write(bp.int16(scale))
                elif self.parent_mesh.position_type == 1:
                    f.write(bp.float(x))
                    f.write(bp.float(y))
                    f.write(bp.float(z))
                f.seek(stride_start + stride)

        def __init__(self,parent_sk_mesh,index = 0):
            self.parent_sk_mesh:SkeletalMeshAsset = parent_sk_mesh
            if self.parent_sk_mesh.bone_count > 0xFF:
                self.vertex_weight_type = 1  # 1: uint16_norm
            else:
                self.vertex_weight_type = 0  # 0: uint8_norm
            self.index = index
            self.name = ""
            self.lod_count_offset = 0
            self.lod_count = 0
            self.lods = []
            self.vertex_stride = 0
            self.normals_stride = 0

            self.mesh_bones = {}
            self.real_bone_indices_to_mesh_bones = {}
            self.color_count = 0
            self.uv_count = 0
            self.normal_type = 0 # 0:int8_norm 1:floats
            self.position_type = 0 # 0:int16_norm 1:floats
            self.end_bytes = None # bytes from the end of the last LOD to the end of the mesh section.
            self.mesh_file = None # BytesIO of reversed ordered LOD mesh data.

        def parse(self, f):
            print("Mesh Start Offset:", f.tell())
            self.name = br.name(f)
            f.seek(48, 1)  # some kind of matrix
            f.seek(1, 1)
            x_count = br.uint16(f)
            f.seek(4 * x_count, 1)
            y_count = br.uint16(f)
            f.seek(4 * y_count, 1)
            print("Binding Count Offset:", f.tell())
            binding_count = br.uint16(f)
            for b in range(binding_count):
                matrix = br.matrix_4x4(f)
                bone_index = br.uint16(f)
                self.mesh_bones[bone_index] = matrix
                self.real_bone_indices_to_mesh_bones[bone_index] = b
                print(bone_index, " = ", b)
            f.seek(2, 1)
            flag_1, flag_2, flag_3 = 0, 0, 0
            if y_count > 0:
                flag_1 = br.uint8(f)
                flag_2 = br.uint8(f)
                flag_3 = br.uint8(f)
            lod_info_type = br.uint16(f)
            self.lod_count_offset = f.tell()
            print("Lod Count Offset: ",self.lod_count_offset)
            self.lod_count = br.uint8(f)
            f.seek(4, 1)
            for l in range(self.lod_count):
                lod = self.LOD(self,l)
                lod.parse(f, lod_info_type)
                self.lods.append(lod)
            print("LOD Info End Offset:", f.tell())
            #this section could be wrong
            self.uv_count = br.uint8(f)
            f.seek(4*self.uv_count,1)
            self.color_count = br.uint8(f)
            f.seek(4*self.color_count,1)
            unk = br.uint32(f)
            count_c = br.uint8(f)
            f.seek(4*count_c,1)
            if lod_info_type == 12:
                f.seek(2,1)
            print("Vertex Stride Offset:", f.tell())
            self.vertex_stride = br.uint16(f)
            self.normals_stride = br.uint16(f)

            if lod_info_type == 12:
                if flag_3 == 1:
                    f.seek(14, 1)
                    pca_count = br.uint32(f)
                    print("PCA Count : ",pca_count)
                    for pca in range(pca_count):
                        pca_length = br.uint16(f)
                        f.seek(pca_length, 1)
                        f.seek(4, 1)
                else:
                    f.seek(18, 1)
            else:
                f.seek(20, 1)
            print("PCA End Offset:", f.tell())
            # Gather extra bytes at the end of each LOD's vertex/normal/face data.
            # Must be done after getting stride lengths.
            for l in self.lods:
                l.gather_extra_bytes(f)

            # guessing the type of normal data int8 or floats
            if self.normals_stride - (4 * self.uv_count) - (4 * self.color_count) > 8:
                self.normal_type = 1
            else:
                self.normal_type = 0

            # guessing the type of vertex position data int16 or floats
            if self.vertex_stride - 16 == 8 or self.vertex_stride - 8 == 8:
                self.position_type = 0 #int16
            elif self.vertex_stride - 16 == 12 or self.vertex_stride - 8 == 12:
                self.position_type = 1 #float
            else:
                self.position_type = 1 #float
            print(f'\nName = {self.name}'
                  f'\nVertex Stride: {self.vertex_stride}'
                  f'\nNormals Stride: {self.normals_stride}'
                  f'\nUV Count: {self.uv_count}'
                  f'\nColor Count: {self.color_count}')
        def write(self, f):
            f.seek(self.lod_count_offset)
            lod_count = br.uint8(f)
            f.seek(4, 1)
            for l in range(self.lod_count):
                self.lods[l].write(f)

        def extract_mesh_file(self,f):
            """
            Creates a file gathering the raw data of all Lods the Mesh.
            :param f: combined mmb file that contains the header and data.
            :return: Path to the extracted raw_mesh file.
            """
            extract_file = io.BytesIO()
            for lod in reversed(self.lods):
                # print(f'Copy at {lod.data_offset} size {lod.data_size}')
                CopyFile(f, extract_file, lod.data_offset, lod.data_size)
            # print(f'Total size: {sum(lod.data_size for lod in self.lods)}')
            return extract_file

    class Bone:
        def __init__(self,f):
            self.name = br.name(f)
            self.matrix = br.matrix_4x4(f)
            self.parent_index = br.uint16(f)

    def __init__(self):
        super().__init__()
        self.name = ""
        self.bone_count = 0
        self.bones = []
        self.mesh_count = 0
        self.meshes = []
        self.end_bytes = None # bytes at the end of the meshes sections
    def parse(self,f):
        super().parse(f)
        print("Bone Count Offset:", f.tell())
        self.bone_count = br.uint32(f)
        for b in range(self.bone_count):
            self.bones.append(self.Bone(f))
        print("Mesh Count Offset:", f.tell())
        self.mesh_count = br.uint32(f)
        for m in range(self.mesh_count):
            mesh = self.Mesh(self,index = m)
            mesh.parse(f)
            self.meshes.append(mesh)
        length =  br.uint32(f)
        f.seek(-4,1)
        self.end_bytes = f.read(length)
    def write(self, f):
        super().write(f)
        f.seek(12)
        bone_count = br.uint32(f)
        for b in range(bone_count):
            self.Bone(f)
        mesh_count = br.uint32(f)
        for m in range(mesh_count):
            self.meshes[m].write(f)
    def clear(self):
        self.bones = []
        self.meshes = []
        self.end_bytes = None
    def get_sorted_lods(self):
        lod_map = {}
        for m in self.meshes:
            for l in m.lods:
                lod_map[l] = l.data_offset
        sorted_lods = {k: v for k, v in sorted(lod_map.items(), key=lambda item: item[1])}
        return sorted_lods
    def get_mesh_data_start_offset(self):
        lods = self.get_sorted_lods()
        first_lod = next(iter(lods))
        print("Mesh Data Start Offset = ",first_lod.data_offset)
        return first_lod.data_offset

class BlenderMeshImporter:
    @staticmethod
    def find_or_create_collection(name):
        index = bpy.data.collections.find(name)
        if index != -1:
            return bpy.data.collections[index]
        else:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
            return collection


    @staticmethod
    def import_mesh(file, skeletal_mesh:SkeletalMeshAsset, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        # Extract raw mesh file
        raw_mesh_file = mesh.extract_mesh_file(file)

        # Create Mesh and Object
        obj_name = f'{mesh.name}_LOD{lod_index}'
        obj_data = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, obj_data)
        collection = BMI.find_or_create_collection(skeletal_mesh.name)
        collection.objects.link(obj)

        # Create BMesh
        bm = bmesh.new()
        bm.from_mesh(obj_data)

        lod = mesh.lods[lod_index]
        # Import vertices
        verts = lod.get_vertex_positions(raw_mesh_file)
        for v in verts:
            bmv = bm.verts.new()
            v_co = (v[0]*-1,v[1],v[2])
            bmv.co = v_co
        bm.verts.ensure_lookup_table()
        # Import triangles
        triangles = lod.get_triangles(raw_mesh_file)
        for tris in triangles:
            face_vertices = []
            for v_index in tris:
                tv = bm.verts[v_index]
                face_vertices.append(tv)
            # print(tris)
            bm_face = bm.faces.new(face_vertices)
            bm_face.normal_flip() #this is required because the *-1 on x vertex co flips the mesh normals
        bm.to_mesh(obj_data)
        bm.free()
        bm = bmesh.new()
        bm.from_mesh(obj_data)
        bm.faces.ensure_lookup_table()
        # Import UVs
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file,uv_index)
            uv_layer = bm.loops.layers.uv.new(f'UVMap_{uv_index}')
            for finder, face in enumerate(bm.faces):
                for lindex, loop in enumerate(face.loops):
                    v_index = loop.vert.index
                    v_uv = (uvs[v_index][0], 1 - uvs[v_index][1])
                    loop[uv_layer].uv = v_uv

        # Import Colors
        for color_index in range(mesh.color_count):
            colors = lod.get_color(raw_mesh_file, color_index)
            color_layer = bm.verts.layers.float_color.new(f"Color_{color_index}")
            for v in bm.verts:
                v[color_layer] = colors[v.index]
        bm.to_mesh(obj_data)
        bm.free()
        obj_data.update()

        # Import Normals
        obj_data.normals_split_custom_set_from_vertices(lod.get_normals(raw_mesh_file))

        # Import Bone Weights
        weights = lod.get_bone_weights(raw_mesh_file)
        mesh_bones = list(mesh.mesh_bones.keys())
        for bone in skeletal_mesh.bones:
            obj.vertex_groups.new(name=bone.name)
        for v_index in range(lod.vertex_count):
            v_bone_weights = weights[v_index]
            for bone_index in v_bone_weights.keys():
                if bone_index < len(mesh_bones):
                    real_bone_index = mesh_bones[bone_index] # Convert mesh bone index to skeleton bone index
                    bone_name = skeletal_mesh.bones[real_bone_index].name
                    obj.vertex_groups[bone_name].add([v_index],v_bone_weights[bone_index], "ADD")
        return obj

    @staticmethod
    def find_or_create_skeleton(skeletal_mesh:SkeletalMeshAsset):
        index = bpy.data.objects.find(skeletal_mesh.name)
        if index != -1:
            return bpy.data.objects[index]
        else:
            return BMI.import_skeleton(skeletal_mesh)

    @staticmethod
    def import_skeleton(skeletal_mesh:SkeletalMeshAsset):
        _armature = bpy.data.armatures.new(skeletal_mesh.name)
        _obj = bpy.data.objects.new(skeletal_mesh.name, _armature)
        collection = BMI.find_or_create_collection(skeletal_mesh.name)
        collection.objects.link(_obj)
        bpy.context.view_layer.objects.active = _obj
        bpy.ops.object.mode_set(mode='EDIT')
        for i,b in enumerate(skeletal_mesh.bones):
            bone = _armature.edit_bones.new(b.name)
            parent_index = b.parent_index
            if b.parent_index == 65535:
                parent_index = -1
            bone.parent = _armature.edit_bones[parent_index]
            bone.tail = Vector([0.0,0.0,0.1])
            parent_matrix = Matrix()
            if bone.parent:
                parent_matrix = bone.parent.matrix
            bone.matrix = parent_matrix @ b.matrix
        scale_matrix = Matrix().Scale(-1.0, 4, Vector((1.0, 0.0, 0.0)))
        _armature.transform(scale_matrix)
        bpy.ops.object.mode_set(mode='OBJECT')
        return _obj
    @staticmethod
    def parent_obj_to_armature(obj,armature):
        obj.modifiers.new(name='Armature', type='ARMATURE')
        obj.modifiers['Armature'].object = armature
        obj.parent = armature
    @staticmethod
    def rotate_model(obj,armature):
        # armature.rotation_euler[0] = math.radians(90)
        # bpy.ops.object.transform_apply(rotation = True)
        rot = Euler(map(math.radians,(90,0,0)),'XYZ')
        mat = rot.to_matrix().to_4x4()
        armature.matrix_world = mat
        bpy.ops.object.transform_apply(rotation=True)

class BlenderMeshExporter:
    @staticmethod
    def find_object_by_name(name=""):
        obj = None
        try:
            obj = bpy.data.objects[name]
        except KeyError:
            raise KeyError(f"{name} object was not found.") from None
        return obj
    @staticmethod
    def copy_mmb_file():
        """
        Takes the merged mmb file and creates a copy of it.
        :return: Path to the created file.
        """
        SWOMT = bpy.context.scene.SWOMT
        file = SWOMT.AssetPath
        print(f"file = {file}")
        merged_file = get_merged_mmb(file)
        print(f'merged file size = {merged_file.getbuffer().nbytes}')
        mod_file = os.path.splitext(file)[0] + "_MOD.mmb"
        print(f'mod file = {mod_file}')
        if os.path.exists(mod_file):
            return mod_file
        else:
            with open(mod_file, 'wb') as w:
                CopyFile(merged_file, w, 0, merged_file.getbuffer().nbytes)
        return mod_file
    @staticmethod
    def overwrite_vertex_positions(file, skeletal_mesh:SkeletalMeshAsset, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        obj = BME.find_object_by_name(mesh.name+f"_LOD{lod_index}")
        lod:SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            with open(file,'rb+') as f:
                f.seek(lod.data_offset)
                for v in range(lod.vertex_count):
                    lod.write_vertex_position(f, pos=data.vertices[v].co * Vector((-1.0,1.0,1.0)), scale=2)
    @staticmethod
    def get_vertex_blend_indices(vertex):
        '''
        Returns a dictionary of the vertex normalized, sorted and truncated 8 weights.
        '''

        group_weights = {}

        # Gather bone groups
        for vg in vertex.groups:
            if vg.weight > 0.0:
                group_weights[vg.group] = vg.weight

        # Normalize
        total_weight = 0
        for k in group_weights.keys():
            total_weight += group_weights[k]
        # print(total_weight)
        if total_weight == 0:
            raise Exception("Vertex {v} has no weight".format(v=vertex.index))
        normalizer = 1/total_weight
        for gw in group_weights:
            group_weights[gw] *= normalizer

        # Sort Weights
        sorted_weights = sorted(group_weights.items(), key=operator.itemgetter(1), reverse=True)

        #Truncate Weights
        trunc_weights = sorted_weights[0:8]
        # print(trunc_weights)
        return trunc_weights
    @staticmethod
    def convert_coordinate(co):
        return Vector((co[0] *-1,co[1],co[2]))
    @staticmethod
    def write_vertices(file, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        f = file
        obj = BME.find_object_by_name(mesh.name+f"_LOD{lod_index}")
        lod:SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            stride = mesh.vertex_stride
            pos_length = 0
            if mesh.position_type == 0:
                pos_length = 8
            elif mesh.position_type == 1:
                pos_length = 12
            else:
                pos_length = 12
            weight_count = int((stride - pos_length) / 2)
            print(stride, weight_count,mesh.position_type)
            mesh_bones = list(mesh.mesh_bones.keys())
            for v in bm.verts:
                # Write Coordinate
                if mesh.position_type == 0:
                    for co in BME.convert_coordinate(v.co):
                        f.write(bp.int16_norm(co))
                if mesh.position_type == 1:
                    for co in BME.convert_coordinate(v.co):
                        f.write(bp.float(co))
                vertex = data.vertices[v.index]

                weights = BME.get_vertex_blend_indices(vertex)
                bi = b''
                # Write weights
                int_weights = []
                for w in weights:
                    int_weights.append(max(0,min(int(w[1] * ((2 ** 8)-1)),255)))
                weight_sum = sum(int_weights)
                extra_weight = 0xFF - weight_sum
                for i in range(weight_count):
                    if i > len(weights)-1:
                        f.write(bp.uint8_norm(0))
                    else:
                        f.write(bp.uint8(int_weights[i] + extra_weight))
                        extra_weight = 0
                # Write indices
                for i in range(weight_count):
                    if i > len(weights)-1:
                        f.write(bp.uint8(0))
                    else:
                        print(weights[i], weights[i][0], mesh.real_bone_indices_to_mesh_bones)
                        mesh_bone_index = mesh.real_bone_indices_to_mesh_bones[weights[i][0]]
                        f.write(bp.uint8(mesh_bone_index))
            # f.write(b'\xFA\x7F\xFA\x7F')
    @staticmethod
    def write_normals(file, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        f = file
        obj = BME.find_object_by_name(mesh.name+f"_LOD{lod_index}")
        lod:SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            data.loops.data.calc_tangents()
            NTB = [((1.0,0.0,0.0),(0.0,1.0,0.0),1.0)] * len(data.vertices)
            uv_layers = bm.loops.layers.uv.values()
            uv_maps = []
            color_layers = bm.verts.layers.float_color.keys()
            for uvl in uv_layers:
                UVs = [(0.0,0.0)] * len(data.vertices)
                for bface in bm.faces:
                    for loop in bface.loops:
                        u = loop[uvl].uv[0]
                        v = 1 - loop[uvl].uv[1]
                        UVs[loop.vert.index] = [u, v]
                uv_maps.append(UVs)

            for l in data.loops:
                if l.bitangent_sign == -1:
                    flip = -1.0
                else:
                    flip = 1.0
                NTB[l.vertex_index] = (l.normal,l.tangent,flip)
            for v in data.vertices:
                # Write Normals
                normal = NTB[v.index][0] #TODO support other normal format
                f.write(bp.float(normal[0]*-1))
                f.write(bp.float(normal[1]))
                f.write(bp.float(normal[2]))
                tangent = NTB[v.index][1]
                f.write(bp.float(tangent[0]*-1))
                f.write(bp.float(tangent[1]))
                f.write(bp.float(tangent[2]))
                v_flip = NTB[v.index][2]
                f.write(bp.float(v_flip))

                # Write Vertex Color
                for cl in color_layers:
                    color_layer = bm.verts.layers.float_color[cl]
                    vertex_color = bm.verts[v.index][color_layer]
                    for c in vertex_color:
                        f.write(bp.uint8_norm(c))

                # Write UVs
                for uv_map in uv_maps:
                    for uv in uv_map[v.index]:
                        f.write(bp.int16_norm(uv))
            bm.free()
    @staticmethod
    def write_triangles(file, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            # Write Faces
            # for p in data.polygons:
            #     for v in p.vertices:
            #         f.write(bp.uint16(v))
            for p in data.polygons:
                f.write(bp.uint16(p.vertices[0]))
                f.write(bp.uint16(p.vertices[2]))
                f.write(bp.uint16(p.vertices[1]))
            # f.write(b'\xFA\x7F\xFA\x7F\xFA\x7F\xFA\x7F\xFA\x7F\xFA\x7F')
            bm.free()
    @staticmethod
    def copy_previous_mesh_data(source_path, file,mesh_index = 0,lod_index = 0):
        sorted_lods = asset.get_sorted_lods()
        with open(source_path, 'rb') as source:
            for lod in sorted_lods:
                print(lod.data_offset, lod.data_size, lod.vertex_count)
                if lod.index == lod_index and lod.parent_mesh.index == mesh_index:
                    return
                source.seek(lod.data_offset)
                file.write(source.read(lod.data_size))
    @staticmethod
    def create_mesh_file(mesh_index = 0, lod_index = -1):
        """
        Creates a BytesIO of the reverse sorted LODs of a mesh using the edited blender mesh of the given Lod Index.
        :return: Mesh class with LODs created with the new offset and size information.
        :return: BytesIO of all mesh LODs.
        """
        SWOMT = bpy.context.scene.SWOMT
        file = SWOMT.AssetPath
        mesh_file = io.BytesIO()
        offset_diff = 0
        # lod_index = -1
        mesh = asset.meshes[mesh_index]
        # modded_mesh:SkeletalMeshAsset.Mesh = asset.meshes[mesh_index]
        modded_mesh:SkeletalMeshAsset.Mesh = copy.deepcopy(asset.meshes[mesh_index])

        with open(file, 'rb') as source:
            for lod in reversed(asset.meshes[mesh_index].lods):
                current_modded_lod = modded_mesh.lods[lod.index]
                current_modded_lod.parent_mesh = modded_mesh
                new_vertex_data_offset_a = mesh_file.tell()
                if lod.index == lod_index:
                    print("Edited LOD")
                    obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
                    current_modded_lod.vertex_count = len(obj.data.vertices)
                    current_modded_lod.index_count = len(obj.data.polygons) * 3
                    print("New Vertex count: ", len(obj.data.vertices))
                    print("New indices count: ", len(obj.data.polygons) * 3)
                    # Vertices
                    current_modded_lod.vertex_data_offset_a = mesh_file.tell()
                    BME.write_vertices(mesh_file, mesh, lod_index)
                    mesh_file.write(lod.vertex_end_bytes)
                    # Normals
                    current_modded_lod.vertex_data_offset_b = mesh_file.tell()
                    BME.write_normals(mesh_file, mesh, lod_index)
                    mesh_file.write(lod.normals_end_bytes)
                    # Indices
                    current_modded_lod.face_block_offset = mesh_file.tell()
                    BME.write_triangles(mesh_file, mesh, lod_index)
                    mesh_file.write(lod.faces_end_bytes)

                else: #Unedited lod taken from source file.
                    source.seek(lod.data_offset)

                    current_modded_lod.vertex_data_offset_a = new_vertex_data_offset_a
                    offset_diff = new_vertex_data_offset_a - lod.vertex_data_offset_a
                    current_modded_lod.vertex_data_offset_b = lod.vertex_data_offset_b + offset_diff
                    current_modded_lod.face_block_offset = lod.face_block_offset + offset_diff

                    mesh_file.write(source.read(lod.data_size))

                current_modded_lod.data_size = mesh_file.tell() - new_vertex_data_offset_a
                print("\n", lod.index, lod.data_offset, lod.data_size, lod.vertex_count)
                print("Vertex_data_offset_a = ", new_vertex_data_offset_a, "   was  ", lod.vertex_data_offset_a,
                      "  diff = ", offset_diff)
                print("Mesh_File data start : ", new_vertex_data_offset_a, "Data Size : ", current_modded_lod.data_size)
        modded_mesh.mesh_file = mesh_file
        return modded_mesh

asset : SkeletalMeshAsset = None
BMI = BlenderMeshImporter
BME = BlenderMeshExporter

class SWOMTSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", subtype="FILE_PATH")

# OPERATORS #
class LoadMMB(bpy.types.Operator):
    """Reads data from base .mmb file"""
    bl_idname = "object.load_mmb"
    bl_label = "Load"

    def execute(self,context):
        SWOMT = context.scene.SWOMT
        with open(SWOMT.AssetPath, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = Path(SWOMT.AssetPath).stem
            global asset
            asset = sk_mesh

        return {'FINISHED'}

class ImportLOD(bpy.types.Operator):
    """Imports the given LOD"""
    bl_idname = 'object.import_lod'
    bl_label = 'Import'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return asset is not None

    def execute(self,context):
        sk_mesh = asset
        mesh = sk_mesh.meshes[self.mesh_index]
        lod = mesh.lods[self.lod_index]
        SWOMT = context.scene.SWOMT
        merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
        obj = BMI.import_mesh(merged_mmb,
                              skeletal_mesh=sk_mesh,
                              mesh=mesh,
                              lod_index=lod.index)
        armature = BMI.find_or_create_skeleton(sk_mesh)
        BMI.parent_obj_to_armature(obj,armature)
        BMI.rotate_model(obj,armature)
        return {'FINISHED'}

class ExportLOD(bpy.types.Operator):
    """Exports the given LOD"""
    bl_idname = 'object.export_lod'
    bl_label = 'Export'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return asset is not None

    def execute(self,context):
        # mod_file = BME.copy_mmb_file()
        # BME.overwrite_vertex_positions(file=mod_file,
        #                                skeletal_mesh=asset,
        #                                mesh=asset.meshes[self.mesh_index],
        #                                lod_index=self.lod_index)
        SWOMT = bpy.context.scene.SWOMT
        file = SWOMT.AssetPath
        mesh = asset.meshes[self.mesh_index]
        lod = mesh.lods[self.lod_index]
        obj = BME.find_object_by_name(mesh.name + f"_LOD{self.lod_index}")
        print(file)
        print(self.mesh_index, self.lod_index)
        print(mesh.lod_count_offset)
        f = io.BytesIO()
        # modded_mesh, mesh_file = BME.create_mesh_file(self.mesh_index, self.lod_index)
        modded_asset = SkeletalMeshAsset()

        # Create modded mesh files and mesh data
        for m in asset.meshes:
            if m.index == self.mesh_index:
                modded_mesh = BME.create_mesh_file(self.mesh_index,self.lod_index)
            else:
                modded_mesh = BME.create_mesh_file(m.index)
            modded_asset.meshes.append(modded_mesh)

        # Copy Header from source mmb
        with open(file, 'rb') as mmb:
            f.write(mmb.read(asset.get_mesh_data_start_offset()))

        # Write sorted mesh data
        new_header_size = -1
        for lod in modded_asset.get_sorted_lods():
            print(lod.parent_mesh.index, lod.parent_mesh.name,lod.index, lod.vertex_data_offset_a, lod.data_size, f.tell())
            if new_header_size == -1:
                if not lod.is_header_lod:
                    new_header_size = f.tell()
            mesh_file = lod.parent_mesh.mesh_file
            mesh_file.seek(lod.vertex_data_offset_a)
            lod.data_offset = f.tell()
            f.write(mesh_file.read(lod.data_size))

        modded_asset.size = new_header_size
        # Update header values
        modded_asset.write(f)

        with open(file, "wb") as outfile:
            outfile.write(f.getbuffer())

        with open(SWOMT.AssetPath, 'rb') as f_:
            asset.clear()
            asset.parse(f_)

        return {'FINISHED'}
        with open(file, 'rb') as mmb:
            f.write(mmb.read(asset.meshes[self.mesh_index].lod_count_offset))
            f.write(bp.uint8(1)) #LOD count
            f.write(bp.uint32(858993168))

            bm = bmesh.new()
            bm.from_mesh(obj.data)
            tris = len(obj.data.polygons)
            vertex_offset = 0
            vertex_size = (mesh.vertex_stride * len(obj.data.vertices)) + 4
            normals_offset = vertex_offset + vertex_size
            normal_size = (mesh.normals_stride * len(obj.data.vertices)) + 4
            face_offset = normals_offset + normal_size
            print(vertex_offset, vertex_size, normals_offset,normal_size,face_offset)

            f.write(bp.uint32(len(obj.data.vertices))) # Vertex count
            f.write(bp.uint32(tris*3)) # Indices count
            f.write(bp.uint32(int(face_offset / 2))) # Face_offset / 2
            f.write(bp.uint32(vertex_offset)) # Vertices Offset
            f.write(bp.uint32(normals_offset)) # Normals Offset
            f.write(bp.uint32(face_offset)) # Face Offset
            f.write(bp.uint32(0)) # data_offset
            f.write(bp.uint32(0))  #data_size
            f.write(bp.uint32(0)) # unknown
            f.write(mesh.end_bytes)
            f.write(asset.end_bytes)



        with open(file+"__","wb") as outfile:
            outfile.write(f.getbuffer())
            vertex_start = outfile.tell()
            BME.write_vertices(outfile,mesh,self.lod_index)
            normals_start = outfile.tell()
            BME.write_normals(outfile,mesh,self.lod_index)
            face_start = outfile.tell()
            BME.write_triangles(outfile,mesh,self.lod_index)
            print(vertex_start,normals_start,face_start)

        return {'FINISHED'}
# PANELS #
class SWOMTPanel(bpy.types.Panel):
    """Creates a Panel in the Scene Properties window"""
    bl_label = "Star Wars: Outlaws Mesh Tool"
    bl_idname = "OBJECT_PT_swomtpanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"


    def draw(self,context):
        SWOMT = context.scene.SWOMT

        layout = self.layout
        row = layout.row()
        row.prop(SWOMT, "AssetPath")
        layout.row().operator("object.load_mmb")

class MeshPanel(bpy.types.Panel):
    bl_label = "Mesh"
    bl_idname = "OBJECT_PT_meshpanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_swomtpanel"

    def draw(self,context):
        SWOMT = context.scene.SWOMT

        layout = self.layout
        row = layout.row()
        if asset:
            row.label(text=asset.name)
            for mi, m in enumerate(asset.meshes):
                mesh_row = layout.row()
                mesh_box = mesh_row.box()
                mesh_box.label(text = m.name, icon = "MESH_ICOSPHERE")
                for li,l in enumerate(m.lods):
                    row = mesh_box.row()
                    row.label(text = f"LOD{li} - {l.vertex_count}", icon = "CON_SIZELIKE")
                    lod_import_button = row.operator("object.import_lod")
                    lod_import_button.lod_index = li
                    lod_import_button.mesh_index = mi
                    lod_export_button = row.operator("object.export_lod")
                    lod_export_button.lod_index = li
                    lod_export_button.mesh_index = mi

classes=[SWOMTSettings,
         SWOMTPanel,
         MeshPanel,
         LoadMMB,
         ImportLOD,
         ExportLOD]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SWOMT = bpy.props.PointerProperty(type=SWOMTSettings)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
