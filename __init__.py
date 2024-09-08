import bpy
import bmesh
from struct import unpack, pack
import numpy as np
import math
from mathutils import Matrix, Euler, Vector
from pathlib import Path

file_path = "W:\OutlawsModding\helix\\baked\\art\characters\characters\chr_body\chr_body_1_gold_kay\chr_body_1_gold_kay-combined.mmb_0"
file_path = Path(file_path)

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
    def uint8Norm(v):
        if 0.0 <= v <= 1.0:
            i = int(v * ((2 ** 8)-1))
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
    def int16Norm(v):
        if -1.0 < v < 1.0:
            if v >= 0:
                v = int(abs(v) * (2 ** 15))
            else:
                v = 2 ** 16 - int(abs(v) * (2 ** 15))
        else:
            raise Exception("Couldn't normalize value as int16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<H', v)
    @staticmethod
    def uint16Norm(v):
        if 0.0 < v < 1.0:
            i = v * (2 ** 16) - 1
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

class SkeletalMeshAsset(Asset):
    class Mesh:
        class LOD:
            def __init__(self):
                self.vertex_count = 0
                self.index_count = 0
                self.vertex_data_offset_a = 0
                self.vertex_data_offset_b = 0
                self.face_block_offset = 0
                self.data_offset = 0
                self.data_size = 0
            def parse(self, f):
                self.vertex_count = br.uint32(f)
                self.index_count = br.uint32(f)
                unknown_size = br.uint32(f)
                self.vertex_data_offset_a = br.uint32(f)
                self.vertex_data_offset_b = br.uint32(f)
                self.face_block_offset = br.uint32(f)
                self.data_offset = br.uint32(f)
                self.data_size = br.uint32(f)
                lod_screen_size = br.float(f)  # not confirmed screen size

            def get_vertex_positions(self,raw_mesh_file):
                vertices = []
                with open(raw_mesh_file,'rb') as f:
                    f.seek(self.vertex_data_offset_a)
                    for v in range(self.vertex_count):
                        x = br.int16_norm(f)
                        y = br.int16_norm(f)
                        z = br.int16_norm(f)
                        w = br.int16(f)
                        pos = (x*w,y*w,z*w)
                        f.seek(16,1) #TODO get real stride from Mesh parent class
                        vertices.append(pos)
                return vertices

            def get_bone_weights(self, raw_mesh_file):
                bone_weights = []
                with open(raw_mesh_file, 'rb') as f:
                    f.seek(self.vertex_data_offset_a)
                    for v in range(self.vertex_count):
                        f.seek(8, 1)  # skip positions
                        weight_count = 8  # TODO get real index count from stride
                        weights = []
                        iw = {}
                        for w in range(weight_count):
                            weight = br.uint8_norm(f)
                            if weight > 0.0:
                                weights.append(weight)

                        for i in range(weight_count):
                            if i < len(weights):
                                iw[br.uint8(f)] = weights[i]
                            else:
                                f.seek(1,1)

                        bone_weights.append(iw)
                return bone_weights

            def get_triangles(self,raw_mesh_file):
                """
                Seeks to Lod.face_block_offset and reads all triangle indices.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 3 vertex indices to form a triangle.
                """
                tris = []
                with open(raw_mesh_file, 'rb') as f:
                    f.seek(self.face_block_offset)
                    for i in range(int(self.index_count/3)):
                        f1 = br.uint16(f)
                        f2 = br.uint16(f)
                        f3 = br.uint16(f)
                        tris.append((f1,f2,f3))
                return tris

            def get_normals(self,raw_mesh_file):
                normals = []
                with open(raw_mesh_file, 'rb') as f:
                    f.seek(self.vertex_data_offset_b)
                    for i in range(self.vertex_count):
                        x = br.int8_norm(f) *-1
                        y = br.int8_norm(f)
                        z = br.int8_norm(f)
                        w = br.int8(f)
                        v = Vector((x*w, y*w, z*w)).normalized()
                        v.negate() #TODO not sure about this
                        f.seek(12,1)
                        normals.append(v)
                return normals

            def get_uvs(self,raw_mesh_file):
                """
                Seeks to Lod.vertex_data_offset_b and reads all UV data.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 2 floats as UV coordinates.
                """
                uvs = []
                with open(raw_mesh_file, 'rb') as f:
                    f.seek(self.vertex_data_offset_b)
                    for i in range(self.vertex_count):
                        f.seek(12,1) #skip normals and color #TODO find real value
                        u = br.int16_norm(f)
                        v = br.int16_norm(f)
                        uvs.append((u,v))
                return uvs

            def get_color(self,raw_mesh_file):
                """
               Seeks to Lod.vertex_data_offset_b and reads all Color data.
               :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
               :return: a List of Tuples containing 4 floats as RGBA coordinates.
               """
                colors = []
                with open(raw_mesh_file, 'rb') as f:
                    f.seek(self.vertex_data_offset_b)
                    for i in range(self.vertex_count):
                        f.seek(8, 1)  # skip normals #TODO find real value
                        r = br.uint8_norm(f)
                        g = br.uint8_norm(f)
                        b = br.uint8_norm(f)
                        a = br.uint8_norm(f)
                        f.seek(4,1) # skip uvs
                        colors.append((r,g,b,a))
                return colors

        def __init__(self):
            self.name = ""
            self.lod_count = 0
            self.lods = []
            self.vertex_stride = 0
            self.normals_stride = 0
            self.mesh_bones = {}
        def parse(self, f):
            self.name = br.name(f)
            f.seek(48, 1)  # some kind of matrix
            f.seek(1, 1)
            x_count = br.uint16(f)
            f.seek(1, 1)
            f.seek(4 * x_count, 1)
            f.seek(1, 1)
            u_count = br.uint16(f)
            for b in range(u_count):
                matrix = br.matrix_4x4(f)
                bone_index = br.uint16(f)
                self.mesh_bones[bone_index] = matrix
            f.seek(2, 1)
            lod_info_type = br.uint16(f)
            self.lod_count = br.uint8(f)
            f.seek(4, 1)
            for l in range(self.lod_count):
                lod = self.LOD()
                lod.parse(f)
                if lod_info_type == 2:
                    f.seek(28,
                           1)  # if lod_info_type = 2 there's more data, this should be handled by the LOD.parse function.
                self.lods.append(lod)
            f.seek(19, 1)  # this is specific to chr_body_1_gold_kay.mmb_0 for now #TODO figure out between lods and strides
            self.vertex_stride = br.uint16(f)
            self.normals_stride = br.uint16(f)
            f.seek(20, 1)  # TODO figure out between strides and next mesh
        def extract_mesh_file(self,f):
            """
            Creates a file gathering the raw data of all Lods the Mesh.
            This
            :param f: combined mmb file that contains the header and data.
            :return: Path to the extracted raw_mesh file.
            """
            extract_file = Path.joinpath(file_path.parent, file_path.stem + "_-_" + self.name + ".raw_mesh")
            with open(extract_file, "wb+") as w:
                print(f"created {extract_file}")
                for lod in reversed(self.lods):
                    print(f'Copy at {lod.data_offset} size {lod.data_size}')
                    CopyFile(f, w, lod.data_offset, lod.data_size)
                print(f'Total size: {sum(lod.data_size for lod in self.lods)}')
            return extract_file

    class Bone:
        def __init__(self,f):
            self.name = br.name(f)
            self.matrix = br.matrix_4x4(f)
            self.parent_index = br.uint16(f)

    def __init__(self):
        super().__init__()
        self.bone_count = 0
        self.bones = []
        self.mesh_count = 0
        self.meshes = []
    def parse(self,f):
        super().parse(f)
        self.bone_count = br.uint32(f)
        for b in range(self.bone_count):
            self.bones.append(self.Bone(f))
        self.mesh_count = br.uint32(f)
        for m in range(self.mesh_count):
            mesh = self.Mesh()
            mesh.parse(f)
            self.meshes.append(mesh)

class BlenderMeshImporter:
    @staticmethod
    def import_mesh(file, skeletal_mesh:SkeletalMeshAsset, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        # Extract raw mesh file
        raw_mesh_file = mesh.extract_mesh_file(file)

        # Create Mesh and Object
        obj_data = bpy.data.meshes.new(mesh.name)
        obj = bpy.data.objects.new(mesh.name, obj_data)
        new_collection = bpy.data.collections.new(skeletal_mesh.bones[0].name) #TODO search for existing collection first
        bpy.context.scene.collection.children.link(new_collection)
        new_collection.objects.link(obj)

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
            bm_face = bm.faces.new(face_vertices)
            bm_face.normal_flip() #this is required because the *-1 on x vertex co flips the mesh normals
        bm.to_mesh(obj_data)
        bm.free()
        bm = bmesh.new()
        bm.from_mesh(obj_data)
        bm.faces.ensure_lookup_table()
        # Import UVs
        uvs = lod.get_uvs(raw_mesh_file)
        uv_layer = bm.loops.layers.uv.new("UVMap")
        for finder, face in enumerate(bm.faces):
            for lindex, loop in enumerate(face.loops):
                v_index = loop.vert.index
                v_uv = (uvs[v_index][0],uvs[v_index][1]*-1+1)
                loop[uv_layer].uv = v_uv

        # Import Colors
        colors = lod.get_color(raw_mesh_file)
        color_layer = bm.verts.layers.float_color.new("Color")
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
        for bone in sk_mesh.bones:
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
    def import_skeleton(skeletal_mesh:SkeletalMeshAsset):
        _armature = bpy.data.armatures.new("armature_data")
        _obj = bpy.data.objects.new("Armature", _armature)
        bpy.context.scene.collection.objects.link(_obj) #TODO find existing collection
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

#testing
with open(file_path, 'rb') as file:
    print(f"opened : {file_path}")
    sk_mesh = SkeletalMeshAsset()
    sk_mesh.parse(file)
    BMI = BlenderMeshImporter
    obj = BMI.import_mesh(file, sk_mesh, sk_mesh.meshes[0], 0)
    armature = BMI.import_skeleton(sk_mesh)
    BMI.parent_obj_to_armature(obj,armature)
    BMI.rotate_model(obj,armature)