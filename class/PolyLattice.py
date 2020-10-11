import sys
import numpy as np
import reference as rf
import copy as cp
import collections as cln
import time

pVlist = []
nVlist = []

class PolyLattice:
    def __init__(self, hss, ref_vertex, vertex, actual_dim, M, b):
        self.lattice = hss
        self.ref_vertex = ref_vertex
        self.vertex = vertex
        self.actual_dim = actual_dim
        self.M = M
        self.b = b

    # linear Transformation
    def linearTrans(self, M, b):
        if M.shape[1] != self.M.shape[0]:
            print("dimension is inconsistant")
            sys.exit(1)
        self.M = np.dot(M, self.M)
        self.b = np.dot(M, self.b) + b

        # update dim
        if M.shape[0] < self.actual_dim:
            self.actual_dim = M.shape[0]

        return self

    def linearTrans_mnist(self, M, b):

        self.vertex = (np.dot(M, self.vertex.transpose()) + b).transpose()
        keys_list = list(self.ref_vertex.keys())
        for i in range(len(keys_list)):
            self.ref_vertex[keys_list[i]] = self.vertex[i,:]

        # update dim
        if M.shape[0] < self.actual_dim:
            self.actual_dim = M.shape[0]

        return self

    # intersection between polytope and hyperplane pl:[a0,a1,...an,d], a*x + d = 0, d= 0
    def intersectPlane(self, idx):
        # map the plane into lower dimension
        a = self.M[idx,:]
        d = self.b[idx,:]
        new_pl = np.concatenate((a, d))
        # insert ones to vertices matrix
        start_time = time.time()

        Varray = self.vertex
        Varray1 = np.insert(Varray, Varray.shape[1], 1, axis=1)
        Vsign = np.dot(Varray1, new_pl)

        end_time = time.time() - start_time
        # x1 = np.dot(self.M, Varray.transpose()) + self.b
        # x255 = x1[255,:]
        # x255p = len(x255[x255>=0])
        # ********* no intersection *******************#
        if np.all(Vsign<=0):
            return [], self, end_time
        elif np.all(Vsign>=0):
            return self, [] ,end_time

        # ********************************************#

        ref_keys = list(self.ref_vertex.keys())
        pVlist = [ref_keys[idx] for idx in range(Vsign.shape[0]) if Vsign[idx] >= 0]
        sign_dict = cln.OrderedDict((el, 1) for el in pVlist)
        nVlist = [ref_keys[idx] for idx in range(Vsign.shape[0]) if Vsign[idx] <= 0]
        sign_dict.update(cln.OrderedDict((el, -1) for el in nVlist))

        # ********** self.actual_dim==1 ***************#
        if self.actual_dim==1:

            p0 = self.ref_vertex[pVlist[0]]
            p1 = self.ref_vertex[nVlist[0]]
            vvector = p1 - p0
            interp = p0 - vvector * ((d + np.dot(p0, a.transpose())) / (np.dot(vvector, a.transpose())))
            self.vertex = np.concatenate((self.vertex, interp))

            ref0 = rf.Refer(-1)

            p_ref_vertex = cln.OrderedDict({pVlist[0]: p0, ref0: interp})
            p_hss = [cln.OrderedDict({pVlist[0]:[set(),set()], ref0: [set(),set()]})]
            p_lattice = PolyLattice(p_hss, p_ref_vertex, 1, cp.copy(self.M), cp.copy(self.b))

            n_ref_vertex = cln.OrderedDict({nVlist[0]: p1, ref0: interp})
            n_hss = [cln.OrderedDict({nVlist[0]:[set(),set()], ref0: [set(),set()]})]
            n_lattice = PolyLattice(n_hss, n_ref_vertex, 1, cp.copy(self.M), cp.copy(self.b))


        # ************* self.actual_dim>2 ***************#
        # consider the smaller set


        if len(pVlist) >= len(nVlist):
            lessVlist = nVlist
            sign_flag = 1
        else:
            lessVlist = pVlist
            sign_flag = -1

        f1_set = set() # store all the 1_face that intersects with plane
        ref_interp = cln.OrderedDict()
        for lessV in lessVlist:
            for f1 in self.lattice[0][lessV][1]:
                for f0 in self.lattice[1][f1][0]:
                    lessVn = f0
                    if sign_dict[lessVn] == sign_flag:
                        p0 = self.ref_vertex[lessV]
                        p1 = self.ref_vertex[lessVn]
                        vvector = p1 - p0
                        point = p0 - vvector * ((d + np.dot(p0, a)) / (np.dot(vvector, a)))
                        pointxx = np.dot(self.M, np.array([point]).transpose()) + self.b

                        f1_set.add(f1)
                        # input new vertex
                        ref_interp.update({f1:point})
                        point = np.array([point])
                        self.vertex = np.append(self.vertex, point, axis=0)

        # extract the intersected hss
        orig_hss = self.intersect_hss(1, self.actual_dim-1, f1_set)

        # assign different memory address to the hss
        new_hss = cp.deepcopy(orig_hss)

        # insert the new hss to the orginal lattice
        point_refs = self.insert_hss(orig_hss, new_hss, ref_interp)

        pVlist.extend(point_refs)
        nVlist.extend(point_refs)
        n_lattice = self.extract_polylattice(nVlist,f1_set)
        p_lattice = self.extract_polylattice(pVlist, f1_set)

        return p_lattice, n_lattice, end_time

    def intersectPlane_mnist(self, idx):

        a = np.zeros(self.vertex.shape[1])
        a[idx] = 1
        d = np.array([0])

        vertex = self.vertex

        if np.all(vertex[:,idx]>=0):
            return self, []
        elif np.all(vertex[:,idx]<=0):
            return [], self

        ref_keys = list(self.ref_vertex.keys())
        pVlist = [ref_keys[i] for i in range(vertex.shape[0]) if vertex[i,idx] >= 0]
        sign_dict = cln.OrderedDict((el, 1) for el in pVlist)
        nVlist = [ref_keys[i] for i in range(vertex.shape[0]) if vertex[i,idx] <= 0]
        sign_dict.update(cln.OrderedDict((el, -1) for el in nVlist))

        # ********** self.actual_dim==1 ***************#
        if self.actual_dim == 1:
            p0 = self.ref_vertex[pVlist[0]]
            p1 = self.ref_vertex[nVlist[0]]
            vvector = p1 - p0
            interp = p0 + vvector * np.abs((p0[idx]) / vvector[idx])
            self.vertex = np.concatenate((self.vertex, interp))

            ref0 = rf.Refer(-1)

            p_ref_vertex = cln.OrderedDict({pVlist[0]: p0, ref0: interp})
            p_hss = [cln.OrderedDict({pVlist[0]: [set(), set()], ref0: [set(), set()]})]
            p_lattice = PolyLattice(p_hss, p_ref_vertex, 1, cp.copy(self.M), cp.copy(self.b))

            n_ref_vertex = cln.OrderedDict({nVlist[0]: p1, ref0: interp})
            n_hss = [cln.OrderedDict({nVlist[0]: [set(), set()], ref0: [set(), set()]})]
            n_lattice = PolyLattice(n_hss, n_ref_vertex, 1, cp.copy(self.M), cp.copy(self.b))

            return p_lattice, n_lattice

        # ************* self.actual_dim>2 ***************#
        # consider the smaller set
        if len(pVlist) >= len(nVlist):
            lessVlist = nVlist
            sign_flag = 1
        else:
            lessVlist = pVlist
            sign_flag = -1

        f1_set = set()  # store all the 1_face that intersects with plane
        ref_interp = cln.OrderedDict()
        for lessV in lessVlist:
            for f1 in self.lattice[0][lessV][1]:
                for f0 in self.lattice[1][f1][0]:
                    lessVn = f0
                    if sign_dict[lessVn] == sign_flag:
                        p0 = self.ref_vertex[lessV]
                        p1 = self.ref_vertex[lessVn]
                        vvector = p1 - p0
                        point = p0 + vvector * np.abs((p0[idx]) / vvector[idx])
                        # point = p0 - vvector * ((d + np.dot(p0, a)) / (np.dot(vvector, a)))

                        f1_set.add(f1)
                        # input new vertex
                        ref_interp.update({f1: point})
                        point = np.array([point])
                        self.vertex = np.append(self.vertex, point, axis=0)

        # extract the intersected hss
        orig_hss = self.intersect_hss(1, self.actual_dim - 1, f1_set)

        # assign different memory address to the hss
        new_hss = cp.deepcopy(orig_hss)

        # insert the new hss to the orginal lattice
        point_refs = self.insert_hss(orig_hss, new_hss, ref_interp)

        pVlist.extend(point_refs)
        nVlist.extend(point_refs)
        n_lattice = self.extract_polylattice(nVlist, f1_set)
        p_lattice = self.extract_polylattice(pVlist, f1_set)

        return p_lattice, n_lattice

    # intersection between polytope and half space pl:[a0,a1,...an,d], a*x + d <= 0
    def intersectHalfspace(self, pl_a, pl_d):

        # map the plane into lower dimension
        a = np.dot(pl_a, self.M)
        d = np.dot(pl_a, self.b) + pl_d
        new_pl = np.concatenate((a, d))

        # insert ones to vertices matrix
        # Varray = np.array(self.vertex)
        Varray = self.vertex
        Varray1 = np.insert(Varray, Varray.shape[1], 1, axis=1)

        # signs indicate vertex's side of plane
        Vsign = np.dot(Varray1, new_pl)

        ref_keys = list(self.ref_vertex.keys())
        pVlist = [ref_keys[idx] for idx in range(Vsign.shape[0]) if Vsign[idx] >= 0]
        sign_dict = cln.OrderedDict((el, 1) for el in pVlist)
        nVlist = [ref_keys[idx] for idx in range(Vsign.shape[0]) if Vsign[idx] <= 0]
        sign_dict.update(cln.OrderedDict((el, -1) for el in nVlist))

        # ********* no intersection *******************#
        if not pVlist:
            return self
        elif not nVlist:
            return None

        # ********** self.actual_dim==1 ***************#
        if self.actual_dim==1:

            p0 = self.ref_vertex[pVlist[0]]
            p1 = self.ref_vertex[nVlist[0]]
            vvector = p1 - p0
            interp = p0 - vvector * ((d + np.dot(p0, a.transpose())) / (np.dot(vvector, a.transpose())))
            interp = np.array([interp])
            self.vertex = np.append(self.vertex, interp, axis=0)
            ref0 = rf.Refer(-1)

            n_ref_vertex = cln.OrderedDict({nVlist[0]: p1, ref0: interp})
            n_hss = [cln.OrderedDict({nVlist[0]:[set(),set()], ref0: [set(),set()]})]
            n_vertex = np.concatenate((p1, interp), axis=0 )
            n_lattice = PolyLattice(n_hss, n_ref_vertex, n_vertex, 1, cp.copy(self.M), cp.copy(self.b))

            return n_lattice

        # ************* self.actual_dim>2 ***************#
        # consider the smaller set
        if len(pVlist) >= len(nVlist):
            lessVlist = nVlist
            sign_flag = 1
        else:
            lessVlist = pVlist
            sign_flag = -1

        f1_set = set() # store all the 1_face that intersects with plane
        ref_interp = cln.OrderedDict()
        for lessV in lessVlist:
            for f1 in self.lattice[0][lessV][1]:
                for f0 in self.lattice[1][f1][0]:
                    lessVn = f0
                    if sign_dict[lessVn] == sign_flag:
                        p0 = self.ref_vertex[lessV]
                        p1 = self.ref_vertex[lessVn]
                        vvector = p1 - p0
                        point = p0 - vvector * ((d + np.dot(p0, a)) / (np.dot(vvector, a)))

                        f1_set.add(f1)
                        # input new vertex
                        ref_interp.update({f1:point})
                        point = np.array([point])

                        self.vertex = np.append(self.vertex, point, axis=0)


        # extract the intersected hss
        orig_hss = self.intersect_hss(1, self.actual_dim-1, f1_set)
        # self.test_face_num(self.lattice)
        # self.test_face_num(orig_hss)

        # assign different memory address to the hss
        new_hss = cp.deepcopy(orig_hss)

        # insert the new hss to the orginal lattice
        point_refs = self.insert_hss(orig_hss, new_hss, ref_interp)

        # pVlist.extend(point_refs)
        nVlist.extend(point_refs)

        n_lattice = self.extract_polylattice(nVlist,f1_set)

        return n_lattice


    # insert hss to lattice class
    def insert_hss(self, orig_hss, inter_hss, ref_interp):
        for m in range(self.actual_dim):
            if m == self.actual_dim-1:
                self.lattice[m].update(inter_hss[m])
                continue

            orig_keys = list(orig_hss[m].keys())
            inter_keys = list(inter_hss[m].keys())
            for i in range(len(orig_keys)):
                orig_k = orig_keys[i]
                inter_k = inter_keys[i]
                self.lattice[m+1][orig_k][0].add(inter_k)
                inter_hss[m][inter_k][1].add(orig_k)

            self.lattice[m].update(inter_hss[m])

        # update and merge ref_v
        orig_keys = list(orig_hss[0].keys())
        inter_keys = list(inter_hss[0].keys())
        for i in range(len(orig_keys)):
            orig_k = orig_keys[i]
            inter_k = inter_keys[i]
            ref_interp.update({inter_k: ref_interp[orig_k]})
            ref_interp.pop(orig_k, None)
        self.ref_vertex.update(ref_interp)

        # return the references of the new generated points
        return list(ref_interp.keys())

    # extract lattice from a vertex set
    def extract_polylattice(self, vlist, f1_set):
        new_ref_vertex = cln.OrderedDict((ref, np.copy(self.ref_vertex[ref])) for ref in vlist)
        hss = self.intersect_hss(0, self.actual_dim-1, vlist)
        vertex = np.array(list(new_ref_vertex.values()))
        aLattice = PolyLattice(hss, new_ref_vertex, vertex, self.actual_dim, np.copy(self.M), np.copy(self.b))

        return aLattice

    # extract lattice from a set
    def intersect_hss(self, m, n, aset):
        set0 = set()
        set1 = cp.copy(aset)
        set2 = set()
        hss = []
        m_face = cln.OrderedDict()

        for i in range(m,n+1):
            for ref in set1:
                val_temp = cp.copy(self.lattice[i][ref])
                set2.update(val_temp[1])
                m_face.update({ref: [cp.copy(self.lattice[i][ref][0]), cp.copy(self.lattice[i][ref][1])]})
                val_temp2 = cp.copy(val_temp[0])
                for r in val_temp2:
                    if r not in set0:
                        m_face[ref][0].remove(r)

            hss.append(m_face)
            m_face = cln.OrderedDict()
            set0 = cp.copy(set1)
            set1 = cp.copy(set2)
            set2 = set()

        # m == 1 when to find the intersected hss
        if m == 1 and n >= 1:
            new_ref = rf.Refer(-1)
            new_facet = cln.OrderedDict({new_ref: [set(), set()]})
            for ref in hss[n-1].keys():
                hss[n-1][ref][1]= set([new_ref])
                new_facet[new_ref][0].add(ref)

            hss.append(new_facet)

        return hss

    # get vertex belonging to m-face
    # def face_vertex(self, m, aset):
    #     if m == 0:
    #         vs = []
    #         for s in aset:
    #             vs.append(self.ref_vertex[s])
    #         return vs
    #     else:
    #         new_set =set()
    #         for s in aset:
    #             new_set.update(self.lattice[m][s][0])
    #
    #     V = []
    #     V.extend(self.face_vertex(m-1, new_set))
    #     return V

    # set some dim to zero for Relu function
    def map_negative_poly(self, n):
        if self.actual_dim == 0:
            return self

        self.M[n, :] = 0
        self.b[n, :] = 0
        return self


    def map_negative_poly_mnist(self, n, flag):
        if self.actual_dim == 0:
            return self

        self.vertex[:,n] = 0

        for key in self.ref_vertex.keys():
            self.ref_vertex[key][n] = 0

        return self

    # # remove non-vertice points
    # def memory_reduce(self):
    #
    #     u, s, vh = np.linalg.svd(self.M)
    #
    #     rk = len(list(s[s > 1e-10]))
    #     if rk >= self.actual_dim:
    #         return
    #     else:
    #         self.actual_dim = rk
    #         if self.actual_dim == 0:
    #             return
    #
    #     # u_new = u[:, :rk]
    #     # s_new = np.diag(s[:rk])
    #     vh_new = vh[:rk, :]
    #     # self.M = np.dot(u_new, s_new)
    #
    #     old_vertex = self.vertex.transpose()
    #     new_vertex = np.dot(vh_new, old_vertex).transpose()
    #     new_vertex_center = np.array([new_vertex.mean(axis=0)])
    #     new_vertex = new_vertex - new_vertex_center
    #
    #     vertex_temp = []
    #     for i in range(new_vertex.shape[0]):
    #         vertex_temp.append(tuple(new_vertex[i, :].tolist()))
    #
    #     old_new_dict = cln.OrderedDict(zip(self.vertex, vertex_temp))
    #
    #     # self.b = np.dot(self.M, new_vertex_center.transpose()) + self.b
    #
    #     # ref_vertex_keys = list(self.ref_vertex.keys())
    #     # self.vertex = []
    #     # for i in range(new_vertex.shape[0]):
    #     #     vertex_temp = tuple(new_vertex[i, :].tolist())
    #     #     self.vertex.append(vertex_temp)
    #     #     self.ref_vertex[ref_vertex_keys[i]] = vertex_temp
    #
    #     new_facet_list = []
    #     m = self.actual_dim - 1
    #     for f in self.lattice[m].keys():
    #         f_vertex = self.face_vertex(m, {f})
    #         f_vertex = [old_new_dict[e] for e in f_vertex]
    #         f_vertex_array = np.array(f_vertex).transpose()
    #
    #         pinv_vertex = np.linalg.pinv(f_vertex_array)
    #         ones_vector = np.ones((1, len(f_vertex)))
    #         con_matrix = np.dot(ones_vector, pinv_vertex)
    #
    #         sign_vertex = np.dot(con_matrix, new_vertex.transpose()) - np.ones((1, len(self.vertex)))
    #         if (sign_vertex >= -1e-10).all() or (sign_vertex <= 1e-10).all():
    #             new_facet_list.append(f)
    #
    #     # update lattice
    #     self.extract_hss_top(m, new_facet_list)

    # # extract lattice from m_face sets
    # def extract_hss_top(self, m, keys):
    #     hss = []
    #     ref_last = set()
    #     ref_current = cp.copy(keys)
    #     ref_future = set()
    #     m_face = cln.OrderedDict()
    #     for i in range(m,-1,-1):
    #         for r in ref_current:
    #             m_face.update({r: [cp.copy(self.lattice[i][r][0]), cp.copy(self.lattice[i][r][1])]})
    #             ref_future.update(cp.copy(self.lattice[i][r][0]))
    #
    #         for ref in m_face.keys():
    #             set_temp = cp.copy(m_face[ref][1])
    #             for r in set_temp:
    #                 if r not in ref_last:
    #                     m_face[ref][1].remove(r)
    #
    #         hss.append(m_face)
    #         m_face = cln.OrderedDict()
    #         ref_last = cp.copy(ref_current)
    #         ref_current = cp.copy(ref_future)
    #         ref_future = set()
    #
    #     self.lattice = list(reversed(hss))
    #     self.ref_vertex = cln.OrderedDict((ref, self.ref_vertex[ref]) for ref in self.lattice[0].keys())
    #     self.vertex = np.array([self.ref_vertex[ref] for ref in self.lattice[0].keys()])
    #     return self

    #
    # def test_face_num(self, hss):
    #     for m in range(self.actual_dim - 1):
    #         if m + 1 > len(hss):
    #             break
    #         face_m0 = hss[m]
    #         face_m1 = hss[m + 1]
    #         f1_f0 = set()
    #         for k in hss[m + 1].keys():
    #             f1_f0.update(hss[m + 1][k][0])
    #
    #         f0_f1 = set()
    #         for k in hss[m].keys():
    #             f0_f1.update(hss[m][k][1])
    #
    #         if len(f1_f0) != len(face_m0):
    #             print('f1_f0', m)
    #         if len(f0_f1) != len(face_m1):
    #             print('f0_f1', m)


