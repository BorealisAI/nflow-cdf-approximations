# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from scipy.spatial import ConvexHull, Delaunay
import itertools 
from flows import Flow
import torch 
import math 
import numpy as np
from collections import defaultdict
import time
from typing import Union
from collections import Counter
import random 
from torch.nn import functional as F
from sortedcontainers import SortedSet
from functools import partial

class Estimator(abc.ABC):
    def __init__(self, convex_hull: ConvexHull, flow: Flow):
        self.convex_hull = convex_hull
        self.flow = flow 
        self.num_dimensions = self.convex_hull.points.shape[1]

    @abc.abstractmethod
    def estimate(self, num_points):
        pass

class Cube:
    def __init__(self, mins, maxs):
        all_vertices = []
        mins = torch.FloatTensor(mins).view(-1)
        maxs = torch.FloatTensor(maxs).view(-1)
        self.num_dimensions = maxs.shape[0]
        self.equal_dims = torch.where(maxs==mins)[0]
        self.unequal_dims = torch.where(maxs>mins)[0]
        self.rank = self.unequal_dims.shape[0]
        code2idx = {}
        for i in range(0,2**self.rank):
            vertices = []
            equal_idx = 0
            unequal_idx = 0
            for j in range(self.num_dimensions):
                if equal_idx<self.equal_dims.shape[0] and j==self.equal_dims[equal_idx]:
                    vertices.append(mins[j])
                    equal_idx += 1
                elif unequal_idx<self.unequal_dims.shape[0] and j==self.unequal_dims[unequal_idx]:    
                    if i&(1<<unequal_idx):
                        vertices.append(mins[j])
                    else:
                        vertices.append(maxs[j])
                    unequal_idx += 1
            all_vertices.append(torch.stack(vertices))
            code2idx[tuple((all_vertices[-1]==mins).cpu().numpy())] = len(all_vertices)-1
        all_vertices = torch.stack(all_vertices)
        self.mins = mins.view(1,-1)
        self.maxs = maxs.view(1,-1)
        self.volume = (self.maxs-self.mins)[:,self.unequal_dims].prod()
        self.points = all_vertices 
        self.surface_normals = []
        self.boundary_simplices = []
        e = torch.eye(self.num_dimensions)[self.unequal_dims,:]
        permutations_tensor = torch.LongTensor(list(itertools.permutations(range(self.rank-1))))
        num_simplices_per_boundary = math.factorial(self.rank-1)
        for i,idx in enumerate(self.unequal_dims):
            eye_idx = torch.cat([e[:i],e[i+1:]],dim=0)
            eye_idx[:,idx] = 1
            
            # simplices and surface_normals corresponding to max
            start = mins+0
            start[idx] = maxs[idx]
            end = maxs + 0
            
            points = start+(eye_idx[permutations_tensor]*(end-start).view(1,-1)).cumsum(dim=1)  
            start_idx = code2idx[tuple((start==mins).cpu().numpy())]
            self.boundary_simplices.extend([[start_idx]+[code2idx[tuple(j)] for j in i] for i in (points==mins[None,None]).cpu().numpy()])
            self.surface_normals.extend([e[i]*self.volume/(maxs[idx]-mins[idx])/num_simplices_per_boundary]*num_simplices_per_boundary)
            
            # simplices and surface_normals corresponding to min
            start[idx] = mins[idx]
            points[:,:,idx] = mins[idx]
            start_idx = code2idx[tuple((start==mins).cpu().numpy())]
            self.boundary_simplices.extend([[start_idx]+[code2idx[tuple(j)] for j in i] for i in (points==mins[None,None]).cpu().numpy()])
            self.surface_normals.extend([-e[i]*self.volume/(maxs[idx]-mins[idx])/num_simplices_per_boundary]*num_simplices_per_boundary)
        self.surface_normals = torch.stack(self.surface_normals)
        self.boundary_simplices = torch.stack(list(map(torch.LongTensor,self.boundary_simplices)))
        
        self.maxs[0,self.equal_dims] = self.mins[0,self.equal_dims]+1e-3
        self.volume = (self.maxs-self.mins).prod()
    
    def contains(self, points):
        r = (points-self.mins)/(self.maxs-self.mins)
        return (((r<=1)*(r>=0)).sum(dim=1)==self.num_dimensions)
    
    def sample(self, num_points):
        return self.mins + torch.rand(num_points,self.num_dimensions)*(self.maxs-self.mins)


class MCEstimator:
    def __init__(self, convex_hull: Union[ConvexHull, Cube], flow: Flow):
        self.convex_hull = convex_hull
        self.flow = flow 
        self.cube = str(type(convex_hull))==str(Cube) # To avoid problems with autoreload isinstance(convex_hull, Cube )
        if not(self.cube):
            self.num_dimensions = self.convex_hull.points.shape[1]
            self.points = torch.FloatTensor(convex_hull.points)
            self.delaunay = Delaunay(convex_hull.points[convex_hull.vertices])
    
    def estimate(self, num_points):
        t = time.time()
        samples = self.flow.sample(num_points)["x"].detach()
        if self.cube:
            ret = num_points, self.convex_hull.contains(samples.cpu()).float().mean().cpu()
        else:
            ret = num_points, (self.delaunay.find_simplex(samples.cpu().numpy() )>=0).mean() 
        self.time = time.time() - t
        return ret


class ImportanceSamplingEstimator:
    def __init__(self, convex_hull: Union[ConvexHull, Cube], flow: Flow):
        self.convex_hull = convex_hull
        self.flow = flow 
        self.num_dimensions = self.convex_hull.points.shape[1]
        self.cube = str(type(convex_hull))==str(Cube) # To avoid problems with autoreload isinstance(convex_hull, Cube )
        if self.cube:
            self.total_volume = self.convex_hull.volume
        else:
            self.points = convex_hull.points[convex_hull.vertices]
            self.delaunay = Delaunay(self.points)
            self.simplices = torch.FloatTensor(self.points[self.delaunay.simplices])
            self.volumes = torch.abs(torch.linalg.det(self.simplices[:,1:,:]-self.simplices[:,0:1,:])).numpy()*1/math.factorial(self.num_dimensions)
            self.probs = self.volumes/self.volumes.sum()
            self.total_volume = self.volumes.sum()
        
    def estimate(self, num_points):
        t = time.time()
        if self.cube:
            samples = self.convex_hull.sample(num_points)
            flow_out = self.flow.forward(samples)
            ret = num_points, (torch.exp(flow_out["logpx"]).mean().detach()*self.total_volume).cpu().numpy()
        else:
            random_simplices = np.random.choice(range(len(self.simplices)),size=num_points,p=self.probs)
            idx_vectors = -torch.log(torch.rand(num_points,self.num_dimensions+1)+1e-5)
            idx_vectors = idx_vectors/idx_vectors.sum(dim=1,keepdims=True)
            
            samples = (self.simplices[random_simplices]*idx_vectors[:,:,None]).sum(dim=1)
            flow_out = self.flow.forward(samples)
            ret = num_points, torch.exp(flow_out["logpx"]).mean().detach().cpu().numpy()*self.volumes.sum()
        self.time = time.time() - t
        return ret 
    

class AdaptiveBoundaryEstimator(Estimator):
    def __init__(self, convex_hull: Union[ConvexHull,Cube], flow:Flow):
        super().__init__(convex_hull, flow)

        self.points = torch.FloatTensor(convex_hull.points).cuda()
        
        # compute g-fields of vertices
        self.f_vector = torch.ones([1,self.num_dimensions]).to(self.flow.device)/self.num_dimensions
        if str(type(convex_hull))==str(Cube):
            self.boundary_simplices = convex_hull.boundary_simplices
            self.total_surface_normals = convex_hull.surface_normals.cuda()
            self.centroid_hull = self.points.mean(dim=0)
        else:
            self.boundary_simplices = torch.LongTensor(convex_hull.simplices)
            # compute surface normals
            self.points_simplices = self.points[self.boundary_simplices]
            mat = self.points_simplices-self.points_simplices[:,0:1,:]
            self.centroid_hull = centroid_hull = self.points.mean(dim=0,keepdim=True)
            mat[:,0] = self.points_simplices.mean(dim=1)-centroid_hull
            sign_ = torch.sign(torch.linalg.det(mat))
            mat = mat[:,1:]
            
            surface_normals = []
            for idx in range(self.num_dimensions):
                m = torch.cat([mat[:,:,:idx],mat[:,:,idx+1:]],dim=2)
                surface_normals.append((-1)**idx * torch.linalg.det(m))
            
            self.total_surface_normals = sign_[:,None] * torch.stack(surface_normals,dim=1)*1/math.factorial(self.num_dimensions-1)
        self.simplex_volumes = self.total_surface_normals.norm(dim=1)
        self.surface_normals = self.total_surface_normals/self.simplex_volumes.view(-1,1)
        
        # store points corresponding to each normal
        self.points_list = defaultdict(set)
        self.dotproducts_list = defaultdict(lambda : torch.FloatTensor([]).to(self.points))
        
        # create simplices
        self.simplicial_feats = {}
        self.edges = defaultdict(lambda : {"simplices":set(),"length":None,"error":{},"centroid_idx":None})
        self.pts2normals = defaultdict(lambda : {"normals":set()})
        self.next_simplex_idx = 0
        
        idx = 0  
        for simplex in self.boundary_simplices:
            self.simplicial_feats[self.next_simplex_idx] = {
                "idx":idx,
                "simplex":simplex,
                "edges": [],
                "volume": self.simplex_volumes[idx]#.cpu()
            }
            simplex = tuple(sorted(simplex.numpy()))
            for comb in itertools.combinations(simplex,2):
                self.edges[comb]["simplices"].add(self.next_simplex_idx)
                self.simplicial_feats[self.next_simplex_idx]["edges"].append(comb)
            for pt in simplex:
                self.pts2normals[pt]["normals"].add(idx)
            idx += 1
            self.next_simplex_idx += 1
       
        for pt in range(len(self.points)):
            self.pts2normals[pt]
     
        self.dotproducts = None
        self.probs = None
        self.uz = None
        self.add_dotproducts(list(self.pts2normals.keys()))
        self.volume = 0
        self.error = 0
        self.update_new_simplices(list(self.simplicial_feats.keys()))
        
        self.times = []
        
        self.total_points = self.points.shape[0]
        self.EF_idx = 0
        self.sorted_containers = None

    def add_dotproducts(self, point_idxs):
        all_point_idxs = []
        all_normal_idxs = []
        idxs_zero = []
        unq_pts = []
        for id_ in point_idxs:
            for i,normal in enumerate(sorted(self.pts2normals[id_]["normals"])):
                if i==0:
                    idxs_zero.append(len(all_point_idxs))
                    unq_pts.append(id_)
                all_point_idxs.append(id_)
                all_normal_idxs.append(normal)
                

        points = self.points[all_point_idxs]
        normals = self.surface_normals[all_normal_idxs]
        
        dots, probs, uz = self.flow.dotProduct(points, normals)
        
        X = torch.zeros(len(point_idxs),len(self.surface_normals)).cuda()
        Z = torch.zeros(len(point_idxs)).cuda()
        UZ = torch.zeros(len(point_idxs),self.num_dimensions).cuda()
        
        if self.dotproducts is None:
            self.dotproducts = X
            self.probs = Z
            self.uz = UZ
        else:
            self.dotproducts = torch.cat([self.dotproducts,X],dim=0)
            self.probs = torch.cat([self.probs, Z],dim=0)
            self.uz = torch.cat([self.uz, UZ],dim=0)
        
        self.dotproducts[all_point_idxs,all_normal_idxs] = dots 
        
        self.probs[unq_pts] = probs[idxs_zero]
        self.uz[unq_pts] = uz[idxs_zero]
        
    def get_dotproducts(self, simplices, normals):
        dots = []
        for i in range(self.num_dimensions):
            dots.append(self.dotproducts[simplices[:,i],normals][:,None])
        dots = torch.cat(dots,dim=1)
        return dots
    
    def volume_estimate(self):
        return sum(self.simplicial_feats[simplex_idx]["volume_element"] for simplex_idx in self.simplicial_feats)
    
    def update_new_simplices(self, simplex_idxs):
        with torch.no_grad():
            boundary_simplices = torch.stack([self.simplicial_feats[i]["simplex"] for i in simplex_idxs])
            
            centroid = self.points[boundary_simplices].mean(dim=1,keepdim=True)
            dots = self.get_dotproducts(boundary_simplices,[self.simplicial_feats[i]["idx"] for i in simplex_idxs])
            
            volumes = torch.stack([self.simplicial_feats[i]["volume"] for i in simplex_idxs]).view(-1,1)
            vol_element = volumes*dots

            dists = self.num_dimensions*(centroid-self.points[boundary_simplices]).norm(dim=2)**2  
            
            vol_element = vol_element.sum(dim=1)/self.num_dimensions
            
            error = volumes[:,0]*((torch.square(dots-dots.mean(dim=1,keepdim=True)).sum(dim=1))**0.5+1e-5)*dists.sum(dim=1)
            
            vol_element = vol_element.cpu()
            error = error.cpu()
            
        idx = 0
        for simplex_idx in simplex_idxs:
            simplex_info = self.simplicial_feats[simplex_idx]
            
            simplex_info["vol_element"] = (vol_element[idx])            
            simplex_info["error"] = error[idx]
            
            simplex_info["volume_element"] = vol_element[idx]
            simplex_info["centroid"] = centroid[idx,0]
            self.volume += simplex_info["volume_element"] # Update the estimate volume
            self.error += error[idx]
            idx += 1

    def stochastic_estimate(self,total_points):
        t = time.time()
        
        volumes = []
        num_points = int(total_points**0.5)
        num_simplices = num_points 
        num_points = 1
        num_simplices = total_points//num_points
        sample_simplices = torch.distributions.Categorical(self.simplex_volumes).sample((num_simplices,1)).detach().cpu().numpy().reshape(-1)
        from collections import Counter
        total_points = 0
        for idx,num_points in Counter(sample_simplices).items():
            
            simplex = self.boundary_simplices[idx]
            
            idx_vectors = -torch.log(torch.rand(num_points,self.num_dimensions)+1e-5)
            idx_vectors = idx_vectors/idx_vectors.sum(dim=1,keepdims=True)
            idx_vectors = idx_vectors.to(self.points)
            
            total_points += num_points 
            samples = (self.points[simplex][None,:,:]*idx_vectors[:,:,None]).sum(dim=1)
            
            dots, probs, uz = self.flow.dotProduct(samples,self.surface_normals[idx].reshape(1,-1)*torch.ones(samples.shape[0],self.num_dimensions).cuda())
            volumes.append(dots.sum())
        self.time = time.time()-t
        return sum(volumes)/total_points*self.simplex_volumes.sum()
    
    def split_simplices(self):
        times = [time.time()]
        
        def error_fn(e):
            if self.edges[e]["length"] is None:
                self.edges[e]["length"] = (self.points[e[0]]-self.points[e[1]]).norm()
            
            length = self.edges[e]["length"]
            
            simplices = sorted(self.edges[e]["simplices"])
            error_simplex = torch.stack([self.simplicial_feats[x]["error"] for x in simplices])
            max_error, max_idx = error_simplex.max(dim=0)
            
            return (max_error).item(), length.item()
        
        error_fns = [error_fn]
        if self.sorted_containers is None:
            for edge in self.edges:
                self.edges[edge]["errors"] = [tuple(map(lambda x: float(x),error_fn(edge))) for error_fn in error_fns]
            def fn(idx,x):
                return self.edges[x]["errors"][idx]
            funcs = [partial(fn,idx) for idx in range(len(error_fns))]
            self.sorted_containers = [SortedSet(self.edges,key=funcs[idx]) for idx in range(len(error_fns))]
        
        self.EF_idx = (self.EF_idx+1) % len(error_fns)
        error_fn = error_fns[self.EF_idx] 
        
        iter_ = reversed(self.sorted_containers[self.EF_idx])
        edges_to_split = [next(iter_)]
        
        times.append(time.time())
        highest_error = None
        split_edge = None
        simplex_idxs_to_update = set()
        edges_to_remove = set()
        edges_to_update = set()
        edges_to_add = set()
        start_points_idx = self.points.shape[0]
        simplices_split = 0
        for edge in edges_to_split:
            edge_info = self.edges[edge]
            
            edges_to_remove.add(edge)

            # record highest error
            if highest_error is None:
                highest_error = error_fns[0](edge)
                split_edge = edge
            
            new_point = (self.points[edge,:]).mean(dim=0,keepdim=True)
            self.points = torch.cat([self.points,new_point],dim=0)
            centroid_idx = self.points.shape[0] - 1 

            # delete edge from self.edges
            simplices_split += len(edge_info["simplices"])
            
            # split the simplices
            for simplex_idx in edge_info["simplices"]:
                simplex_feat = self.simplicial_feats[simplex_idx]

                # delete this simplex
                del self.simplicial_feats[simplex_idx]
                if "volume_element" in simplex_feat:
                    # Subtract the volume-element of this simplex to be split
                    self.volume -= simplex_feat["volume_element"]
                    self.error -= simplex_feat["error"]
                if simplex_idx in simplex_idxs_to_update:
                    simplex_idxs_to_update.remove(simplex_idx)
                

                # contains v1
                smplx0 = simplex_feat["simplex"]*(simplex_feat["simplex"]!=edge[0])+(simplex_feat["simplex"]-edge[0]+centroid_idx)*(simplex_feat["simplex"]==edge[0])
                # contains v0
                smplx1 = simplex_feat["simplex"]*(simplex_feat["simplex"]!=edge[1])+(simplex_feat["simplex"]-edge[1]+centroid_idx)*(simplex_feat["simplex"]==edge[1])

                self.simplicial_feats[self.next_simplex_idx] = {"idx":simplex_feat["idx"],
                    "simplex": smplx0,
                    "edges":[],
                    "volume":simplex_feat["volume"]/2,
                    }
                
                self.simplicial_feats[self.next_simplex_idx+1] = {"idx":simplex_feat["idx"],
                    "simplex": smplx1,
                    "edges":[],
                    "volume":simplex_feat["volume"]/2 
                    }

                self.pts2normals[centroid_idx]["normals"].add(simplex_feat["idx"])
                # remove this simplex from all edge references
                for other_edge in simplex_feat["edges"]:
                    contains_v0 = edge[0] in other_edge
                    contains_v1 = edge[1] in other_edge
                    if contains_v0 and contains_v1:
                        # do nothing
                        continue
                    elif contains_v0:
                        self.edges[other_edge]["simplices"].add(self.next_simplex_idx+1)
                        self.simplicial_feats[self.next_simplex_idx+1]["edges"].append(other_edge)
                    elif contains_v1:
                        self.edges[other_edge]["simplices"].add(self.next_simplex_idx)
                        self.simplicial_feats[self.next_simplex_idx]["edges"].append(other_edge)
                    else:
                        self.edges[other_edge]["simplices"].add(self.next_simplex_idx)
                        self.edges[other_edge]["simplices"].add(self.next_simplex_idx+1)
                        self.simplicial_feats[self.next_simplex_idx]["edges"].append(other_edge)
                        self.simplicial_feats[self.next_simplex_idx+1]["edges"].append(other_edge)

                    self.edges[other_edge]["simplices"].remove(simplex_idx)
                    
                    edges_to_update.add(other_edge)
                    if simplex_idx in self.edges[other_edge]["error"]:
                        del self.edges[other_edge]["error"][simplex_idx]

                # the below list of edges is ordered such that the id of simplex that contains edge[0] gets added to set of [edge[0],centroid_idx]
                for vertex_i in range(2):
                    for v in simplex_feat["simplex"]:
                        if v.item()==edge[vertex_i]:
                            continue
                        comb = tuple(sorted([v.item(),centroid_idx]))
                        edges_to_add.add(comb)
                        self.edges[comb]["simplices"].add(self.next_simplex_idx)
                        
                        self.simplicial_feats[self.next_simplex_idx]["edges"].append(comb)
                        simplex_idxs_to_update.add(self.next_simplex_idx)
                        
                    
                    self.next_simplex_idx += 1
            
        times.append(time.time())
        
        self.add_dotproducts(range(start_points_idx,len(self.points)))
        times.append(time.time())

        self.update_new_simplices(simplex_idxs_to_update)        
        times.append(time.time())
        
        for edge in edges_to_remove:
            for idx,container in enumerate(self.sorted_containers):
                container.discard(edge)
            del self.edges[edge]
        
        for edge in edges_to_update:
            if edge in self.edges and "errors" in self.edges[edge]:
                for idx,container in enumerate(self.sorted_containers):
                    container.remove(edge)
                    self.edges[edge]["errors"][idx] = error_fns[idx](edge)
                    container.add(edge)
            elif edge in self.edges:
                self.edges[edge]["errors"] = [error_fns[idx](edge) for idx in range(len(error_fns))]
                for idx,container in enumerate(self.sorted_containers):
                    container.add(edge)
                
        for edge in edges_to_add:
            self.edges[edge]["errors"] = [error_fn(edge) for error_fn in error_fns]
            for idx,container in enumerate(self.sorted_containers):
                container.add(edge)
        
        
        times.append(time.time())
        self.total_points = self.points.shape[0]
        
        self.times.append([i-j for i,j in zip(times[1:],times) ])
        return highest_error, split_edge, simplices_split

    
    def estimate(self, num_points, verbose=False):
        outs = [[self.total_points,abs(self.volume),0]]
        import time

        while self.total_points<num_points:
            highest_error, split_edge, simplices_split = self.split_simplices()
            outs.append([self.total_points,abs(self.volume),np.array(self.times).sum()])
            
            if verbose:
                T = "("+",".join(map(lambda x: f"{x:+0.2f}",self.times[-1]))+")"
                print(f"\r{self.total_points:^6d} {split_edge} {T} {self.error:^2.5f} {simplices_split:^6d} {len(self.edges):^6d} {self.volume:+.5f} {highest_error[0]:+.5f} {highest_error[1]:+.5f}",end="")                
            
        return outs,self.volume


    
