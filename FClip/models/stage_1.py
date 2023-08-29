from collections import OrderedDict, defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import torch

from FClip.line_parsing import OneStageLineParsing
from FClip.config import M
from FClip.losses import ce_loss, sigmoid_l1_loss, focal_loss, l12loss
from FClip.nms import structure_nms_torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from collections import defaultdict
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
class FClip(nn.Module):
    def __init__(self, backbone):
        super(FClip, self).__init__()
        self.backbone = backbone
        self.M_dic = M.to_dict()
        self._get_head_size()
        self.conv1 = GCNConv(32, 32)

        self.Linear1 = nn.Linear(128, 1)
        self.Linear2 = nn.Linear(128, 16)
        self.Linear3 = nn.Linear(4, 16)
        self.Linear4 = nn.Linear(32, 32)
        self.Linear5 = nn.Linear(160,128)
        self.conv1d1 = nn.Conv1d(256, 1000, 1)
        self.conv1d2 = nn.Conv1d(1000, 128, 1)
        self.conv1d3 = nn.Conv1d(1000, 256, 1)
        self.torchcat = torch.cat
        self.frelu = F.relu
        self.torch_from_numpy = torch.from_numpy
        self.torchIntTensor = torch.IntTensor
    
    def _get_head_size(self):

        head_size = []
        for h in self.M_dic['head']['order']:
            head_size.append([self.M_dic['head'][h]['head_size']])

        self.head_off = np.cumsum([sum(h) for h in head_size])

    def to_int(self,x):
        return tuple(map(int, x))
    def lcmap_head(self, output, target):
        name = "lcmap"

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx-1]

        pred = output[s: self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)

        if self.M_dic['head'][name]['loss'] == "Focal_loss":
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = focal_loss(pred, target, alpha)
        elif self.M_dic['head'][name]['loss'] == "CE":
            loss = ce_loss(pred, target, None)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).softmax(1)[:, 1], loss * weight

    def lcoff_head(self, output, target, mask):
        name = 'lcoff'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(self.M_dic['head'][name]['head_size'], batch, row, col)

        loss = sum(
            sigmoid_l1_loss(pred[j], target[j], offset=-0.5, mask=mask)
            for j in range(2)
        )

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(1, 0, 2, 3).sigmoid() - 0.5, loss * weight

    def lleng_head(self, output, target, mask):
        name = 'lleng'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        if self.M_dic['head'][name]['loss'] == "sigmoid_L1":
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == "L1":
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0., 1.)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def angle_head(self, output, target, mask):
        name = 'angle'

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        if self.M_dic['head'][name]['loss'] == "sigmoid_L1":
            loss = sigmoid_l1_loss(pred, target, mask=mask)
            pred = pred.sigmoid()
        elif self.M_dic['head'][name]['loss'] == "L1":
            loss = l12loss(pred, target, mask=mask)
            pred = pred.clamp(0., 1.)
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred, loss * weight

    def jmap_head(self, output, target, n_jtyp):
        name = "jmap"
        _, batch, row, col = output.shape

        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s: self.head_off[offidx]].reshape(n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)

        if self.M_dic['head'][name]['loss'] == "Focal_loss":
            alpha = self.M_dic['head'][name]['focal_alpha']
            loss = sum(
                    focal_loss(pred[i], target[i], alpha) for i in range(n_jtyp)
                )
        elif self.M_dic['head'][name]['loss'] == "CE":
            loss = sum(
                    ce_loss(pred[i], target[i], None) for i in range(n_jtyp)
                )
        else:
            raise NotImplementedError

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1], loss * weight

    def joff_head(self, output, target, n_jtyp, mask):
        name = "joff"

        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]

        pred = output[s: self.head_off[offidx]].reshape(
            n_jtyp, self.M_dic['head'][name]['head_size'], batch, row, col)

        loss = sum(
                    sigmoid_l1_loss(pred[i, j], target[i, j], scale=1.0, offset=-0.5, mask=mask[i])
                    for i in range(n_jtyp)
                    for j in range(2)
                )
        weight = self.M_dic['head'][name]['loss_weight']
        return pred.permute(2, 0, 1, 3, 4).sigmoid() - 0.5, loss * weight

    def lmap_head(self, output, target):
        name = "lmap"
        _, batch, row, col = output.shape
        order = self.M_dic['head']['order']
        offidx = order.index(name)
        s = 0 if offidx == 0 else self.head_off[offidx - 1]
        pred = output[s: self.head_off[offidx]].reshape(batch, row, col)

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                .mean(2)
                .mean(1)
        )

        weight = self.M_dic['head'][name]['loss_weight']
        return pred.sigmoid(), loss * weight

    def forward(self, input_dict, isTest=False):

        if isTest:
            return self.test_forward(input_dict)
        else:
            return self.trainval_forward(input_dict)

    def test_forward(self, input_dict):

        extra_info = {
            'time_front': 0.0,
            'time_stack0': 0.0,
            'time_stack1': 0.0,
            'time_backbone': 0.0,
        }

        extra_info['time_backbone'] = time.time()
        image = input_dict["image"]
        outputs, feature, backbone_time = self.backbone(image)
        extra_info['time_front'] = backbone_time['time_front']
        extra_info['time_stack0'] = backbone_time['time_stack0']
        extra_info['time_stack1'] = backbone_time['time_stack1']
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        output = outputs[0]

        heatmap = {}
        heatmap["lcmap"] = output[:, 0:                self.head_off[0]].softmax(1)[:, 1]
        heatmap["lcoff"] = output[:, self.head_off[0]: self.head_off[1]].sigmoid() - 0.5
        heatmap["lleng"] = output[:, self.head_off[1]: self.head_off[2]].sigmoid()
        heatmap["angle"] = output[:, self.head_off[2]: self.head_off[3]].sigmoid()

        parsing = True
        if parsing:
            lines, scores = [], []
            for k in range(output.shape[0]):
                line, score = OneStageLineParsing.fclip_torch(
                    lcmap=heatmap["lcmap"][k],
                    lcoff=heatmap["lcoff"][k],
                    lleng=heatmap["lleng"][k],
                    angle=heatmap["angle"][k],
                    delta=M.delta,
                    resolution=M.resolution
                )
                if M.s_nms > 0:
                    line, score = structure_nms_torch(line, score, M.s_nms)
                lines.append(line[None])
                scores.append(score[None])

            heatmap["lines"] = torch.cat(lines)
            heatmap["score"] = torch.cat(scores)
        return {'heatmaps': heatmap, 'extra_info': extra_info}

    def trainval_forward(self, input_dict):

        image = input_dict["image"]
        outputs, feature, backbone_time = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict["target"].copy()
        n_jtyp = 1
        T["lcoff"] = T["lcoff"].permute(1, 0, 2, 3)

        losses = []
        accuracy = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()

            L = OrderedDict()
            Acc = OrderedDict()
            heatmap = {}
            lcmap, L["lcmap"] = self.lcmap_head(output, T["lcmap"])
            lcoff, L["lcoff"] = self.lcoff_head(output, T["lcoff"], mask=T["lcmap"])
            # heatmap["lcmap"] = lcmap
            # heatmap["lcoff"] = lcoff

            lleng, L["lleng"] = self.lleng_head(output, T["lleng"], mask=T["lcmap"])
            angle, L["angle"] = self.angle_head(output, T["angle"], mask=T["lcmap"])
            # heatmap["lleng"] = lleng
            # heatmap["angle"] = angle


            #MY PENTA
            t = time.time()
            lines_for_train, scores = [], []

            for k in range(output.shape[1]):
                lines_for_train, score = OneStageLineParsing.fclip_torch(
                    lcmap=lcmap[k],
                    lcoff=lcoff[k],
                    lleng=lleng[k],
                    angle=angle[k],
                    delta=M.delta,
                    resolution=M.resolution
                )
                if M.s_nms > 0:
                    lines_for_train, score = structure_nms_torch(lines_for_train, score, M.s_nms)

                lines_for_train = lines_for_train.cpu()
            
                lines_graph_v1 = []
                lines_graph_v2 = []
                vertices_hash = defaultdict(list)

                # Making graph from lines
                for idx, line in enumerate(lines_for_train):
                    v1 = str(line[0])[7:-29]
                    v2 = str(line[1])[7:-29]

                    vertices_hash[v1].append(idx)
                    vertices_hash[v2].append(idx)

                    v1_lines = vertices_hash[v1]
                    v2_lines = vertices_hash[v2]

                    if (len(v1_lines) > 1):
                        for line in v1_lines[:-1]:
                            lines_graph_v1.append(line)
                            lines_graph_v2.append(idx)

                    if (len(v2_lines) > 1):
                        for line in v2_lines[:-1]:
                            lines_graph_v1.append(line)
                            lines_graph_v2.append(idx)

                edge_index = self.torchIntTensor([lines_graph_v1, lines_graph_v2] )
                print(len(lines_graph_v1))
                print('graph creation time:', time.time() - t)

                t = time.time()

                #SEMANTIC FEATURES
                semantic_features = self.Linear1(feature[k]).view(256,128)
                semantic_features = self.Linear2(semantic_features).view(256,16)
                semantic_features = self.conv1d1(semantic_features).view(1000, 16)

                #GEOMETRIC FEATURES
                centre_features = np.zeros((len(lines_for_train), 2)).astype(np.float32)
                length_features = np.zeros((len(lines_for_train), 1)).astype(np.float32)
                angle_features = np.zeros((len(lines_for_train), 1)).astype(np.float32)

                for i, (v0, v1) in enumerate(lines_for_train):
                    v = (v0 + v1) / 2
                    vint = self.to_int(v)
                    centre_features[i] = vint
                    length_features[i] = math.sqrt(sum((v0 - v1) ** 2)) / 2

                    if v0[0] <= v[0]:
                        vv = v0
                    else:
                        vv = v1

                    if math.sqrt(sum((vv - v) ** 2)) <= 1e-4:
                        continue
                    angle_features[i] = sum((-vv + v).detach().numpy() * np.array([0., 1.])) / math.sqrt(
                        sum((vv - v) ** 2))

                centre_features = centre_features / 128
                length_features = length_features / 128
                print('Geometrics time:', time.time() - t)
                #geometric_features = self.torchcat((self.torch_from_numpy(centre_features),self.torch_from_numpy(length_features), self.torch_from_numpy(angle_features)),axis=1)
                geometric_features = self.Linear3(self.torchcat((self.torch_from_numpy(centre_features).cuda(),self.torch_from_numpy(length_features).cuda(), self.torch_from_numpy(angle_features).cuda()),axis=1))

                graph_features = self.torchcat((semantic_features,geometric_features),axis=1)

                #print('semantic features:',semantic_features.device)
                #print('edge index cpu:',self.edge_index.cpu().device)
                #print('edge index cuda:',self.edge_index.cuda().device)
                graph_out = self.conv1(graph_features, edge_index.cuda())


                print('GCN time:', time.time() - t)

                #CENTRE MAP
                t = time.time()
                lcmap_graph = self.conv1d2(graph_out).view(128, 32)
                lcmap_graph = self.Linear4(lcmap_graph)
                lcmap_graph = self.frelu(lcmap_graph)
                lcmap_graph = lcmap_graph.view(128,32)
                lcmap_k = self.torchcat((lcmap[k],lcmap_graph),axis=1)
                lcmap[k] = self.Linear5(lcmap_k)
                print('GCN-lcmap time:', time.time() - t)

                #OFFSET
                t = time.time()
                lcoff_graph = self.conv1d3(graph_out).view(2, 128, 32)
                lcoff_graph = self.Linear4(lcoff_graph)
                lcoff_k = self.torchcat((lcoff[k], lcoff_graph), axis=2)
                lcoff[k] = self.Linear5(lcoff_k)
                print('GCN-lcoff time:', time.time() - t)

                #LENGTH
                t = time.time()
                lleng_graph = self.conv1d2(graph_out).view(128, 32)
                lleng_graph = self.Linear4(lleng_graph)
                lleng_graph = self.frelu(lleng_graph)
                lleng_graph = lleng_graph.view(128, 32)
                lleng_k = self.torchcat((lleng[k], lleng_graph), axis=1)
                lleng[k] = self.Linear5(lleng_k)
                print('GCN-length time:', time.time() - t)

                #ANGLE
                t = time.time()
                angle_graph = self.conv1d2(graph_out).view(128, 32)
                angle_graph = self.Linear4(angle_graph)
                angle_graph = self.frelu(angle_graph)
                angle_graph = angle_graph.view(128, 32)
                angle_k = self.torchcat((angle[k], angle_graph), axis=1)
                angle[k] = self.Linear5(angle_k)
                print('GCN-angle time:', time.time() - t)



            losses.append(L)
            accuracy.append(Acc)

            if stack == 0 and input_dict["do_evaluation"]:
                result["heatmaps"] = heatmap

        result["losses"] = losses
        result["accuracy"] = accuracy

        return result


