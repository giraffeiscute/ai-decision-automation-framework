"""
AED_deployment_model
Author: Kexin Cao
Institute: Tsinghua University
Date: 20241220 
"""


from gurobipy import *
import time

class ModelBuilder(object):
    def __init__(self):
        self.data = None
        
        self.x_i = {}              # IP 模型中的決策變數 x_i
        self.cons_dist = {}        # IP 模型中的相鄰地點限制約束

        self.deploy_decision = []  # IP 模型選出來的地點
        
        self.obj_val = 0           # IP 模型目標值
        self.run_time = 0          # IP 模型執行時間
        
        ##%% for self.IP_MLP_model
        self.x_i_mlp = {}
        self.cons_dist_mlp = {}

        self.deploy_decision_mlp = []
        
        self.obj_val_mlp = 0
        self.run_time_mlp = 0
                
    def build_IP(self, data, candidate_loc_id, loc_num):
        """
        建構並求解傳統整數規劃模型。
        參數：
        - data: 包含 loc_score 與 indicator_i_j 的資料物件
        - candidate_loc_id: 候選地點 ID 清單
        - loc_num: 可選擇的地點數量上限
        """
        start_time = time.time()
        self.data = data
        self.IP_model = Model('IP_model')  # 建立 Gurobi 模型
        obj = LinExpr()  # 目標函數初始化
        lhs = LinExpr()  # 限制式左邊初始化

        # 建立每個候選地點的二元決策變數與目標函數
        for loc_id in candidate_loc_id:
            i = loc_id
            self.x_i[i] = self.IP_model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x_' + str(i))
            obj.addTerms(data.loc_score[i], self.x_i[i])  # 加入該地點的打分
            lhs.addTerms(1, self.x_i[i])  # 用於地點數量限制式
            # if i % 200 == 0:
            #     print(i) 
        self.IP_model.setObjective(obj, GRB.MAXIMIZE) # 最大化總得分

        # 若有距離限制（infinite == 0），則加入相鄰地點不可同時選擇的限制式
        if data.infinite == 0:
            for i in candidate_loc_id:
                for j in candidate_loc_id:
                    if j > i and data.indicator_i_j.get((i, j), 0) == 1: # 代表這兩個地點距離過近，根據距離限制 (indicator_i_j[i,j] == 1)，只能選擇其中一個來部署
                        self.cons_dist[i, j] = self.IP_model.addConstr(
                            self.x_i[i] + self.x_i[j] <= 1,  # 為地點 i 與地點 j 加上一條限制式：x_i + x_j ≤ 1
                            name='cons_dist' + '_' + str(i) + '_' + str(j) # 限制式名稱為 'cons_dist_i_j'，方便未來模型檢查或調試使用
                        )
            # if i % 200 == 0:
            #     print(i) 
        
        # 加入總選擇地點數量不能超過 loc_num 的限制式
        self.cons_sum = self.IP_model.addConstr(lhs <= loc_num, name = 'cons_sum')

        # 設定 Gurobi 求解參數
        self.IP_model.setParam('OutputFlag', 1)  # 開啟求解日誌
        self.IP_model.setParam('Presolve', 1)   # 啟用預處理
        # self.IP_model.setParam('Threads', 16)       # 设置使用16个线程
        # self.IP_model.setParam('MIPFocus', 2)       # 关注求解器的节省时间性能
        # self.IP_model.setParam('Presolve', 2)       # 开启预求解
        # self.IP_model.setParam('Cuts', 2)           # 强化切割平面
        # self.IP_model.setParam('Heuristics', 0.5)   # 调高启发式的方法比例
        # self.IP_model.setParam('ImproveStartTime', 1200) # 设置改善开始时间

        # 計算模型構建時間
        finish_time = time.time() - start_time 
        print('model construction time:',finish_time)

        # # 設定求解時間限制
        self.IP_model.setParam('TimeLimit', 3600*3)

        # 求解模型
        self.IP_model.optimize()  
        # self.IP_model.write('IP_model.lp')

        # 儲存結果
        self.obj_val = self.IP_model.ObjVal
        self.run_time = self.IP_model.run_time
        for loc_id in candidate_loc_id:
            if self.x_i[loc_id].x >= 0.5:
                self.deploy_decision.append(loc_id)  # 將選中的地點加入決策清單

    def build_IP_mlp(self, data, candidate_loc_id, loc_num): ##%% 建IP-MLP模型
        """
        建構並求解使用 MLP 打分的整數規劃模型。
        參數與傳統版相同，但使用 data.loc_score_mlp 作為目標函數的依據。
        """
        start_time = time.time()
        self.data = data
        self.IP_MLP_model = Model('IP_MLP_model')
        obj = LinExpr()
        lhs = LinExpr()

        # 建立決策變數與目標函數（使用 MLP 打分）
        for loc_id in candidate_loc_id:
            i = loc_id
            self.x_i_mlp[i] = self.IP_MLP_model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = 'x_' + str(i))
            obj.addTerms(data.loc_score_mlp[i], self.x_i_mlp[i])
            lhs.addTerms(1, self.x_i_mlp[i])
            # if i % 200 == 0:
            #     print(i) 
        self.IP_MLP_model.setObjective(obj, GRB.MAXIMIZE)   

        # 若需距離限制，則同樣加入不可同時選擇的限制式 
        if data.infinite == 0:
            for i in candidate_loc_id:
                for j in candidate_loc_id:
                    if j > i and data.indicator_i_j[i, j] == 1:  # 改用 numpy 的索引方式
                        self.cons_dist_mlp[i, j] = self.IP_MLP_model.addConstr(
                            self.x_i_mlp[i] + self.x_i_mlp[j] <= 1,  # 為地點 i 與地點 j 加上一條限制式：x_i + x_j ≤ 1
                            name=f'cons_dist_{i}_{j}'  # 限制式名稱為 'cons_dist_i_j'，方便未來模型檢查或調試使用
                        )
            # if i % 200 == 0:
            #     print(i) 
        
        # 加入地點數量限制
        self.cons_sum = self.IP_MLP_model.addConstr(lhs <= loc_num, name = 'cons_sum')
        # 設定 Gurobi 參數
        self.IP_MLP_model.setParam('OutputFlag', 1) 
        self.IP_MLP_model.setParam('Presolve', 1)   
        # self.IP_model.setParam('Threads', 16)       # 设置使用16个线程
        # self.IP_model.setParam('MIPFocus', 2)       # 关注求解器的节省时间性能
        # self.IP_model.setParam('Presolve', 2)       # 开启预求解
        # self.IP_model.setParam('Cuts', 2)           # 强化切割平面
        # self.IP_model.setParam('Heuristics', 0.5)   # 调高启发式的方法比例
        # self.IP_model.setParam('ImproveStartTime', 1200) # 设置改善开始时间

        # 模型構建時間
        finish_time = time.time() - start_time 
        print('model construction time:',finish_time)
        # 設定最大求解時間
        self.IP_MLP_model.setParam('TimeLimit', 3600*3)
        self.IP_MLP_model.optimize()  
        # self.IP_model.write('IP_model.lp')

        # 儲存求解結果
        self.obj_val_mlp = self.IP_MLP_model.ObjVal
        self.run_time_mlp = self.IP_MLP_model.run_time
        for loc_id in candidate_loc_id:
            if self.x_i_mlp[loc_id].x >= 0.5:
                self.deploy_decision_mlp.append(loc_id) 
        
                   
                    
                    
                    
                    
                    
                    
                    