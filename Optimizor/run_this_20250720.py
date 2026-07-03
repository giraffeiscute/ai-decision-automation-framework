"""
AED_deployment_model
Author: Kexin Cao
Institute: Tsinghua University
Date: 20241220 
"""

import Data_20250720 as Data
import ModelBuilder_20250720 as ModelBuilder
from gurobipy import *
import random
import numpy as np
import pandas as pd
import csv

if __name__ == "__main__":
    
    file_name = 'test_poi_df_NNtotal_20250415.csv'
    file_name_ohca = 'ohca_df.csv'

    # 設定想要測試的部署點數列表
    # loc_num = 5
    loc_num_list = [5,10,20,40,60,80,100]
    # loc_num_list = [100]

    #AED覆蓋範圍
    dist_limit = 0.96

    # AED之間的距離限制
    # dist_list = [0.6,0.8,0.96,1.2,0]
    # dist_list = [0.96]
    dist_list = [0,0.6,0.8,0.96,1,1.2,1.4,1.6]
    # dist_list = [0.8, 0.96, 1, 1.2]

    # 建築物總數量，和隨機選擇的數量
    build_num = 5000
    build_num_choose_cnt = 10 # 重複次數（隨機不同選擇的次數）

    # 總建築物數
    build_sum = 99724

    # OHCA 覆蓋率隨機評估次數
    ohca_cover_random_cnt = 10
    

    # random_candidate_loc_id = []
    # for cnt in range(build_num_choose_cnt):
    #     random_candidate_loc_id.append(random.choices(range(build_sum), k=build_num))
    # df = pd.DataFrame(random_candidate_loc_id).T
    # output_name = 'build_num_choose_' + str(build_num) + '.csv'
    # df.to_csv(output_name, index=False)    


    for dist in dist_list:
        data = Data.Data()
        # data.read_data(file_name,file_name_ohca,loc_num=loc_num,dist_limit=dist_limit,build_num=build_num)   
        data.read_npy(file_name,file_name_ohca,dist_limit=dist,build_num=build_num)   #對不同的距離限制建構物件
        print('read finishing !')
        # data = np.load('indicator_i_j.npy')

        #對不同部署點數loc_num初始化輸出框架
        for loc_num in loc_num_list:
            # 載入隨機候選位置的ID集合
            output_name = 'build_num_choose_100_set_' + str(build_num_choose_cnt) + '.npy'
            random_candidate_loc_id = np.load(output_name).astype(np.int64)
            # 初始化IP-SHAP結果陣列，用於存放不同次數run的結果
            deploy_decision_coverage_shap = np.zeros(build_num_choose_cnt)
            deploy_decision_shap = np.zeros((build_num_choose_cnt, loc_num))
            deploy_decision_survivalrate_shap = np.zeros((build_num_choose_cnt,len(data.ohca_lat)))
            deploy_decision_survivalrate_average_shap = np.zeros(build_num_choose_cnt)
            ## 初始化IP-MLP結果陣列，用於存放不同次數run的結果
            deploy_decision_coverage_mlp = np.zeros(build_num_choose_cnt)
            deploy_decision_mlp = np.zeros((build_num_choose_cnt, loc_num))
            deploy_decision_survivalrate_mlp = np.zeros((build_num_choose_cnt,len(data.ohca_lat)))
            deploy_decision_survivalrate_average_mlp = np.zeros(build_num_choose_cnt)

            for cnt in range(build_num_choose_cnt):
                model_handler = ModelBuilder.ModelBuilder()
                candidate_loc_id = random_candidate_loc_id[cnt]

                ##%% IP-SHAP 模型求解 + 计算覆盖率
                print('IP-SHAP-dist-',dist,'cnt-',cnt)
                model_handler.build_IP(data,candidate_loc_id,loc_num)

                # 印出模型結果與運算時間
                print('{} = {}'.format('obj_val', model_handler.obj_val), end='')
                print()
                print('{} = {}'.format('run_time', model_handler.run_time), end='')
                print()
                print('{} = {}'.format('deployment_decision', model_handler.deploy_decision), end='')
                print()
                deploy_decision_shap[cnt] = np.array(model_handler.deploy_decision)  # 保存決策結果  
                
                ohca_cover_cnt_predict = 0
                # 計算每個OHCA點是否被部署點覆蓋，並計算存活率
                for ohca in range(len(data.ohca_lat)):
                    iscover = 0
                    min_dist = np.inf
                    for i in range(len(model_handler.deploy_decision)):
                        loc = model_handler.deploy_decision[i]
                        # 計算部署點與OHCA點距離 判斷AED是否覆蓋
                        dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
                        if dist_loc_ohca <= min_dist:
                            min_dist = dist_loc_ohca
                        if dist_loc_ohca <= dist_limit:
                            iscover = 1
                    #計算生存機率
                    t_loc_ohca = min_dist / 300 * 1000
                    # 根據公式計算生存率
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  
                    deploy_decision_survivalrate_shap[cnt][ohca] = s_loc_ohca   # 記錄生存機率
                    if iscover == 1:
                        ohca_cover_cnt_predict += 1
                deploy_decision_coverage_shap[cnt] = ohca_cover_cnt_predict # 記錄覆蓋數量
                print(deploy_decision_survivalrate_shap[cnt])
                deploy_decision_survivalrate_average_shap[cnt] = np.average(deploy_decision_survivalrate_shap[cnt])   

                ## IP-MLP 模型求解 + 覆蓋率計算（與上面類似）
                print('IP-MLP-dist-', dist, 'cnt-', cnt)
                model_handler.build_IP_mlp(data,candidate_loc_id,loc_num)

                # 印出模型結果與運算時間
                print('{} = {}'.format('obj_val', model_handler.obj_val_mlp), end='')
                print()
                print('{} = {}'.format('run_time', model_handler.run_time_mlp), end='')
                print()
                print('{} = {}'.format('deployment_decision', model_handler.deploy_decision_mlp), end='')
                print()
                deploy_decision_mlp[cnt] = np.array(model_handler.deploy_decision_mlp)   
                ohca_cover_cnt_predict = 0
                for ohca in range(len(data.ohca_lat)):
                    iscover = 0
                    min_dist = np.inf
                    for i in range(len(model_handler.deploy_decision_mlp)):
                        loc = model_handler.deploy_decision_mlp[i]
                        # 計算部署點與OHCA點距離 判斷AED是否覆蓋
                        dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
                        if dist_loc_ohca <= min_dist:
                            min_dist = dist_loc_ohca
                        if dist_loc_ohca <= dist_limit:
                            iscover = 1

                    #計算生存機率
                    t_loc_ohca = min_dist / 300 * 1000
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  
                    deploy_decision_survivalrate_mlp[cnt][ohca] = s_loc_ohca   # 記錄生存機率
                    if iscover == 1:
                        ohca_cover_cnt_predict += 1   
                deploy_decision_coverage_mlp[cnt] = ohca_cover_cnt_predict # 記錄覆蓋數量
                print(deploy_decision_survivalrate_mlp[cnt])
                deploy_decision_survivalrate_average_mlp[cnt] = np.average(deploy_decision_survivalrate_mlp[cnt])
            
 
            output_folder = 'results_mlp_20250720/'
            ##%% np.save for IP-SHAP
            output_name = output_folder+'shap/deployment_decision_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt)  + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_shap)   
            output_name = output_folder+'shap/deployment_decision_coverage_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_coverage_shap) 
            output_name = output_folder+'shap/deploy_decision_survivalrate_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_shap)
            output_name = output_folder+'shap/deploy_decision_survivalrate_average_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_average_shap)

            ##%% np.save for IP-MLP
            output_name = output_folder+'mlp/deployment_decision_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt)  + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_mlp)   
            output_name = output_folder+'mlp/deployment_decision_coverage_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_coverage_mlp) 
            output_name = output_folder+'mlp/deploy_decision_survivalrate_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_mlp)
            output_name = output_folder+'mlp/deploy_decision_survivalrate_average_' + str(loc_num)+ '_set_' + str(build_num_choose_cnt) + '_dist_' + str(dist) + '.npy'
            np.save(output_name,deploy_decision_survivalrate_average_mlp)
                                
            # df = pd.DataFrame(deploy_decision).T
            # output_name = 'deploy_decision_' + str(build_num) + '.csv'
            # df.to_csv(output_name, index=False) 
            print('model finishing _ ',loc_num) 
            
            # ohca_cover_cnt_predict = 0
            # # ohca_cover_cnt_random = 0
            # random_numbers = {}
            # for i in range(100):
            #     random_numbers[i] = random.choices(range(build_num), k=loc_num)
            # ohca_cover_cnt_random = {}
    '''
  (1 + e^{-0.26 + 0.106 \cdot t_{\text{aed}} + 0.139 \cdot t_{\text{cpr}}})^{-1}, & t_{\min} < 4, \\
    0, & t_{\min} \geq 4.
'''        

    # for ohca in range(len(data.ohca_lat)):
    #     iscover = 0
    #     for i in range(len(model_handler.deploy_decision)):
    #         loc = model_handler.deploy_decision[i]
    #         dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
    #         if dist_loc_ohca <= dist_limit:
    #             iscover = 1
    #     if iscover == 1:
    #         ohca_cover_cnt_predict += 1 
            
            
    # for j in range(100):
    #     ohca_cover_cnt_random[j] = 0
    #     for ohca in range(len(data.ohca_lat)):
    #         iscover = 0
    #         for i in range(len(random_numbers[j])):
    #             loc = random_numbers[j][i]
    #             dist_loc_ohca = data.haversine(data.loc_lat[loc],data.loc_lon[loc],data.ohca_lat[ohca],data.ohca_lon[ohca])
    #             if dist_loc_ohca <= dist_limit:
    #                 iscover = 1
    #         if iscover == 1:
    #             ohca_cover_cnt_random[j] += 1
    # print('{} : {}'.format('ohca_cover_cnt_predict', ohca_cover_cnt_predict), end='')
    # print()
    # print('{} : {}'.format('ohca_cover_cnt_random', ohca_cover_cnt_random), end='')
    # print()
        
    

