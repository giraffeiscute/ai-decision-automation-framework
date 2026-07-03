"""
AED_deployment_model - 計算非零生存率平均值
Author: Kexin Cao (Modified)
Institute: Tsinghua University
Date: 20251003
"""

import Data_without_dmin_20251003 as Data
import numpy as np
import os

if __name__ == "__main__":
    
    # 檔案名稱設定
    file_name = 'test_poi_df_XGBtotal_20250415.csv'
    file_name_ohca = 'ohca_df.csv'

    # 要測試的AED設置數量列表
    loc_num_list = [5, 10, 20, 40, 60, 80, 100]

    # AED之間的距離限制列表
    dist_list = [0, 0.6, 0.8, 0.96, 1, 1.2, 1.4, 1.6]

    # 建築物總數量
    build_num = 5000

    # 重複次數（隨機不同選擇的次數）
    build_num_choose_cnt = 10

    # 輸入資料夾
    input_folder = 'results_xgb_20250415/'


    # 對每個距離限制進行處理
    for dist in dist_list:
        # 讀取資料
        data = Data.Data()
        data.read_npy(file_name, file_name_ohca, build_num=build_num)
        print(f'資料讀取完成！距離限制 = {dist}')

        # 對不同部署點數loc_num進行計算
        for loc_num in loc_num_list:
            
            # 初始化IP-SHAP結果陣列
            deploy_decision_survivalrate_shap = np.zeros((build_num_choose_cnt, len(data.ohca_lat)))
            deploy_decision_survivalrate_average_shap = np.zeros(build_num_choose_cnt)
            deploy_decision_survivalrate_nonzero_average_shap = np.zeros(build_num_choose_cnt)
            
            # 初始化IP-MLP結果陣列
            deploy_decision_survivalrate_mlp = np.zeros((build_num_choose_cnt, len(data.ohca_lat)))
            deploy_decision_survivalrate_average_mlp = np.zeros(build_num_choose_cnt)
            deploy_decision_survivalrate_nonzero_average_mlp = np.zeros(build_num_choose_cnt)

            for cnt in range(build_num_choose_cnt):
                
                ##%% IP-SHAP 模型 - 讀取部署決策並計算生存率
                print(f'處理 IP-SHAP - dist: {dist}, loc_num: {loc_num}, cnt: {cnt}')
                
                # 讀取已儲存的部署決策
                input_name = f'{input_folder}shap/deployment_decision_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
                deploy_decision_shap = np.load(input_name)
                deploy_decision = deploy_decision_shap[cnt].astype(int)  # 取出第cnt次的決策
                
                # 計算每個OHCA點的生存率
                for ohca in range(len(data.ohca_lat)):
                    min_dist = np.inf
                    
                    # 找出距離此OHCA點最近的AED位置
                    for loc in deploy_decision:
                        # 計算部署點與OHCA點距離
                        dist_loc_ohca = data.haversine(data.loc_lat[loc], data.loc_lon[loc],
                                                       data.ohca_lat[ohca], data.ohca_lon[ohca])
                        if dist_loc_ohca < min_dist:
                            min_dist = dist_loc_ohca
                    
                    # 計算反應時間(分鐘) = 距離(公里) / 速度(300公尺/分鐘) * 1000
                    t_loc_ohca = min_dist / 300 * 1000
                    
                    # 根據公式計算生存率
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  # 超過4分鐘生存率為0
                    
                    deploy_decision_survivalrate_shap[cnt][ohca] = s_loc_ohca
                
                # 計算所有點的平均生存率
                deploy_decision_survivalrate_average_shap[cnt] = np.average(deploy_decision_survivalrate_shap[cnt])
                
                # 計算非零生存率的平均值
                nonzero_survivalrate = deploy_decision_survivalrate_shap[cnt][deploy_decision_survivalrate_shap[cnt] > 0]
                if len(nonzero_survivalrate) > 0:
                    deploy_decision_survivalrate_nonzero_average_shap[cnt] = np.average(nonzero_survivalrate)
                else:
                    deploy_decision_survivalrate_nonzero_average_shap[cnt] = 0
                
                # print(f'SHAP 平均生存率: {deploy_decision_survivalrate_average_shap[cnt]:.4f}, '
                #       f'非零平均生存率: {deploy_decision_survivalrate_nonzero_average_shap[cnt]:.4f}')

                ##%% IP-MLP 模型 - 讀取部署決策並計算生存率
                print(f'處理 IP-MLP - dist: {dist}, loc_num: {loc_num}, cnt: {cnt}')
                
                # 讀取已儲存的部署決策
                input_name = f'{input_folder}mlp/deployment_decision_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
                deploy_decision_mlp = np.load(input_name)
                deploy_decision = deploy_decision_mlp[cnt].astype(int)  # 取出第cnt次的決策
                
                # 計算每個OHCA點的生存率
                for ohca in range(len(data.ohca_lat)):
                    min_dist = np.inf
                    
                    # 找出距離此OHCA點最近的AED位置
                    for loc in deploy_decision:
                        # 計算部署點與OHCA點距離
                        dist_loc_ohca = data.haversine(data.loc_lat[loc], data.loc_lon[loc],
                                                       data.ohca_lat[ohca], data.ohca_lon[ohca])
                        if dist_loc_ohca < min_dist:
                            min_dist = dist_loc_ohca
                    
                    # 計算反應時間(分鐘)
                    t_loc_ohca = min_dist / 300 * 1000
                    
                    # 根據公式計算生存率
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0
                    
                    deploy_decision_survivalrate_mlp[cnt][ohca] = s_loc_ohca
                
                # 計算所有點的平均生存率
                deploy_decision_survivalrate_average_mlp[cnt] = np.average(deploy_decision_survivalrate_mlp[cnt])
                
                # 計算非零生存率的平均值
                nonzero_survivalrate = deploy_decision_survivalrate_mlp[cnt][deploy_decision_survivalrate_mlp[cnt] > 0]
                if len(nonzero_survivalrate) > 0:
                    deploy_decision_survivalrate_nonzero_average_mlp[cnt] = np.average(nonzero_survivalrate)
                else:
                    deploy_decision_survivalrate_nonzero_average_mlp[cnt] = 0
                
                # print(f'MLP 平均生存率: {deploy_decision_survivalrate_average_mlp[cnt]:.4f}, '
                #       f'非零平均生存率: {deploy_decision_survivalrate_nonzero_average_mlp[cnt]:.4f}')

            ##%% 儲存 IP-SHAP 結果
            output_name = f'result_xgb_R+_20251013/shap/deploy_decision_survivalrate_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_shap)
            
            output_name = f'result_xgb_R+_20251013/shap/deploy_decision_survivalrate_average_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_average_shap)
            
            output_name = f'result_xgb_R+_20251013/shap/deploy_decision_survivalrate_nonzero_average_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_nonzero_average_shap)

            ##%% 儲存 IP-MLP 結果
            output_name = f'result_xgb_R+_20251013/mlp/deploy_decision_survivalrate_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_mlp)
            
            output_name = f'result_xgb_R+_20251013/mlp/deploy_decision_survivalrate_average_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_average_mlp)
            
            output_name = f'result_xgb_R+_20251013/mlp/deploy_decision_survivalrate_nonzero_average_{loc_num}_set_{build_num_choose_cnt}_dist_{dist}.npy'
            np.save(output_name, deploy_decision_survivalrate_nonzero_average_mlp)
            
            print(f'已完成 loc_num = {loc_num}, dist = {dist} 的計算並儲存結果')