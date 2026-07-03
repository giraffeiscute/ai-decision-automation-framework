import Data_20250720 as Data
import numpy as np
import os

if __name__ == "__main__":
    
    # 檔案名稱設定
    file_name = 'test_poi_df_NNtotal_20250415.csv'   
    file_name_ohca = 'ohca_df.csv'
    
    # 要測試的AED設置數量列表
    loc_num_list = [5, 10, 20, 40, 60, 80, 100]
    
    # 距離限制(公里)
    dist_limit = 0.96
    
    # 建築物數量
    build_num = 5000
    
    # 建築物選擇集合數量
    build_num_choose_cnt = 10
    
    # 每個集合的隨機覆蓋測試次數
    ohca_cover_random_cnt = 10
    
    # 輸出資料夾名稱
    output_folder = 'results_random_R+_20251003'
    
    # 如果資料夾不存在，則建立它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'已建立資料夾: {output_folder}')

    # 讀取資料一次即可
    data = Data.Data() 
    data.read_npy(file_name, file_name_ohca, dist_limit=dist_limit, build_num=build_num)   
    print('資料讀取完成！')
    
    # 針對每個AED設置數量進行計算
    for loc_num in loc_num_list:  
        # 初始化結果陣列
        random_choose_coverage = np.zeros((build_num_choose_cnt, ohca_cover_random_cnt))  # 覆蓋率
        random_choose_survivalrate_average = np.zeros((build_num_choose_cnt, ohca_cover_random_cnt))  # 平均生存率
        random_choose_survivalrate_nonzero_average = np.zeros((build_num_choose_cnt, ohca_cover_random_cnt))  # 非零生存率平均
        
        # 針對每個建築物選擇集合
        for cnt in range(build_num_choose_cnt):
            # 直接讀取已存在的隨機選擇地點檔案
            input_name = 'results_random_20250822/random_choose_loc_id_' + str(loc_num) + '_set_' + str(cnt) + '.npy'
            random_choose_loc_id = np.load(input_name).astype(np.int64)
            
            # 記錄每次隨機選擇的覆蓋數量
            ohca_cover_cnt_random = {}
            
            # 針對每次隨機覆蓋測試
            for i in range(ohca_cover_random_cnt):
                # 初始化每個OHCA點的生存率陣列
                random_choose_survivalrate = np.zeros(len(data.ohca_lat))
                ohca_cover_cnt_random[i] = 0
                
                # 計算每個OHCA點的覆蓋情況和生存率
                for ohca in range(len(data.ohca_lat)):
                    iscover = 0  # 是否被覆蓋
                    min_dist = np.inf  # 最小距離初始化為無限大
                    
                    # 找出距離此OHCA點最近的AED位置
                    for id in range(len(random_choose_loc_id[i])):
                        loc = int(random_choose_loc_id[i][id])
                        # 計算AED位置與OHCA點之間的距離
                        dist_loc_ohca = data.haversine(data.loc_lat[loc], data.loc_lon[loc], 
                                                       data.ohca_lat[ohca], data.ohca_lon[ohca])
                        # 更新最小距離
                        if dist_loc_ohca <= min_dist:
                            min_dist = dist_loc_ohca                        
                        # 判斷是否在覆蓋範圍內
                        if dist_loc_ohca <= dist_limit:
                            iscover = 1
                    
                    # 計算反應時間(分鐘) = 距離(公里) / 速度(300公尺/分鐘) * 1000
                    t_loc_ohca = min_dist / 300 * 1000
                    
                    # 根據公式計算生存率
                    if t_loc_ohca <= 4:
                        s_loc_ohca = (1 + np.exp(-0.26 + 0.106 * t_loc_ohca + 0.139 * 10.5))**(-1)
                    else:
                        s_loc_ohca = 0  # 超過4分鐘生存率為0
                    
                    random_choose_survivalrate[ohca] = s_loc_ohca      
                    
                    # 統計被覆蓋的OHCA點數量
                    if iscover == 1:
                        ohca_cover_cnt_random[i] += 1
                
                # 記錄此次隨機測試的結果
                random_choose_coverage[cnt][i] = ohca_cover_cnt_random[i]  # 覆蓋數量
                random_choose_survivalrate_average[cnt][i] = np.average(random_choose_survivalrate)  # 所有點的平均生存率
                
                # 計算非零生存機率的平均值
                nonzero_survivalrate = random_choose_survivalrate[random_choose_survivalrate > 0]
                if len(nonzero_survivalrate) > 0:
                    random_choose_survivalrate_nonzero_average[cnt][i] = np.average(nonzero_survivalrate)
                else:
                    random_choose_survivalrate_nonzero_average[cnt][i] = 0
        
        # 儲存結果到檔案
        # 儲存覆蓋率結果
        output_name = f'{output_folder}/random_choose_coverage_{loc_num}_set_{build_num_choose_cnt}'
        np.save(output_name, random_choose_coverage) 
        
        # 儲存平均生存率結果
        output_name = f'{output_folder}/random_choose_survivalrate_average_{loc_num}_set_{build_num_choose_cnt}'
        np.save(output_name, random_choose_survivalrate_average)
        
        # 儲存非零生存機率平均值結果
        output_name = f'{output_folder}/random_choose_survivalrate_nonzero_average_{loc_num}_set_{build_num_choose_cnt}'
        np.save(output_name, random_choose_survivalrate_nonzero_average)
        
        print(f'已完成 loc_num = {loc_num} 的計算')