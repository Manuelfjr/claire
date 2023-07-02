from multiprocessing import Pool
import numpy as np
from datetime import datetime 
from tqdm import tqdm
import pandas as pd


class TransformPairwise:
    def __init__(self, data=None, works=-1):
        self.data = data
        self.works = works

    def generate_pij_matrix(self, data=None):
        n = self.data.shape[0]
        m = self.data.shape[1]
        data_final = pd.DataFrame(0, columns=self.data.columns, index=self.data.index)
        for item in self.data.index:
            tmp = pd.DataFrame(index=self.data.index, columns=self.data.columns)
            columns = []

            for model in self.data.columns:
                tmp[model] = (self.data[model] == self.data[model][item]).replace({True: 1, False: 0}).values

            info = []
            for model_binary in tmp.columns:
                tmp_base = tmp[[model_binary]].copy()
                tmp2 = tmp[[i for i in tmp.columns if model_binary != i]]
                d = []
                for compare in tmp2.columns:
                    d.append((tmp_base[model_binary] == tmp2[compare]).sum() - 1)
                info.append(sum(d) / ((n - 1) * (m - 1)))
                # print(tmp2)
            self.data.iloc[item, :] = info
        return self.data

    def combination_models(self, models, params):
        return [models[params_model](**arg) for params_model in params.keys() for arg in params[params_model]]



class TransformPairwise2:
    def __init__(self, data=None, works=-1):
        self.data = data
        self.works = works

    def generate_pij_matrix(self, data=None):
        self.transformed_matrix = pd.DataFrame(np.nan, columns = self.data.columns, index = range(self.data.shape[0]))
        i_line = 0
        for idx1, i in tqdm(enumerate(self.data.values)):
            j_line = 0
            item_i = list(i)
            
            tmp = self.data.drop(idx1, axis = 0)

            boolean_values = []
            for _, j in enumerate(tmp.values):
                model_j = list(j)
                boolean_values.append([k == l for k,l in zip(item_i, model_j)])
            data_boolean = pd.DataFrame(boolean_values)

            for idx3, b in enumerate(data_boolean.T.values):
                tmp2 = data_boolean.drop(idx3, axis=1)

                self.transformed_matrix.iloc[i_line, j_line] = pd.DataFrame([list(b == m) for m in tmp2.T.values]).T.sum().sum()/((self.data.shape[1] - 1)*(self.data.shape[0]-1))

                j_line += 1
                #break
            i_line +=  1
        return self.transformed_matrix


    def combination_models(self, models, params):
        return [models[params_model](**arg) for params_model in params.keys() for arg in params[params_model]]



# n, m = 300, 5
# k = 3
# data = pd.DataFrame(columns = [f'c{i}' for i in range(m)])
# data = pd.DataFrame({
#     "m1": [0,1,0,0,1,1],
#     "m2": [1,1,1,2,1,2],
#     "m3": [1,2,1,2,2,2]
# })
# print(data)
# print()
# # for i in range(m):
# #     values = np.random.randint(k, size = n)
# #     data[f"c{i}"] = values

# # init_time = datetime.now()
# tp = TransformPairwise2(data)
# transformed = tp.generate_pij_matrix()
# # final_time = datetime.now()
# # interval_time = final_time - init_time
# # print("new method: {}".format(interval_time))
# print(transformed)

#init_time = datetime.now()
#tp = TransformPairwise(data)
#transformed = tp.generate_pij_matrix()
#final_time = datetime.now()
#interval_time = final_time - init_time
#print(interval_time)

# for idx_instance, j in enumerate(data.values):
#     item_j = list(j)#pd.DataFrame(list(j), index = data.columns).T
#     tmp = data.drop(idx_instance, axis = 0).copy()
#     data_boolean_j = pd.DataFrame([list(item_j == k) for k in tmp.values])
#     values = 0
#     for idx_model, i in enumerate(data_boolean_j.T.values):
#         tmp = data_boolean_j.drop(idx_model, axis = 1).copy()
#         model_i = list( i )
#         models_agreement = pd.DataFrame([list(model_i & k) for k in tmp.T.values]).sum(axis=0)
#         values = values + models_agreement
#         tmp = data_boolean_j.copy()
#     print(values)
#     values = 0
#     print(data_boolean_j)


    #print(tmp.index[tmp.apply(lambda row: row.tolist() == item_j, axis=1)])
    #compare_matrix = np.apply_along_axis(lambda x:  [x == i for i in tmp.values], axis=0, arr=tmp.values)
    #print(compare_matrix)
    #print(np.apply_along_axis(lambda x: [x == i for i in item_j], axis=1, arr=tmp.values))

    # print("s")
    # break
    # tmp = data.copy()
# for j in data.T.values:
#     for idx, i in enumerate(j):
#         item_j = list(j)
#         item_j.pop(idx)
#         item_j = [int(k) for k in i == item_j]
#         #print(len(j), len(item_j))
#         print(item_j)

#         #print(i == item_j)
#         break
#     break

#init_time = datetime.now()
#tp = TransformPairwise(data)
#transformed = tp.generate_pij_matrix()
#final_time = datetime.now()
#interval_time = final_time - init_time
#print(interval_time)

    # def transform(self):
    #     with Pool(self.works) as pool:
    #         result = pool.map(self._generate_pij_matrix, [self.data])
    #     return result
