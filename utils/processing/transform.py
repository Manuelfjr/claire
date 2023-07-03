# from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm


class TransformPairwise:
    def __init__(self, data=None, works=-1):
        self.data = data
        self.works = works

    def generate_pij_matrix(self, data=None):
        self.transformed_matrix = pd.DataFrame(np.nan, columns=self.data.columns, index=range(self.data.shape[0]))
        i_line = 0
        for idx1, i in tqdm(enumerate(self.data.values)):
            j_line = 0
            item_i = list(i)

            tmp = self.data.drop(idx1, axis=0)

            boolean_values = []
            for _, j in enumerate(tmp.values):
                model_j = list(j)
                boolean_values.append([k == l for k, l in zip(item_i, model_j)])
            data_boolean = pd.DataFrame(boolean_values)

            for idx3, b in enumerate(data_boolean.T.values):
                tmp2 = data_boolean.drop(idx3, axis=1)

                self.transformed_matrix.iloc[i_line, j_line] = pd.DataFrame(
                    [list(b == m) for m in tmp2.T.values]
                ).T.sum().sum() / ((self.data.shape[1] - 1) * (self.data.shape[0] - 1))

                j_line += 1
                # break
            i_line += 1
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
