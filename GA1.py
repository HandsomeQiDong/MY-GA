import numpy as np
import xlrd
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 10               #DNA长度 表示一个x或一个y
POP_SIZE = 200               #种群数量
CROSSOVER_RATE = 0.6        #交叉概率
MUTATION_RATE = 0.05        #变异概率
N_GENERATIONS = 1000         #迭代代数
X_BOUND = [0, 691]          #根据问题描述，最大横坐标691
Y_BOUND = [0, 317]          #根据问题描述，最大纵坐标317
N_demand_point = 32         #根据问题描述，需求点个数32
coord_fileroot = 'point coord.xlsx'
demand_fileroot = 'point demand.xlsx'
R = 20                  #服务点服务半径


def get_basic_data(coord_fileroot,demand_fileroot):
    X_coord = []            #记录需求点X坐标
    Y_coord = []            #记录需求点Y坐标
    demand_list = []        #记录需求点需求量
    coord_file = xlrd.open_workbook(coord_fileroot)
    coord_sheet = coord_file.sheet_by_index(0)
    X_coord = coord_sheet.col_values(1)     #读取需求点X坐标
    Y_coord = coord_sheet.col_values(2)     #读取需求点Y坐标
    demand_file = xlrd.open_workbook(demand_fileroot)
    demand_sheet = demand_file.sheet_by_index(0)
    demand_list = demand_sheet.col_values(1) #读取需求点需求值
    return X_coord,Y_coord,demand_list

def F(x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10):         #适应度函数
    total_demand = []
    X_coord, Y_coord, demand_list = get_basic_data(coord_fileroot,demand_fileroot)
    for i in range(POP_SIZE):               #遍历每一条DNA
        been_service = [[], [], [], [], []]
        demand_point_index = list(np.linspace(0, 31, 32))
        # copy_demand_point_index = demand_point_index
        all_d = 0
        a = [x1_10[i],y1_10[i],x2_10[i],y2_10[i],x3_10[i],y3_10[i],x4_10[i],y4_10[i],x5_10[i],y5_10[i]] #记录当前DNA中各服务点的xy坐标
        for j in range(0,10,2):                     #j表示服务点的下标
            x = a[j]
            y = a[j+1]
            for k in demand_point_index:         #k表示需求点的下标
                if k != ' ':
                    k = int(k)
                    d = pow((x-X_coord[k+1])**2+(y-Y_coord[k+1])**2,1/2)    #计算服务点j与需求点k的距离
                    if d <= R:
                        been_service[int(j/2)].append(k)
                        demand_point_index[k] = ' '            #需求点下标列表中的下标k改为空格
                    else:
                        continue
        for i in been_service:
            for j in i:
                all_d += demand_list[j+1]
        total_demand.append(all_d)
    return total_demand  # 输出每条DNA所表示的候选策略所满足的总需求的列表

def get_fitness(pop):       #获取适应度
    x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10 = translateDNA(pop)    #将编码转换成十进制
    pred = F(x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10)          #计算适应度函数
    best_fitness_in_history.append(max(pred))
    current_fitness.append(max(best_fitness_in_history))        #迄今为止的最优适应度函数
    index = pred.index(max(pred))                               #一代中最优适应度函数值的下标
    #以下计算当前代的平均适应度
    total = 0
    for i in pred:
        total += i
    ave = total/POP_SIZE
    ave_fitness.append(ave)                 #记录当前代的平均适应度到列表中

    best_gene = [x1_10[index],y1_10[index],x2_10[index],y2_10[index],x3_10[index],y3_10[index],x4_10[index],y4_10[index],x5_10[index],y5_10[index]]                                              #一代中最优个体的基因型(十进制)
    best_gene_in_history.append(best_gene)
    return pred     #返回适应度函数值

def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x1 = pop[:, 0:DNA_SIZE]
    y1 = pop[:, DNA_SIZE:DNA_SIZE*2]
    x2 = pop[:, DNA_SIZE * 2:DNA_SIZE*3]
    y2 = pop[:, DNA_SIZE * 3:DNA_SIZE*4]
    x3 = pop[:, DNA_SIZE * 4:DNA_SIZE*5]
    y3 = pop[:, DNA_SIZE * 5:DNA_SIZE*6]
    x4 = pop[:, DNA_SIZE * 6:DNA_SIZE * 7]
    y4 = pop[:, DNA_SIZE * 7:DNA_SIZE * 8]
    x5 = pop[:, DNA_SIZE * 8:DNA_SIZE * 9]
    y5 = pop[:, DNA_SIZE * 9:DNA_SIZE * 10]
    #二进制转十进制
    x1_10 = x1.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y1_10 = y1.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    x2_10 = x2.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y2_10 = y2.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    x3_10 = x3.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y3_10 = y3.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    x4_10 = x4.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y4_10 = y4.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    x5_10 = x5.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y5_10 = y5.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    return x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10



def single_fitness(pop1):
    pop = np.array([pop1],[pop1])
    print(pop)
    total_demand = []
    X_coord, Y_coord, demand_list = get_basic_data(coord_fileroot,demand_fileroot)
    x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10 = translateDNA(pop)
    been_service = [[], [], [], [], []]
    demand_point_index = list(np.linspace(0, 31, 32))
    # copy_demand_point_index = demand_point_index
    all_d = 0
    a = [x1_10[0], y1_10[0], x2_10[0], y2_10[0], x3_10[0], y3_10[0], x4_10[0], y4_10[0], x5_10[0],
         y5_10[0]]  # 记录当前DNA中各服务点的xy坐标
    for j in range(0, 10, 2):  # j表示服务点的下标
        x = a[j]
        y = a[j + 1]
        for k in demand_point_index:  # k表示需求点的下标
            if k != ' ':
                k = int(k)
                d = pow((x - X_coord[k + 1]) ** 2 + (y - Y_coord[k + 1]) ** 2, 1 / 2)  # 计算服务点j与需求点k的距离
                if d <= R:
                    been_service[int(j / 2)].append(k)
                    demand_point_index[k] = ' '  # 需求点下标列表中的下标k改为空格
                else:
                    continue
    for i in been_service:
        for j in i:
            all_d += demand_list[j + 1]
    total_demand.append(all_d)
    return total_demand

def crossover_and_mutation(pop, CROSSOVER_RATE= CROSSOVER_RATE):        #交叉和变异
    new_pop = []                                            #交叉变异后的种群
    for father in pop:                                      # 遍历种群中的每一个个体，将该个体作为父亲
        child = father                                      # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:                # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]        # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points1 = np.random.randint(low=0, high=DNA_SIZE * 10)  # 随机产生交叉的点
            child[cross_points1:] = mother[cross_points1:]    # 孩子得到位于交叉点后的母亲的基因
            ######多点交叉#####
            cross_points2 = np.random.randint(low=0, high=DNA_SIZE * 10)
            child[cross_points2:] = mother[cross_points2:]
            cross_points3 = np.random.randint(low=0, high=DNA_SIZE * 10)
            child[cross_points3:] = mother[cross_points3:]
            fitness_child = single_fitness(child)
            fitness_mother = single_fitness(mother)
            print(fitness_child,fitness_mother)
            ##################
        mutation(child)                                     # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop


def mutation(child, MUTATION_RATE = MUTATION_RATE):                       #变异
    ###################多点变异##############################
    x = np.random.rand(DNA_SIZE*10)                       #随机产生0-1随机数，长度与染色体长度对应
    for i in x:
        if i < MUTATION_RATE:                        # 以MUTATION_RATE的概率进行变异
            mutate_point =np.argwhere(x==i)           # 对应的数比MUTATION_RATE小，发生变异
            child[mutate_point] = child[mutate_point] ^ 1           # 将变异点的二进制为反转   “^”按位异或运算符


    ###################单点变异###########################
    # if np.random.rand() < MUTATION_RATE:                        # 以MUTATION_RATE的概率进行变异
    #     mutate_point = np.random.randint(0, DNA_SIZE*10)            # 随机产生一个实数，代表要变异基因的位置
    #     child[mutate_point] = child[mutate_point] ^ 1           # 将变异点的二进制为反转   “^”按位异或运算符

# ************************************************改到这里**********
def select(pop, fitness):                                       # 选择算子
    #######################锦标赛选择方法########################################
    temp_idx = []
    for i in range(POP_SIZE):
        temp_fitness = []
        temp_index = np.random.choice(np.arange(POP_SIZE),size=4,replace=False)
        for j in temp_index:
            temp_fitness = np.append(temp_fitness,fitness[j])
        temp_idx = np.append(temp_idx,temp_index[np.argmax(temp_fitness)])
    idx = temp_idx.astype('int64')

    #########################轮盘赌选择方法################################
    # idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE-1, replace=True,p=(fitness) / (fitness.sum()))#从np.arange(POP_SIZE)中按照概率分布p选择POP_SIZE个样本，可以有重复
    # idx = np.append(idx,np.argmax(fitness))
    return pop[idx] #返回选择结果


def print_info(pop):                                    #报文
    fitness = np.array(get_fitness(pop))                          #记录种群的适应度函数值
    max_fitness_index = np.argmax(fitness)              #获取fitness列表中最大适应度函数值的下标
    print("(最后一代最好个体)max_fitness:", fitness[max_fitness_index])   #打印最大适应度函数值
    x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10 = translateDNA(pop)                            #记录种群的决策变量
    print("(最后一代最好个体)最优的基因型：", pop[max_fitness_index])        #打印最优基因型（二进制）
    print("(最后一代最好个体)(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5):", (x1_10[max_fitness_index], y1_10[max_fitness_index]),
          (x2_10[max_fitness_index], y2_10[max_fitness_index]),(x3_10[max_fitness_index], y3_10[max_fitness_index]),
          (x4_10[max_fitness_index], y4_10[max_fitness_index]),(x5_10[max_fitness_index], y5_10[max_fitness_index]))#打印最优基因型（十进制）
    print("(历史最优个体)max_fitness：",max(best_fitness_in_history))
    print("(历史最有个体(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5)：",best_gene_in_history[best_fitness_in_history.index(max(best_fitness_in_history))])


if __name__ == "__main__":
    for i in range(5):
        best_fitness_in_history = []  # 历史上每代最优个体的适应度函数
        best_gene_in_history = []  # 历史上每代最优个体的基因型
        current_fitness = []  # 迄今为止最优个体的适应度函数
        ave_fitness = []  # 当前代的平均适应度函数
        pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 10))  # matrix (POP_SIZE, DNA_SIZE)DNA_SIZE是x或y的长度，POP_SIZE行DNA_size列
        for _ in range(N_GENERATIONS):  # 迭代N代
            x1_10,y1_10,x2_10,y2_10,x3_10,y3_10,x4_10,y4_10,x5_10,y5_10 = translateDNA(pop)            #将种群的所有DNA二进制转十进制
            pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
            fitness = get_fitness(pop)
            pop = select(pop, np.array(fitness))  # 选择生成新的种群
        print_info(pop)
        # fig = plt.figure()
        plt.plot(current_fitness)
        plt.plot(best_fitness_in_history)
        plt.plot(ave_fitness)
        plt.savefig("锦标赛选择+多点变异+多点交叉第"+str(i+1)+"次运行")
        plt.clf()
        # plt.show(block = True)                    #如果不注释掉，那么多次独立运行中，每完成一次独立运行，需要把弹出的图表关掉才能进行下一次独立运行